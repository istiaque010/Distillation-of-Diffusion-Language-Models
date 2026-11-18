"""Phase 8: train a student (one run of configs/distill.yaml).

One update:
    trajectory half (8 states):  student logits at the masked positions of x_a
                                 targets: 'traj'  -> hard teacher token on the span, teacher(x_b) soft on the rest
                                          'state' -> teacher(x_a) soft on every masked position
    real half (8 states):        student logits at the masked positions -> CE against the real tokens
    loss = lambda_KD * L_KD + (1 - lambda_KD) * L_CE ; AdamW, warm-up + cosine, gradient clipping 1.0.
The teacher runs under torch.no_grad() in eval mode; only the student's parameters are in the optimizer.
Logits are computed only at the masked positions (the output layer is ~1/4 of the model's compute).

Validation (fixed held-out states, identical in every run and at every check; step 0 = the teacher itself):
    val_traj      L_KD with T = 1 on 400 held-out trajectory states ('traj' targets; prompts never trained on)
    val_span_ce   its hard part: CE of the teacher's next-span tokens
    val_sched_ce  CE on 400 real validation windows masked with the student-pass quota (clean labels)
    val_ce/acc    Project 3's validation protocol (400 windows, uniform masking level): reconstruction quality

Run:  python -m distill.train_student --run kd      -> checkpoints/<run>/student.pt, results/metrics/train/<run>/
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import yaml

import distill
from distill import losses
from distill.models import count_parameters, load_teacher, make_student, save_student
from distill.states import RealStates, TrajectoryStates
from distill.trajectories import load as load_traj
from mdlm.data import StoryWindows
from mdlm.train import ValProtocol
from pardec.decode import hidden
from src.utils import env_info, peak_memory_mb, set_seed

COLS = ["step", "loss", "l_kd", "l_ce", "val_traj", "val_span_ce", "val_sched_ce", "val_ce", "val_acc", "lr",
        "grad_norm", "s_per_step", "mem_mb", "elapsed_s"]


def load_config(run: str) -> dict:
    cfg = yaml.safe_load(open(os.path.join(distill.CONFIGS, "distill.yaml"), encoding="utf-8"))
    return {**cfg["common"], **cfg["runs"][run], "run": run}


def lr_at(step, c):
    if step < c["warmup_steps"]:
        return c["lr"] * (step + 1) / c["warmup_steps"]
    p = min(1.0, (step - c["warmup_steps"]) / max(1, c["steps"] - c["warmup_steps"]))
    return c["min_lr"] + 0.5 * (1 + math.cos(math.pi * p)) * (c["lr"] - c["min_lr"])


def logits_at(model, x, where):
    """Model logits only at positions `where` (bool, B x W) -> (N, V)."""
    return model.head(hidden(model, x)[where])


def traj_loss(student, teacher, st, T, targets):
    """Per-position L_KD terms at the masked positions of a trajectory batch, in the order of st['masked']."""
    m = st["masked"]
    zs = logits_at(student, st["x"], m)
    if targets == "state":
        with torch.no_grad():
            zt = logits_at(teacher, st["x"], m)
        return losses.kd_soft(zs, zt, T), None
    with torch.no_grad():                             # teacher at its LATER state, only where still masked there
        zt = logits_at(teacher, st["x_later"], st["rest"])
    span = st["span"][m]                              # masked = span + rest, both in row-major order
    per = torch.empty(zs.shape[0])
    per[span] = losses.ce(zs[span], st["ids"][m][span])
    per[~span] = losses.kd_soft(zs[~span], zt, T)
    return per, span


def make_val(c, teacher_trajs):
    rng = np.random.default_rng(1234)
    n = c["eval_states"]
    vt = TrajectoryStates(teacher_trajs, c["budgets"]).sample(n, rng)
    pool = load_traj("train")["plen"]
    vr = RealStates("val", pool, c["budgets"]).sample(n, rng)
    vp = ValProtocol(n, 256, seed=1234)
    return vt, vr, vp


@torch.no_grad()
def evaluate(student, teacher, val, batch=50):
    vt, vr, vp = val
    student.eval()
    tr, sp, sp_n, n = 0.0, 0.0, 0, 0
    for i in range(0, len(vt["x"]), batch):
        st = {k: (v[i:i + batch] if torch.is_tensor(v) else v) for k, v in vt.items()}
        per, span = traj_loss(student, teacher, st, 1.0, "traj")
        tr += per.sum().item()
        n += per.numel()
        sp += per[span].sum().item()
        sp_n += int(span.sum())
    sc, sc_n = 0.0, 0
    for i in range(0, len(vr["x"]), batch):
        m = vr["masked"][i:i + batch]
        z = logits_at(student, vr["x"][i:i + batch], m)
        sc += losses.ce(z, vr["x0"][i:i + batch][m]).sum().item()
        sc_n += int(m.sum())
    p = vp.evaluate(student)
    student.train()
    return {"val_traj": tr / n, "val_span_ce": sp / sp_n, "val_sched_ce": sc / sc_n, "val_ce": p["val_loss"],
            "val_acc": p["val_acc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override (smoke tests)")
    a = ap.parse_args()
    c = load_config(a.run)
    if a.steps:
        c["steps"] = a.steps
    torch.set_num_threads(c["threads"])
    set_seed(c["seed"])
    teacher = load_teacher()
    student = make_student(teacher, c.get("layers"), c["dropout"])
    params = [p for p in student.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": c["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=c["lr"], betas=tuple(c["betas"]))
    assert not any(p.requires_grad for p in teacher.parameters())
    trajs = load_traj("train")
    ts = TrajectoryStates(trajs, c["budgets"])
    rs = RealStates("train", trajs["plen"], c["budgets"], StoryWindows("train", 256))
    val = make_val(c, load_traj("val"))
    rng = np.random.default_rng(c["seed"])
    m_dir = os.path.join(distill.METRICS, "train", a.run)
    os.makedirs(m_dir, exist_ok=True)
    info = {"config": c, "params": count_parameters(student), "teacher_params": count_parameters(teacher),
            "train_trajectories": len(trajs["ids"]), "env": env_info("cpu")}
    distill.save_json(info, m_dir, "run_info.json")
    print(f"{a.run}: {info['params']['total']:,} params, lambda_KD {c['lambda_kd']}, T {c['T']}, "
          f"targets {c['targets']}, {c['steps']} steps", flush=True)
    log = open(os.path.join(m_dir, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(COLS)
    v = evaluate(student, teacher, val)
    w.writerow([0, "", "", "", *(round(v[k], 4) for k in COLS[4:9]), "", "", "", "", 0])
    log.flush()
    print("  step 0 " + ", ".join(f"{k} {x:.4f}" for k, x in v.items()), flush=True)
    t0, acc, n_acc, gn = time.time(), np.zeros(3), 0, 0.0
    lam = c["lambda_kd"]
    for step in range(1, c["steps"] + 1):
        lr = lr_at(step - 1, c)
        for g in opt.param_groups:
            g["lr"] = lr
        st = ts.sample(c["traj_batch"], rng)                     # drawn in every run: identical streams
        sr = rs.sample(c["real_batch"], rng)
        l_kd = l_ce = None
        if lam > 0:
            per, _ = traj_loss(student, teacher, st, c["T"], c["targets"])
            l_kd = per.mean()
        if lam < 1:
            z = logits_at(student, sr["x"], sr["masked"])
            l_ce = losses.ce(z, sr["x0"][sr["masked"]]).mean()
        loss = losses.total(l_kd, l_ce, lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn += float(torch.nn.utils.clip_grad_norm_(student.parameters(), c["grad_clip"]))
        opt.step()
        acc += [loss.item(), l_kd.item() if l_kd is not None else 0.0, l_ce.item() if l_ce is not None else 0.0]
        n_acc += 1
        if step % 50 == 0:
            print(f"step {step:5d} | loss {acc[0] / n_acc:.4f} kd {acc[1] / n_acc:.4f} ce {acc[2] / n_acc:.4f} "
                  f"| lr {lr:.2e} | {(time.time() - t0) / step:.2f} s/step", flush=True)
        if step % c["eval_every"] == 0 or step == c["steps"]:
            v = evaluate(student, teacher, val)
            el = time.time() - t0
            w.writerow([step, *(round(x / n_acc, 4) for x in acc), *(round(v[k], 4) for k in COLS[4:9]),
                        f"{lr:.2e}", round(gn / n_acc, 3), round(el / step, 3), round(peak_memory_mb("cpu")),
                        round(el, 1)])
            log.flush()
            print("  " + ", ".join(f"{k} {x:.4f}" for k, x in v.items()), flush=True)
            acc, n_acc, gn = np.zeros(3), 0, 0.0
    el = time.time() - t0
    save_student(os.path.join(distill.CHECKPOINTS, a.run, "student.pt"), student.eval(),
                 {"run": a.run, "config": c, "steps": c["steps"], "wall_clock_s": round(el, 1)})
    distill.save_json({"run": a.run, "steps": c["steps"], "wall_clock_s": round(el, 1),
                       "s_per_step": round(el / c["steps"], 3), "peak_mem_mb": round(peak_memory_mb("cpu")),
                       "final": v}, m_dir, "train_summary.json")
    log.close()
    print(json.dumps({"run": a.run, "wall_clock_s": round(el, 1), "final": v}, indent=1))


if __name__ == "__main__":
    main()
