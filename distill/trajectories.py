"""Phase 5: collect teacher denoising trajectories.

For every training prompt the frozen teacher generates the full 256-token window with Project 6's best decoder at the
high-step setting: random order + no-adjacent, 32 passes (full-random-K32+noadj+lazy; no early stop, so every
position of the window gets a pass number). Because committed tokens never change, the whole trajectory
x(0) (only the prompt) -> x(1) -> ... -> x(32) (finished window) is stored compactly as
    ids[i, w]   final token at window position w (prompt included; position 0 = <BOS>)
    step[i, w]  pass at which position w was committed (0 = prompt), 1..32
    conf[i, w]  teacher probability of the drawn token at that moment
State after pass s = ids where step <= s, [MASK] elsewhere. Teacher logits are NOT stored (8,193 floats per position
would take ~50 GB); they are recomputed from a state during training, which is exact because the teacher is frozen
and deterministic in eval mode.

Each batch of 50 prompts uses one torch.Generator seeded 7000 + batch index -> reproducible.

Run:  python -m distill.trajectories   -> results/trajectories/{train.npz, sample.jsonl, info.json}
"""
import json
import os
import time

import numpy as np
import torch

import distill
from adaptive.generate import SAMPLING
from bench.load_models import load_tokenizer
from distill.models import load_teacher
from distill.prompts import load as load_prompts
from mdlm.tokens import EOS_ID
from pardec.decode import decode_ids, parse

TEACHER_DECODER = "full-random-K32+noadj+lazy"
WINDOW, BATCH, SEED0, PASSES = 256, 50, 7000, 32


def collect(model, tok, prompts, d, seed0=SEED0, log=True):
    n = len(prompts)
    ids = np.zeros((n, WINDOW), np.int16)
    step = np.zeros((n, WINDOW), np.int8)
    conf = np.zeros((n, WINDOW), np.float16)
    t0 = time.time()
    for j in range(0, n, BATCH):
        chunk = prompts[j:j + BATCH]
        g = torch.Generator().manual_seed(seed0 + j // BATCH)
        x, info = decode_ids(model, [tok.encode(q["prompt"]) for q in chunk], WINDOW, d, SAMPLING["topk20"], g)
        ids[j:j + len(chunk)] = x.numpy()
        step[j:j + len(chunk)] = info["commit_step"].numpy()
        conf[j:j + len(chunk)] = np.nan_to_num(info["commit_conf"].numpy(), nan=1.0)
        if log and (j // BATCH) % 10 == 0:
            el = time.time() - t0
            done = j + len(chunk)
            print(f"{done}/{n} trajectories, {el:.0f} s, eta {el / done * (n - done):.0f} s", flush=True)
    return ids, step, conf, time.time() - t0


def load(split: str = "train"):
    """Arrays of the training trajectories ('train') or of the held-out prompts ('val')."""
    z = np.load(os.path.join(distill.TRAJ, "train.npz"))
    keep = ~z["held_out"] if split == "train" else z["held_out"]
    return {k: z[k][keep] for k in ("ids", "step", "conf", "plen")}


def main():
    torch.set_num_threads(24)
    tok = load_tokenizer()
    model = load_teacher()
    prompts = load_prompts("train")
    ids, step, conf, dt = collect(model, tok, prompts, parse(TEACHER_DECODER))
    plen = np.array([1 + len(tok.encode(q["prompt"])) for q in prompts], np.int16)   # <BOS> + prompt
    pos = np.arange(WINDOW)[None, :]
    assert (step[pos < plen[:, None]] == 0).all() and (step[pos >= plen[:, None]] >= 1).all()
    os.makedirs(distill.TRAJ, exist_ok=True)
    np.savez_compressed(os.path.join(distill.TRAJ, "train.npz"), ids=ids, step=step, conf=conf, plen=plen,
                        held_out=np.array([q["held_out"] for q in prompts]),
                        group=np.array([q["group"] for q in prompts]))
    ends = []
    with open(os.path.join(distill.TRAJ, "sample.jsonl"), "w", encoding="utf-8") as f:
        for i in range(len(prompts)):
            row = ids[i, plen[i]:].tolist()
            ends.append(EOS_ID in row)
            if i % (len(prompts) // 40) == 0:
                story = row[:row.index(EOS_ID)] if EOS_ID in row else row
                f.write(json.dumps({"id": prompts[i]["id"], "prompt": prompts[i]["prompt"],
                                    "completion": tok.decode(story)}, ensure_ascii=False) + "\n")
    cont = step[step > 0].astype(np.int64)
    per_pass = np.bincount(cont, minlength=PASSES + 1)[1:] / len(prompts)
    info = {"decoder": TEACHER_DECODER, "sampling": "topk20", "n": len(prompts),
            "held_out": int(sum(q["held_out"] for q in prompts)), "seconds": round(dt, 1),
            "s_per_trajectory": round(dt / len(prompts), 4), "tokens_per_pass_mean": per_pass.round(2).tolist(),
            "ended_with_eos_rate": float(np.mean(ends)), "stored_bytes_per_trajectory": WINDOW * 5}
    distill.save_json(info, distill.TRAJ, "info.json")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
