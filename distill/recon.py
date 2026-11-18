"""Phase 12 (part 1): masked-token reconstruction, reported SEPARATELY from open-ended generation quality.

For the teacher and every student, on 800 held-out TEST windows (story-aligned, Project 3's fixed_subset, seed 4321):
  uniform    Project 3's protocol: one masking level t ~ U(0, 1) per window, each token masked with prob. t.
             masked-token CE, accuracy, and the diffusion bound exp(NELBO) (an UPPER bound on perplexity; not
             comparable with an autoregressive model's perplexity).
  schedule   the masks a 4-, 8- or 16-pass decoder meets: a prompt of a training-prompt length, a share 1 - j/S of the
             continuation masked (same quota as the decoder, positions at random). CE and accuracy per budget S.
Same windows, levels and masks for every model.

  python -m distill.recon [--models teacher ce kd ...]   -> results/metrics/reconstruction.json
"""
import argparse

import numpy as np
import torch

import distill
from distill.models import load_system
from distill.run import train_runs
from distill.trajectories import load as load_traj
from mdlm.data import fixed_subset
from mdlm.masking import corrupt, eligible_positions, sample_t
from mdlm.tokens import MASK_ID

N, SEED = 800, 4321


def protocols():
    x0 = fixed_subset("test", N, 256, seed=SEED)
    g = torch.Generator().manual_seed(SEED)
    t = sample_t(len(x0), g)
    xt, m = corrupt(x0, t, g)
    uni = {"x0": x0, "x": xt, "masked": m, "t": t}
    sched = {}
    pool = load_traj("train")["plen"]
    for S in (4, 8, 16):
        # every window once, in order (RealStates' quota rule, without sampling windows with replacement)
        rng = np.random.default_rng(SEED + S)
        plen = rng.choice(pool, N)
        j = rng.integers(0, S, N)
        n0 = 256 - plen
        left = np.floor(n0 * (1.0 - j / S)).astype(np.int64)
        u = rng.random((N, 256))
        u[np.arange(256)[None, :] < plen[:, None]] = 2.0
        masked = torch.from_numpy(np.argsort(np.argsort(u, 1), 1) < left[:, None])
        sched[S] = {"x0": x0, "x": torch.where(masked, torch.full_like(x0, MASK_ID), x0), "masked": masked}
    return uni, sched


@torch.no_grad()
def measure(model, uni, sched, batch=50):
    out = {}
    ce = acc = n = 0.0
    nelbo = []
    for i in range(0, N, batch):
        x0, x, m, t = (uni[k][i:i + batch] for k in ("x0", "x", "masked", "t"))
        z = model(x)
        l = torch.nn.functional.cross_entropy(z.transpose(1, 2), x0, reduction="none") * m
        ce += l.sum().item()
        acc += ((z.argmax(-1) == x0) & m).sum().item()
        n += m.sum().item()
        nelbo.append((l.sum(1) / t / eligible_positions(x0).sum(1)).numpy())
    nel = float(np.concatenate(nelbo).mean())
    out["uniform"] = {"masked_ce": ce / n, "masked_acc": acc / n, "nelbo": nel, "ppl_bound": float(np.exp(nel))}
    for S, p in sched.items():
        ce = acc = n = 0.0
        for i in range(0, N, batch):
            x0, x, m = (p[k][i:i + batch] for k in ("x0", "x", "masked"))
            z = model(x)
            ce += (torch.nn.functional.cross_entropy(z.transpose(1, 2), x0, reduction="none") * m).sum().item()
            acc += ((z.argmax(-1) == x0) & m).sum().item()
            n += m.sum().item()
        out[f"schedule_S{S}"] = {"masked_ce": ce / n, "masked_acc": acc / n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    torch.set_num_threads(24)
    models = a.models or ["teacher"] + train_runs()
    uni, sched = protocols()
    import os
    path = os.path.join(distill.METRICS, "reconstruction.json")
    res = distill.load_json(path) if os.path.exists(path) else {}
    for k in models:
        res[k] = measure(load_system(k), uni, sched)
        u = res[k]["uniform"]
        print(f"{k:10s} uniform CE {u['masked_ce']:.4f} acc {u['masked_acc']:.4f} bound {u['ppl_bound']:.2f} | "
              + " ".join(f"S{S} CE {res[k][f'schedule_S{S}']['masked_ce']:.4f}" for S in (4, 8, 16)), flush=True)
        distill.save_json(res, path)


if __name__ == "__main__":
    main()
