"""Phases 11-14: paired comparisons on the test split (400 prompts x 3 seeds), one row per question / hypothesis.

Paired bootstrap over prompts (all seeds of a prompt resampled together, 10,000 resamples, Project 4's
bench.evaluate_quality.paired_diff), as in Projects 5-6:
  scorer perplexity ratio A / B, duplicated adjacent tokens (pp), repetitive stories (pp), mean passes of A and B.
Distinct-2 is a group-level measure (12 cells = 3 seeds x 4 prompt groups): mean paired difference over cells and the
number of cells in which A is lower.

  python -m distill.compare   -> results/metrics/test_comparisons.json
"""
import math
import os

import numpy as np

import distill
from adaptive.run import read_jsonl
from bench.evaluate_quality import paired_diff
from distill.run import system
from src.evaluate import distinct_n


def pairs(best: str):
    T = lambda S, **k: system("teacher", S, **k)  # noqa: E731
    B = lambda S, **k: system(best, S, **k)  # noqa: E731
    return [
        (T(8), T(32), "H1: teacher, 8 vs 32 passes (step reduction alone)"),
        (T(16), T(32), "teacher, 16 vs 32 passes"),
        (T(4), T(8), "teacher, 4 vs 8 passes"),
        (system("ce", 8), T(8), "H2: CE-only student vs teacher, 8 passes (extra training alone)"),
        (system("kd", 8), system("ce", 8), "H3: KD student (pre-registered kd) vs CE-only, 8 passes"),
        (B(8), system("ce", 8), "H3: best KD student vs CE-only, 8 passes"),
        (B(16), system("ce", 16), "best KD vs CE-only, 16 passes"),
        (B(4), system("ce", 4), "best KD vs CE-only, 4 passes"),
        (B(8), T(8), "best KD student vs teacher, 8 passes"),
        (B(16), T(16), "best KD student vs teacher, 16 passes"),
        (B(4), T(4), "H5: best KD student vs teacher, 4 passes"),
        (B(8), T(16), "H4: best KD student at 8 vs teacher at 16 passes"),
        (B(8), T(32), "best KD student at 8 vs teacher at 32 passes"),
        (B(16), T(32), "best KD student at 16 vs teacher at 32 passes"),
        (system("kd", 8), system("kd_state", 8), "H6: trajectory targets vs same-state targets, 8 passes"),
        (B(8), T(8, sampling="topk20t0.8"), "H8: best KD student vs sharper teacher (temperature 0.8), 8 passes"),
        (T(8, sampling="topk20t0.8"), T(8), "sharpening alone: teacher at temperature 0.8 vs 1.0, 8 passes"),
        (B(8, conf=True), B(8), "reveal rule (student): confidence vs random + no-adjacent, 8 passes"),
        (T(8, conf=True), T(8), "reveal rule (teacher): confidence vs random + no-adjacent, 8 passes"),
        (system("kd_small", 8), system("kd", 8), "4-layer vs 6-layer KD student, 8 passes"),
        (system("kd_small", 8), T(8), "4-layer KD student vs teacher, 8 passes"),
    ]


def d2_cells(gen):
    cells = {}
    for g in gen:
        cells.setdefault((g["seed_index"], g["group"]), []).append(g["completion"])
    return {k: distinct_n(v, 2, 100) for k, v in cells.items()}


def main():
    best = distill.load_json(distill.METRICS, "selection.json")["kd_best"]
    sdir = os.path.join(distill.METRICS, "scores", "test")
    out, seen = [], set()
    for a, b, q in pairs(best):
        if (a, b) in seen:
            continue
        seen.add((a, b))
        if not all(os.path.exists(os.path.join(sdir, x + ".jsonl")) for x in (a, b)):
            print("missing:", a, "or", b)
            continue
        ra, rb = read_jsonl(sdir, a + ".jsonl"), read_jsonl(sdir, b + ".jsonl")
        ppl = paired_diff(ra, rb, "scorer_nll", "scorer_tokens")
        dup = paired_diff(ra, rb, "dup", "dup_pairs")
        rep = paired_diff(ra, rb, "repetitive", None)
        ca = d2_cells(read_jsonl(distill.GENERATIONS, "test", a + ".jsonl"))
        cb = d2_cells(read_jsonl(distill.GENERATIONS, "test", b + ".jsonl"))
        dd = [ca[k] - cb[k] for k in ca]
        row = {"a": a, "b": b, "question": q,
               "nfe_a": float(np.mean([r["nfe"] for r in ra])), "nfe_b": float(np.mean([r["nfe"] for r in rb])),
               "ppl_ratio": math.exp(ppl["diff"]), "ppl_ratio_ci95": [math.exp(x) for x in ppl["ci95"]],
               "ppl_p": ppl["p_two_sided"],
               "dup_diff_pp": 100 * dup["diff"], "dup_diff_ci95_pp": [100 * x for x in dup["ci95"]],
               "rep_diff_pp": 100 * rep["diff"], "rep_diff_ci95_pp": [100 * x for x in rep["ci95"]],
               "d2_diff_mean": float(np.mean(dd)), "d2_cells_a_lower": int(sum(x < 0 for x in dd)), "d2_cells": len(dd)}
        out.append(row)
        print(f"{q}\n   {a} ({row['nfe_a']:.1f}) vs {b} ({row['nfe_b']:.1f}): ppl x{row['ppl_ratio']:.3f} "
              f"[{row['ppl_ratio_ci95'][0]:.3f}, {row['ppl_ratio_ci95'][1]:.3f}] p={row['ppl_p']:.4f} | "
              f"dup {row['dup_diff_pp']:+.3f} pp | rep {row['rep_diff_pp']:+.2f} pp | "
              f"D2 {row['d2_diff_mean']:+.4f} (A lower in {row['d2_cells_a_lower']}/{row['d2_cells']})", flush=True)
    distill.save_json(out, distill.METRICS, "test_comparisons.json")


if __name__ == "__main__":
    main()
