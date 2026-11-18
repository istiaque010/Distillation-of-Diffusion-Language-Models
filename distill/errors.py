"""Phase 15: error analysis.

(1) Blind review of 20 test prompts x 4 systems = 80 PAIRED stories (same prompts for every system):
    teacher at 32 passes, teacher at 8, CE-only student at 8, best KD student at 8.
Sample: 5 prompts per group (seed 2029), seed-0 stories, shuffled into results/error_labels/blind_stories.md with
neutral ids S001..S080; the key is saved apart (key.json).

Rubric (1 = very poor ... 5 = as good as a real TinyStories story), the guide's Phase 12 criteria:
  coherence, relevance (to the prompt), grammar, completeness, consistency (entities)
Failure flags (letters), the guide's Phase 15 list:
  R repetition         the same phrase/sentence again and again, or a doubled word
  C contradiction      events contradict earlier sentences or do not follow at all
  W early wrong turn   an early odd word or event (first one or two sentences) sends the rest of the story off course
  G ungrammatical span at least one clearly malformed sentence
  I premature ending   no sensible conclusion (cut off, trailing, ends mid-thought)
  P ignored prompt     ignores the given story start
  E entity             a character's name, species or role changes, or people appear from nowhere

THE LABELS ARE PRODUCED BY AN AI ASSISTANT (Claude, run as a separate agent that saw only blind_stories.md), not by
human annotators, and are not human-verified. One annotator: no inter-annotator agreement. Blinding is imperfect.

(2) Divergence from the high-step teacher (all 1,200 test stories, no labels): for each prompt group, the paired
scorer-perplexity ratio of each 8- and 4-pass system to the teacher at 32 passes: which kinds of prompts suffer
most when passes are cut, and whether distillation closes the gap there.

  python -m distill.errors --stage sample      -> results/error_labels/{blind_stories.md, key.json}
  (labels written to results/error_labels/labels.py by the blind annotator)
  python -m distill.errors --stage count       -> results/metrics/error_analysis.{json,md}
  python -m distill.errors --stage divergence  -> results/metrics/divergence.json
"""
import argparse
import importlib.util
import math
import os
import random

import numpy as np

import distill
from adaptive.run import read_jsonl
from bench.evaluate_quality import paired_diff, wilson
from distill.prompts import load as load_prompts
from distill.run import system

DIMS = ("coherence", "relevance", "grammar", "completeness", "consistency")
CATS = {"R": "repetition", "C": "contradiction", "W": "early_wrong_turn", "G": "ungrammatical",
        "I": "premature_ending", "P": "ignored_prompt", "E": "entity"}
GROUPS = ("short", "character", "action", "longer")


def systems():
    best = distill.load_json(distill.METRICS, "selection.json")["kd_best"]
    return {"teacher_K32": system("teacher", 32), "teacher_K8": system("teacher", 8), "ce_K8": system("ce", 8),
            f"{best}_K8": system(best, 8)}


def stage_sample():
    prompts = load_prompts("eval")
    rng = random.Random(2029)
    chosen = []
    for g in GROUPS:
        chosen += rng.sample([p["id"] for p in prompts if p["group"] == g], 5)
    by_id = {p["id"]: p for p in prompts}
    items = []
    for lab, fn in systems().items():
        rows = {r["prompt_id"]: r for r in read_jsonl(distill.GENERATIONS, "test", fn + ".jsonl") if r["seed_index"] == 0}
        items += [(lab, pid, rows[pid]["completion"]) for pid in chosen]
    rng.shuffle(items)
    os.makedirs(distill.LABELS, exist_ok=True)
    key, lines = {}, ["# Blind stories (Project 7 error analysis)\n",
                      "Each item: the prompt, then a model's continuation. The system is hidden.\n"]
    for i, (lab, pid, comp) in enumerate(items, 1):
        sid = f"S{i:03d}"
        key[sid] = {"system": lab, "prompt_id": pid}
        lines.append(f"\n## {sid}\n\n**Prompt:** {by_id[pid]['prompt']}\n\n**Continuation:** {comp.strip()}\n")
    open(os.path.join(distill.LABELS, "blind_stories.md"), "w", encoding="utf-8", newline="\n").write("".join(lines))
    distill.save_json(key, distill.LABELS, "key.json")
    print(len(items), "stories written")


def load_labels():
    spec = importlib.util.spec_from_file_location("labels", os.path.join(distill.LABELS, "labels.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = {}
    for line in mod._RAW.strip().splitlines():
        parts = line.split(None, 7)
        sid, scores, flags = parts[0], list(map(int, parts[1:6])), parts[6]
        out[sid] = {"scores": dict(zip(DIMS, scores)), "flags": set(flags) - {"-"},
                    "reason": parts[7] if len(parts) > 7 else ""}
    return out


def stage_count():
    key = distill.load_json(distill.LABELS, "key.json")
    lab = load_labels()
    assert set(lab) == set(key), "every blind story needs exactly one label"
    order = list(systems())
    res = {}
    for sysname in order:
        ids = [s for s, v in key.items() if v["system"] == sysname]
        n = len(ids)
        r = {"n": n, "rubric": {d: float(np.mean([lab[s]["scores"][d] for s in ids])) for d in DIMS}}
        r["rubric_overall"] = float(np.mean([np.mean(list(lab[s]["scores"].values())) for s in ids]))
        for letter, cat in CATS.items():
            k = sum(letter in lab[s]["flags"] for s in ids)
            r[cat] = {"count": k, "rate": k / n, "ci95": list(wilson(k, n))}
        res[sysname] = r
    # paired: per prompt, rubric of each 8-pass system minus the teacher at 32 passes
    by = {(v["system"], v["prompt_id"]): np.mean(list(lab[s]["scores"].values())) for s, v in key.items()}
    pids = sorted({v["prompt_id"] for v in key.values()})
    for sysname in order[1:]:
        d = [by[(sysname, p)] - by[(order[0], p)] for p in pids]
        res[sysname]["paired_vs_teacher_K32"] = {"mean_diff": float(np.mean(d)), "worse": int(sum(x < 0 for x in d)),
                                                 "same": int(sum(x == 0 for x in d)), "better": int(sum(x > 0 for x in d))}
    distill.save_json(res, distill.METRICS, "error_analysis.json")
    md = ["| Measure | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    md.append("| Rubric overall (1-5) | " + " | ".join(f"{res[d]['rubric_overall']:.2f}" for d in order) + " |")
    for d in DIMS:
        md.append(f"| {d} | " + " | ".join(f"{res[x]['rubric'][d]:.2f}" for x in order) + " |")
    for cat in CATS.values():
        md.append(f"| {cat} | " + " | ".join(f"{res[x][cat]['count']}/{res[x]['n']}" for x in order) + " |")
    md.append("| vs teacher K32, same prompt: worse / same / better | – | " + " | ".join(
        "{worse} / {same} / {better}".format(**res[x]["paired_vs_teacher_K32"]) for x in order[1:]) + " |")
    open(os.path.join(distill.METRICS, "error_analysis.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print("\n".join(md))


def stage_divergence():
    best = distill.load_json(distill.METRICS, "selection.json")["kd_best"]
    sdir = os.path.join(distill.METRICS, "scores", "test")
    ref = read_jsonl(sdir, system("teacher", 32) + ".jsonl")
    out = {}
    for lab, nm in [("teacher_K16", system("teacher", 16)), ("teacher_K8", system("teacher", 8)),
                    ("ce_K8", system("ce", 8)), (f"{best}_K8", system(best, 8)),
                    ("teacher_K4", system("teacher", 4)), ("ce_K4", system("ce", 4)), (f"{best}_K4", system(best, 4))]:
        rows = read_jsonl(sdir, nm + ".jsonl")
        out[lab] = {}
        for g in ("all",) + GROUPS:
            a = [r for r in rows if g == "all" or r["group"] == g]
            b = [r for r in ref if g == "all" or r["group"] == g]
            d = paired_diff(a, b, "scorer_nll", "scorer_tokens")
            out[lab][g] = {"ppl_ratio_vs_teacher_K32": math.exp(d["diff"]),
                           "ci95": [math.exp(x) for x in d["ci95"]]}
        print(lab, {g: round(v["ppl_ratio_vs_teacher_K32"], 3) for g, v in out[lab].items()}, flush=True)
    distill.save_json(out, distill.METRICS, "divergence.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sample", "count", "divergence"])
    a = ap.parse_args()
    {"sample": stage_sample, "count": stage_count, "divergence": stage_divergence}[a.stage]()


if __name__ == "__main__":
    main()
