"""Phases 9, 11-12: generate with the teacher and the students, score, summarise.

System names: "<model>@<decoder>[@<sampling>]" (see configs/sampling.yaml). Every system: prompts in file order,
batches of 50, one torch.Generator per batch seeded 6000 + 100 * seed + batch (Projects 5-6), so the teacher rows
reproduce Project 6 exactly where the decoder is the same.
Stories -> results/generations/<split>/<name>.jsonl; per-story scores -> results/metrics/scores/<split>/<name>.jsonl;
summary -> results/metrics/<split>_summary.json. Quality measures are Project 4's (bench.evaluate_quality).

  python -m distill.run --split dev                  # the dev grid of configs/sampling.yaml
  python -m distill.run --split test --names ...     # selected systems, 3 seeds
"""
import argparse
import os
import time

import torch
import yaml

import distill
from adaptive.generate import SAMPLING
from adaptive.run import read_jsonl, score_rows, write_jsonl
from bench.evaluate_quality import summarize_rows
from bench.load_models import load_scorer, load_tokenizer
from distill.models import load_system
from distill.prompts import load as load_prompts
from pardec.decode import decode_text, parse

BATCH = 50
REAL = "real-reference"
SAMPLING.setdefault("topk20t0.8", dict(method="top_k", top_k=20, temperature=0.8))   # in memory only


def sampling_cfg():
    return yaml.safe_load(open(os.path.join(distill.CONFIGS, "sampling.yaml"), encoding="utf-8"))


def system(model: str, S: int, conf: bool = False, sampling: str | None = None) -> str:
    c = sampling_cfg()
    d = (c["confidence_decoder"] if conf else c["main_decoder"]).format(S=S)
    return f"{model}@{d}" + (f"@{sampling}" if sampling else "")


def split_name(nm: str):
    parts = nm.split("@")
    return parts[0], parts[1], (parts[2] if len(parts) > 2 else "topk20")


def train_runs() -> list[str]:
    return list(yaml.safe_load(open(os.path.join(distill.CONFIGS, "distill.yaml"), encoding="utf-8"))["runs"])


def dev_names() -> list[str]:
    c = sampling_cfg()["dev"]
    names = [system("teacher", s) for s in c["teacher"]] + list(c["teacher_extra"])
    for r in train_runs():
        names += [system(r, s) for s in c["students"]]
    names += list(c["student_extra"])
    return names


def generate_run(model, tok, prompts, nm: str, seeds) -> list[dict]:
    _, dec, sampling = split_name(nm)
    d = parse(dec)
    rows = []
    for k in seeds:
        for j in range(0, len(prompts), BATCH):
            chunk = prompts[j:j + BATCH]
            metas = [{"prompt_id": q["id"], "group": q["group"], "seed_index": k} for q in chunk]
            rows += decode_text(model, tok, [q["prompt"] for q in chunk], d, sampling, seed=6000 + 100 * k + j // BATCH,
                                metas=metas, label=nm)
    return rows


def summarize(split: str) -> dict:
    sdir = os.path.join(distill.METRICS, "scores", split)
    res = {}
    for fn in sorted(os.listdir(sdir)):
        nm = fn[:-6]
        rows = read_jsonl(sdir, fn)
        gen = read_jsonl(distill.GENERATIONS, split, fn)
        s = summarize_rows(rows, gen)
        s["nfe_mean"] = float(sum(r["nfe"] for r in rows) / len(rows))
        s["nfe_max"] = max(r["nfe"] for r in rows)
        s["new_tokens_mean"] = float(sum(r["new_tokens"] for r in rows) / len(rows))
        s["settings"] = gen[0].get("settings", {})
        res[nm] = s
    distill.save_json(res, distill.METRICS, f"{split}_summary.json")
    return res


def run(split: str, names: list[str]):
    torch.set_num_threads(24)
    tok = load_tokenizer()
    scorer, stok = load_scorer()
    prompts, seeds = (load_prompts("dev"), (0,)) if split == "dev" else (load_prompts("eval"), (0, 1, 2))
    if not os.path.exists(os.path.join(distill.METRICS, "scores", split, REAL + ".jsonl")):
        real = [{"prompt_id": q["id"], "group": q["group"], "seed_index": 0, "prompt": q["prompt"],
                 "completion": " " + q["reference"], "new_tokens": len(tok.encode(q["reference"])),
                 "words": len(q["reference"].split()), "chars": len(q["reference"]) + 1, "ended_with_eos": True,
                 "nfe": 0} for q in prompts]
        write_jsonl(real, distill.GENERATIONS, split, REAL + ".jsonl")
        write_jsonl(score_rows(real, prompts, scorer, stok, tok), distill.METRICS, "scores", split, REAL + ".jsonl")
    print(f"{len(names)} systems on {len(prompts)} prompts x {len(seeds)} seeds", flush=True)
    models = {}
    for nm in names:
        out = os.path.join(distill.METRICS, "scores", split, nm + ".jsonl")
        if os.path.exists(out):
            continue
        key = split_name(nm)[0]
        if key not in models:
            models = {key: load_system(key)}               # keep one model in memory at a time
        t0 = time.perf_counter()
        rows = generate_run(models[key], tok, prompts, nm, seeds)
        t1 = time.perf_counter()
        write_jsonl(rows, distill.GENERATIONS, split, nm + ".jsonl")
        write_jsonl(score_rows(rows, prompts, scorer, stok, tok), out)
        print(f"{nm}: {len(rows)} stories, gen {t1 - t0:.0f} s, score {time.perf_counter() - t1:.0f} s", flush=True)
    res = summarize(split)
    for nm, s in sorted(res.items(), key=lambda kv: (kv[1]["nfe_mean"], kv[0])):
        print(f"{nm:62s} passes {s['nfe_mean']:5.1f}  ppl {s['scorer_ppl']:7.1f}  D2 {s['distinct_2']:.3f}  "
              f"rep {100 * s['repetitive_rate']:5.1f}%  dup {100 * s['dup_rate']:.2f}%", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--names", nargs="*", default=None)
    a = ap.parse_args()
    names = a.names or (dev_names() if a.split == "dev" else distill.load_json(distill.METRICS, "selection.json")["test_names"])
    run(a.split, names)


if __name__ == "__main__":
    main()
