"""Prompts.

  eval (test) = Project 4's 400 test prompts (4 groups x 100), byte-identical to Projects 5-6
  dev         = Project 5's 200 validation prompts (4 groups x 50); every choice is made on dev, test is used once
  train       = NEW: prompts from Project 1's TRAINING split, built with Project 4's rules (same 4 groups), used only
                to collect teacher trajectories. Training and test/dev stories come from different splits.

Run:  python -m distill.prompts   -> prompts/{eval,dev,train}_prompts.jsonl, prompts/MANIFEST.sha256
"""
import hashlib
import json
import os
import shutil

import numpy as np

import distill
import pardec
from bench.load_models import load_tokenizer
from bench.prompts import GROUPS, candidate
from mdlm.tokens import BOS_ID, EOS_ID
from src.dataset import load_tokens

TRAIN_SEED, TRAIN_PER_GROUP = 7070, 1575      # 6,300 prompts; the last 75 of each group are held out (val)
HELD_OUT_PER_GROUP = 75


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def manifest(folder):
    return dict(l.split()[::-1] for l in open(os.path.join(folder, "MANIFEST.sha256")) if l.strip())


def train_prompts(tok) -> list[dict]:
    ids = load_tokens(distill.bench.DATA_DIR, "train")
    starts = np.load(os.path.join(distill.bench.PROJECT3, "data", "train_starts.npy"))
    order = np.random.default_rng(TRAIN_SEED).permutation(len(starts) - 1)
    got = {g: [] for g in GROUPS}
    seen = set()
    for i in order:
        if all(len(v) >= TRAIN_PER_GROUP for v in got.values()):
            break
        s, e = int(starts[i]), int(starts[i + 1])
        row = np.asarray(ids[s + 1:e]).tolist()
        if row and row[-1] == EOS_ID:
            row = row[:-1]
        if not row or BOS_ID in row:
            continue
        text = tok.decode(row)
        for g in GROUPS:                       # rarest group first, a story is used once
            if len(got[g]) >= TRAIN_PER_GROUP:
                continue
            c = candidate(g, text, tok)
            if c is None or c["prompt"] in seen:
                continue
            seen.add(c["prompt"])
            got[g].append({"group": g, "split": "train", "story_index": int(i), "prompt": c["prompt"],
                           "prompt_tokens": c["prompt_tokens"]})
            break
    out = []
    for g in GROUPS:
        for k, r in enumerate(got[g]):
            held = k >= TRAIN_PER_GROUP - HELD_OUT_PER_GROUP
            out.append({"id": f"train-{g}-{k:04d}", **r, "held_out": held})
    return out


def main():
    os.makedirs(distill.PROMPTS, exist_ok=True)
    p6 = manifest(pardec.PROMPTS)
    lines = []
    for fn in ("eval_prompts.jsonl", "dev_prompts.jsonl"):
        dst = os.path.join(distill.PROMPTS, fn)
        shutil.copyfile(os.path.join(pardec.PROMPTS, fn), dst)
        assert sha256(dst) == p6[fn], f"{fn} differs from Project 6's frozen copy"
        lines.append(f"{sha256(dst)}  {fn}\n")
    tok = load_tokenizer()
    tr = train_prompts(tok)
    dst = os.path.join(distill.PROMPTS, "train_prompts.jsonl")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for r in tr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines.append(f"{sha256(dst)}  train_prompts.jsonl\n")
    with open(os.path.join(distill.PROMPTS, "MANIFEST.sha256"), "w", newline="\n") as f:
        f.writelines(lines)
    for fn in ("eval", "dev", "train"):
        print(fn, len(load(fn)), "prompts")


def load(name: str) -> list[dict]:
    """name: 'eval', 'dev' or 'train'; checks the sha256 manifest."""
    fn = f"{name}_prompts.jsonl"
    p = os.path.join(distill.PROMPTS, fn)
    assert manifest(distill.PROMPTS)[fn] == sha256(p), f"{fn} changed since it was frozen"
    return [json.loads(l) for l in open(p, encoding="utf-8")]


if __name__ == "__main__":
    main()
