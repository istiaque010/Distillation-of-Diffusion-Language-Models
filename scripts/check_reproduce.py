"""Phase 1 milestone: the teacher, run through Project 7's runner, reproduces Project 6's dev stories token for token.

Compares results/generations/dev/teacher@full-random-K16+noadj+stop.jsonl (Project 7) with
../project6/results/generations/dev/full-random-K16+noadj+stop.jsonl (Project 6), story by story.

  python -m distill.run --split dev --names "teacher@full-random-K16+noadj+stop"
  python scripts/check_reproduce.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import distill  # noqa: E402
from adaptive.run import read_jsonl  # noqa: E402

a = read_jsonl(distill.GENERATIONS, "dev", "teacher@full-random-K16+noadj+stop.jsonl")
b = read_jsonl(distill.PROJECT6, "results", "generations", "dev", "full-random-K16+noadj+stop.jsonl")
key = lambda r: (r["prompt_id"], r["seed_index"])  # noqa: E731
bb = {key(r): r for r in b}
same = sum(r["new_ids"] == bb[key(r)]["new_ids"] for r in a if key(r) in bb)
res = {"project7_rows": len(a), "project6_rows": len(b), "identical_stories": same,
       "all_identical": same == len(a) == len(b)}
print(json.dumps(res))
distill.save_json(res, distill.METRICS, "reproduce_project6.json")
