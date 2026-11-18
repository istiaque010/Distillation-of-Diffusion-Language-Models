"""Phase 13: latency, throughput and memory (CPU, fp32, 24 threads; no GPU on this PC), as in Project 6.

Latency (--timing): batch 1; 40 test prompts (10 per group), each story decoded to its natural end (early stop); model
loading and tokenisation are outside the timer; 2 untimed warm-up stories, then 40 timed stories, repeated `--repeats`
times (default 1 -> 40 runs >= 20): median and p95 latency, forward passes, ms per pass, story tokens/s. Other
programs' CPU load is measured around every system (> 15% = unreliable, re-run).
Throughput (--throughput): 32 test prompts (8 per group) in batches of 1, 8 and 32; outputs/s and story tokens/s
(median of 3 repeats, 1 untimed warm-up batch).
Memory (--memory): peak resident memory added by one decode (fresh subprocess, batch 1 and 16), parameter count and
checkpoint size. On a GPU the guide asks for peak allocated GPU memory; on this CPU the closest measure is peak RSS.

  python -m distill.benchmark --timing --names ...       -> results/metrics/timing.json
  python -m distill.benchmark --throughput --names ...   -> results/metrics/throughput.json
  python -m distill.benchmark --memory --names ...       -> results/metrics/memory.json
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

import distill
from bench.load_models import load_tokenizer
from distill.models import count_parameters, load_system
from distill.prompts import load as load_prompts
from distill.run import split_name
from pardec.decode import decode_text, parse
from src.evaluate import OtherCpuLoad

LIMIT = 15.0
_MODELS = {}


def model_for(nm):
    key = split_name(nm)[0]
    if key not in _MODELS:
        _MODELS.clear()
        _MODELS[key] = load_system(key)
    return _MODELS[key]


def run(tok, nm, texts, seed):
    _, dec, sampling = split_name(nm)
    return decode_text(model_for(nm), tok, texts, parse(dec), sampling, seed=seed)


def per_group(n):
    ev = load_prompts("eval")
    return [q for g in ("short", "character", "action", "longer") for q in [x for x in ev if x["group"] == g][:n]]


def timing(names, repeats=1):
    torch.set_num_threads(24)
    tok = load_tokenizer()
    prompts = per_group(10)
    path = os.path.join(distill.METRICS, "timing.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    load = OtherCpuLoad()
    for nm in names:
        if nm in res and res[nm]["reliable"]:
            continue
        for q in prompts[:2]:
            run(tok, nm, [q["prompt"]], 99)
        load.start()
        rows = []
        for rep in range(repeats):
            for i, q in enumerate(prompts):
                t0 = time.perf_counter()
                r = run(tok, nm, [q["prompt"]], 1000 * rep + i)[0]
                rows.append({"latency_s": time.perf_counter() - t0, "nfe": r["nfe"], "new_tokens": r["new_tokens"],
                             "model_s": r["model_time_s"]})
        other = load.stop()
        lat = np.array([r["latency_s"] for r in rows])
        res[nm] = {"median_s": float(np.median(lat)), "p95_s": float(np.percentile(lat, 95)),
                   "mean_s": float(lat.mean()), "nfe_mean": float(np.mean([r["nfe"] for r in rows])),
                   "model_share": float(sum(r["model_s"] for r in rows) / lat.sum()),
                   "ms_per_pass": float(1000 * lat.sum() / sum(r["nfe"] for r in rows)),
                   "tokens_per_s": float(sum(r["new_tokens"] for r in rows) / lat.sum()),
                   "runs": len(rows), "other_cpu_pct": other, "reliable": other <= LIMIT}
        print(nm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[nm].items()}, flush=True)
        distill.save_json(res, path)


def throughput(names, batches=(1, 8, 32), repeats=3):
    torch.set_num_threads(24)
    tok = load_tokenizer()
    prompts = [q["prompt"] for q in per_group(8)]
    path = os.path.join(distill.METRICS, "throughput.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    load = OtherCpuLoad()
    for nm in names:
        for B in batches:
            key = f"{nm}#B{B}"
            if key in res and res[key]["reliable"]:
                continue
            run(tok, nm, prompts[:B], 99)
            load.start()
            times, toks = [], []
            for rep in range(repeats):
                t0, n = time.perf_counter(), 0
                for j in range(0, len(prompts), B):
                    n += sum(r["new_tokens"] for r in run(tok, nm, prompts[j:j + B], 100 * rep + j))
                times.append(time.perf_counter() - t0)
                toks.append(n)
            other = load.stop()
            k = int(np.argsort(times)[len(times) // 2])
            res[key] = {"name": nm, "batch": B, "seconds": times[k], "outputs_per_s": len(prompts) / times[k],
                        "tokens_per_s": toks[k] / times[k], "other_cpu_pct": other, "reliable": other <= LIMIT}
            print(key, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[key].items()}, flush=True)
            distill.save_json(res, path)


def memory_one(nm, B):
    import psutil
    torch.set_num_threads(24)
    tok = load_tokenizer()
    prompts = ["Once upon a time"] * B
    run(tok, nm, prompts, 1)   # same-shape warm-up
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak, stop = [base], [False]

    def sample():
        while not stop[0]:
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.001)
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    run(tok, nm, prompts, 2)
    stop[0] = True
    th.join()
    m = model_for(nm)
    return {"name": nm, "batch": B, "peak_added_mb": (peak[0] - base) / 2**20,
            "process_rss_mb": peak[0] / 2**20, "params": count_parameters(m)["total"],
            "weights_mb": sum(p.numel() * p.element_size() for p in m.state_dict().values()) / 2**20}


def memory(names):
    out = []
    code = ("import sys; sys.path.insert(0, '.'); import distill.benchmark as b; "
            "sys.argv = ['b', '--single-memory'] + sys.argv[1:]; b.main()")
    for nm in names:
        for B in (1, 16):
            r = subprocess.run([sys.executable, "-I", "-c", code, nm, str(B)], capture_output=True, text=True,
                               cwd=distill.PROJECT7)
            line = [l for l in r.stdout.splitlines() if l.startswith("{")]
            if line:
                out.append(json.loads(line[-1]))
                print(out[-1], flush=True)
            else:
                print(r.stderr[-1500:])
    distill.save_json(out, distill.METRICS, "memory.json")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--single-memory":
        print(json.dumps(memory_one(sys.argv[2], int(sys.argv[3]))))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--timing", action="store_true")
    g.add_argument("--throughput", action="store_true")
    g.add_argument("--memory", action="store_true")
    a = ap.parse_args()
    if a.timing:
        timing(a.names, a.repeats)
    elif a.throughput:
        throughput(a.names)
    else:
        memory(a.names)


if __name__ == "__main__":
    main()
