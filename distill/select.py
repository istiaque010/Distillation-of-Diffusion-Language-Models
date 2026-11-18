"""Phase 11: apply the selection rule of configs/sampling.yaml, unchanged, to the dev results.

  python -m distill.select   -> results/metrics/selection.json (incl. test_names)
"""
import distill
from distill.run import sampling_cfg, system

KD_RUNS = ["kd", "kd_l25", "kd_l75", "kd_l100", "kd_T1", "kd_T4", "kd_state"]


def main():
    c = sampling_cfg()["selection"]["acceptable"]
    s = distill.load_json(distill.METRICS, "dev_summary.json")

    def ok(nm):
        v = s[nm]
        return v["repetitive_rate"] <= c["max_repetitive"] and v["distinct_2"] >= c["min_distinct_2"]

    cands = {r: s[system(r, 8)]["scorer_ppl"] for r in KD_RUNS if system(r, 8) in s and ok(system(r, 8))}
    kd_best = min(cands, key=cands.get) if cands else "kd"
    names = [system("teacher", S) for S in (32, 16, 8, 4)]
    names += [system("teacher", 8, sampling="topk20t0.8"), system("teacher", 8, conf=True)]
    names += [system("ce", S) for S in (16, 8, 4)]
    names += [system(kd_best, S) for S in (16, 8, 4)] + [system(kd_best, 8, conf=True)]
    for r in ("kd", "kd_state", "kd_small"):
        if system(r, 8) not in names:
            names.append(system(r, 8))
    out = {"rule": c, "kd_candidates_dev_ppl_K8": cands, "kd_best": kd_best,
           "rejected": {n: {"repetitive": v["repetitive_rate"], "distinct_2": v["distinct_2"]}
                        for n, v in s.items() if n != "real-reference" and not ok(n)},
           "test_names": names}
    distill.save_json(out, distill.METRICS, "selection.json")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
