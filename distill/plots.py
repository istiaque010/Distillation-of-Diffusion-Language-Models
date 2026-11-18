"""Phase 15: figures (Okabe-Ito colours, colour-blind safe; every series also has its own marker or line style).

  fig1_quality_vs_passes   test scorer perplexity vs passes: teacher, CE-only student, best KD student (log scale)
  fig2_latency_vs_quality  test: median latency (batch 1) vs scorer perplexity
  fig3_training_curves     validation losses during training, every run
  fig4_ablations           dev perplexity at 8 passes: KD weight, temperature, targets, depth
  fig5_divergence          ratio to the teacher at 32 passes, per prompt group (8 passes)
  fig6_errors              blind AI-labelled failure matrix (80 stories)

  python -m distill.plots   -> results/figures/*.png
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import distill  # noqa: E402
from distill.run import system  # noqa: E402

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
STY = {"teacher": (OI["black"], "o", "-"), "ce": (OI["orange"], "s", "--"), "kd": (OI["blue"], "D", "-"),
       "small": (OI["purple"], "v", ":"), "sharp": (OI["red"], "^", ":"), "state": (OI["green"], "P", "-.")}


def save(fig, name):
    os.makedirs(distill.FIGURES, exist_ok=True)
    fig.savefig(os.path.join(distill.FIGURES, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig1(s, best):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lab, key, st in [("teacher (no retraining)", "teacher", STY["teacher"]), ("CE-only student", "ce", STY["ce"]),
                         (f"KD student ({best})", best, STY["kd"])]:
        xs, ys, lo, hi = [], [], [], []
        for S in (4, 8, 16, 32):
            nm = system(key, S)
            if nm in s:
                v = s[nm]
                xs.append(v["nfe_mean"])
                ys.append(v["scorer_ppl"])
                lo.append(v["scorer_ppl"] - v["scorer_ppl_ci95"][0])
                hi.append(v["scorer_ppl_ci95"][1] - v["scorer_ppl"])
        ax.errorbar(xs, ys, yerr=[lo, hi], color=st[0], marker=st[1], ls=st[2], lw=2, ms=7, capsize=3, label=lab)
    nm = system("teacher", 8, sampling="topk20t0.8")
    if nm in s:
        ax.plot([s[nm]["nfe_mean"]], [s[nm]["scorer_ppl"]], color=STY["sharp"][0], marker="^", ms=9, ls="none",
                label="teacher, temperature 0.8")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([4, 8, 16, 32])
    ax.set_xticklabels(["4", "8", "16", "32"])
    ax.set_xlabel("forward passes per story (mean)")
    ax.set_ylabel("scorer perplexity (lower is better)")
    ax.set_title("Quality vs passes (test, 1,200 stories each, 95% CI)")
    ax.grid(alpha=.3, which="both")
    ax.legend(frameon=False)
    save(fig, "fig1_quality_vs_passes")


def fig2(s, t, best):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, lab, st in [("teacher", "teacher", STY["teacher"]), ("ce", "CE-only student", STY["ce"]),
                         (best, f"KD student ({best})", STY["kd"]), ("kd_small", "4-layer KD student", STY["small"])]:
        pts = []
        for S in (4, 8, 16, 32):
            nm = system(key, S)
            if nm in s and nm in t:
                pts.append((t[nm]["median_s"], s[nm]["scorer_ppl"], S))
        if not pts:
            continue
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=st[0], marker=st[1], ls=st[2], lw=2, ms=7, label=lab)
        for x, y, S in pts:
            ax.annotate(f"{S}", (x, y), textcoords="offset points", xytext=(5, -10) if st[0] == STY["ce"][0] else (5, 4), fontsize=8, color=st[0])
    ax.set_yscale("log")
    ax.set_xlabel("median latency per story, batch 1, CPU (s)")
    ax.set_ylabel("scorer perplexity (lower is better)")
    ax.set_title("Latency vs quality (numbers = passes)")
    ax.grid(alpha=.3, which="both")
    ax.legend(frameon=False)
    save(fig, "fig2_latency_vs_quality")


def read_log(run):
    p = os.path.join(distill.METRICS, "train", run, "log.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    return {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0]}


def fig3(runs, best):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    cols = [OI[c] for c in ("orange", "blue", "sky", "green", "purple", "red", "yellow", "grey", "black")]
    marks = ["s", "D", "o", "^", "v", "<", ">", "P", "X"]
    for i, r in enumerate(runs):
        L = read_log(r)
        if L is None:
            continue
        lw = 2.5 if r in ("ce", best) else 1.2
        for ax, k in zip(axes, ("val_traj", "val_sched_ce", "val_ce")):
            ax.plot(L["step"], L[k], color=cols[i % len(cols)], marker=marks[i % len(marks)], ms=4, lw=lw, label=r)
    for ax, ttl in zip(axes, ("held-out trajectory loss (KD target, T=1)", "CE, real windows, student masks",
                              "CE, real windows, uniform masking (Project 3)")):
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("update")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("loss (nats / masked token)")
    axes[2].legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "fig3_training_curves")


def fig4(dev):
    def ppl(run):
        nm = system(run, 8)
        return dev[nm]["scorer_ppl"] if nm in dev else np.nan
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [5, 3, 4]})
    lam = [("0 (CE)", "ce"), ("0.25", "kd_l25"), ("0.5", "kd"), ("0.75", "kd_l75"), ("1.0", "kd_l100")]
    axes[0].plot(range(len(lam)), [ppl(r) for _, r in lam], color=OI["blue"], marker="D", lw=2)
    axes[0].set_xticks(range(len(lam)))
    axes[0].set_xticklabels([l for l, _ in lam])
    axes[0].set_xlabel("KD weight λ_KD (T = 2)")
    tt = [("1", "kd_T1"), ("2", "kd"), ("4", "kd_T4")]
    axes[1].plot(range(3), [ppl(r) for _, r in tt], color=OI["green"], marker="o", lw=2)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels([l for l, _ in tt])
    axes[1].set_xlabel("temperature T (λ_KD = 0.5)")
    other = [("teacher", "teacher"), ("CE-\nonly", "ce"), ("same-\nstate", "kd_state"),
             ("traj-\nectory", "kd"), ("4-layer", "kd_small")]
    v = [ppl(r) for _, r in other]
    axes[2].bar(range(len(other)), v, color=[OI["black"], OI["orange"], OI["green"], OI["blue"], OI["purple"]],
                edgecolor="white", linewidth=2)
    for i, x in enumerate(v):
        axes[2].text(i, x, f"{x:.1f}", ha="center", va="bottom", fontsize=8)
    axes[2].set_xticks(range(len(other)))
    axes[2].set_xticklabels([l for l, _ in other], fontsize=8)
    t8 = ppl("teacher")
    for ax in axes[:2]:
        ax.axhline(t8, color=OI["black"], ls=":", lw=1)
        ax.text(ax.get_xlim()[0], t8, " teacher", va="bottom", fontsize=8)
    axes[0].set_ylabel("dev scorer perplexity, 8 passes")
    for ax in axes:
        ax.grid(alpha=.3, axis="y")
    fig.suptitle("Ablations (validation prompts, 200 stories each)", fontsize=11)
    save(fig, "fig4_ablations")


def fig5(div):
    groups = ["all", "short", "character", "action", "longer"]
    labs = [k for k in div if k.endswith("_K8")]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.8 / len(labs)
    cols = [OI["black"], OI["orange"], OI["blue"]]
    for i, lab in enumerate(labs):
        v = [div[lab][g]["ppl_ratio_vs_teacher_K32"] for g in groups]
        lo = [v[k] - div[lab][g]["ci95"][0] for k, g in enumerate(groups)]
        hi = [div[lab][g]["ci95"][1] - v[k] for k, g in enumerate(groups)]
        ax.bar(np.arange(5) + (i - (len(labs) - 1) / 2) * w, v, w * 0.9, yerr=[lo, hi], capsize=2,
               color=cols[i % 3], label=lab.replace("_K8", " (8 passes)"), edgecolor="white")
    ax.axhline(1, color=OI["grey"], lw=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(groups)
    ax.set_ylabel("perplexity ratio to teacher at 32 passes")
    ax.set_title("Where cutting passes hurts (test, paired, 95% CI)")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(alpha=.3, axis="y")
    save(fig, "fig5_divergence")


def fig6(err):
    syss = list(err)
    cats = ["repetition", "contradiction", "early_wrong_turn", "ungrammatical", "premature_ending", "ignored_prompt",
            "entity"]
    M = np.array([[err[s][c]["count"] for s in syss] for c in cats])
    fig, ax = plt.subplots(figsize=(6.6, 4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=20, aspect="auto")
    for i in range(len(cats)):
        for j in range(len(syss)):
            ax.text(j, i, f"{M[i, j]}", ha="center", va="center", color="white" if M[i, j] > 12 else "black")
    ax.set_xticks(range(len(syss)))
    ax.set_xticklabels([s.replace("_", " ") for s in syss], fontsize=9)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([c.replace("_", " ") for c in cats])
    ax.set_title("Failure matrix: stories (of 20) with each problem\n(blind labels by an AI assistant, not human)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=.8)
    save(fig, "fig6_errors")


def main():
    sel = distill.load_json(distill.METRICS, "selection.json")
    best = sel["kd_best"]
    s = distill.load_json(distill.METRICS, "test_summary.json")
    dev = distill.load_json(distill.METRICS, "dev_summary.json")
    fig1(s, best)
    tp = os.path.join(distill.METRICS, "timing.json")
    if os.path.exists(tp):
        fig2(s, distill.load_json(tp), best)
    from distill.run import train_runs
    fig3(train_runs(), best)
    fig4(dev)
    dp = os.path.join(distill.METRICS, "divergence.json")
    if os.path.exists(dp):
        fig5(distill.load_json(dp))
    ep = os.path.join(distill.METRICS, "error_analysis.json")
    if os.path.exists(ep):
        fig6(distill.load_json(ep))
    print("figures in", distill.FIGURES)


if __name__ == "__main__":
    main()
