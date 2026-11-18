# Project 7 — Distillation of Diffusion Language Models: phase-by-phase instructions

Source plan: `docs/Project_7_Distillation_of_Diffusion_Language_Models_Guide.docx` (kept locally, not in the repository).

**Goal:** distill the Project 3 masked diffusion language model (the *teacher*) into a *student* that writes acceptable
stories with fewer refinement passes than the teacher.

**Research question:** can trajectory-based teacher supervision improve the quality–latency trade-off of a low-step
masked diffusion language model?

```
Project 3 teacher (frozen, 32 passes) → teacher trajectories → student training (KD + clean-token CE)
   → student at 16 / 8 / 4 passes → compare with teacher 32 / 16 / 8 / 4 and a CE-only student → quality, latency
```

## How to use this file
1. Work through the phases in order. Each has **Learn**, **Files**, **Run**, **Check** and **Result**.
2. Run everything from the `project7` folder with `.venv` active (Windows: `.venv\Scripts\activate`).
3. Status changes from ⬜ to ✅ when the phase is done and checked.

## Progress

| # | Phase | Status |
|---|---|---|
| 1 | Reuse the existing diffusion teacher | ✅ |
| 2 | Environment and dependencies | ✅ |
| 3 | Define the distillation target | ✅ |
| 4 | Choose the student architecture | ✅ |
| 5 | Teacher denoising trajectories | ✅ |
| 6 | Teacher target pairs | ✅ |
| 7 | Distillation loss | ✅ |
| 8 | Training loop | ✅ |
| 9 | Low-step student sampling | ✅ |
| 10 | Functional correctness | ✅ |
| 11 | Comparison baselines | ✅ |
| 12 | Generation quality | ✅ |
| 13 | Latency, throughput, memory | ✅ |
| 14 | Ablations | ✅ |
| 15 | Error analysis | ✅ |
| 16 | Repository and reproducibility | ✅ |

## Decisions for this PC and this project (senior-researcher choices, with reasons)

- **CPU, not GPU** (no GPU on this PC), fp32, 24 threads, as in Projects 1–6. The guide's CUDA synchronisation, AMP
  and "peak GPU memory" become plain wall-clock timing, fp32 and peak resident memory. Absolute speeds would differ on
  a GPU; pass counts and the ranking of systems mostly would not.
- **Teacher = Project 3's `mdlm_base` (MDLM-13.9M), frozen.** Loaded through Project 4's loader (sha256-checked).
- **Reuse, don't copy.** Project 6's `pardec` package is imported read-only: its decoder runs the teacher *and* the
  students, so every model is decoded by exactly the same code. Through it: Project 5 (`adaptive`), Project 4
  (`bench`: tokenizer, prompts, scorer TinyStories-33M, quality measures, paired bootstrap), Project 3 (`mdlm`: model,
  masking, data windows) and Project 1 (`src`). Numbers are directly comparable with Projects 4–6.
- **Decoder = Project 6's best**: random order + no-adjacent, linear quota, early stop, lazy logits
  (`full-random-K{S}+noadj+stop+lazy`). The guide suggests a confidence-ranking rule "initially"; Projects 3, 5 and 6
  all found that confidence order makes this model's stories 12–25× worse (it commits neighbouring tokens together).
  Confidence order is kept as an ablation (Phase 14) instead of the main rule.
- **Budget = one 256-token window** (Projects 4–6): output 255 − prompt tokens, the story ends at the first `<EOS>`.
- **Short fine-tuning, not training from scratch.** The student starts as an exact copy of the teacher (guide: "same
  architecture and tokenizer first"), then 1,500 updates of 16 windows (~6.1 M tokens, 3.8% of the teacher's training
  tokens). Learning rate 3e-5 (a 1e-4 pilot made held-out loss worse, see Phase 8). One update takes 0.5–1.05 s on this
  CPU (CE-only runs need no teacher pass), so one run takes 12–26 min and the 9-run grid ~3.3 h. One student is
  trained for all three budgets (4, 8 and 16 passes) at once: each training state comes from a randomly chosen budget.
- **Matched training for every run.** All runs see exactly the same stream of states (same seed); only the loss or
  the depth changes. The CE-only student is the same configuration with λ_KD = 0.
- **Tune on validation, test once.** Project 5's 200 validation prompts choose the best KD setting; Project 4's 400
  test prompts × 3 seeds are used once, for the systems fixed in `configs/sampling.yaml`.
- **"Common rubric" → AI labels, disclosed.** There is no human annotator. As in Projects 4–6, 80 stories are labelled
  blind by an AI assistant run as a separate agent, and every table says so.

---
## Phase 1 — Reuse the existing diffusion teacher

**Learn:** the teacher is the model Project 3 trained: a 6-layer bidirectional Transformer (13.9 M parameters) that
predicts every `[MASK]` position of a 256-token window at once. It is frozen: eval mode (no dropout) and
`requires_grad = False` everywhere, so nothing in this project can change it.

**Files:** `distill/__init__.py` (imports Projects 6 → 5 → 4 → 3 → 1 read-only), `distill/models.py`
(`load_teacher`), `scripts/check_reproduce.py`

**Run:**
```
python -m distill.run --split dev --names "teacher@full-random-K16+noadj+stop"
python scripts/check_reproduce.py
```

**Check:** the teacher, decoded through Project 7's runner, gives the same stories as Project 6 did.

**Result:** ✅ the checkpoint loads through Project 4's loader (sha256 recorded there); all 13,891,968 parameters are
frozen. Decoded through Project 7's runner with Project 6's decoder and seeds, the teacher gives **token-identical**
stories to Project 6 for all 200 validation prompts (`results/metrics/reproduce_project6.json`). Teacher on the
200 validation prompts (random order + no-adjacent, top-k 20):

| Passes | Scorer perplexity | Distinct-2 | Repetitive |
|---|---|---|---|
| 32 | 40.2 | 0.517 | 2.5% |
| 16 | 48.3 | 0.517 | 0.5% |
| 8 | 60.9 | 0.535 | 0.5% |
| 4 | 113.3 | 0.547 | 0.0% |
| real continuations | 4.4 | 0.621 | 6.5% |

Every halving of the passes costs 20–90% in perplexity: the room that distillation could win back.

---

## Phase 2 — Environment and dependencies

**Files:** `requirements.txt`, `requirements.lock.txt`

**Run:**
```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Result:** ✅ Python 3.12.10, PyTorch 2.14.1+cpu, tokenizers 0.23.3, transformers 5.22.2 (scorer only), numpy,
pandas, matplotlib, tqdm, PyYAML, pytest: the versions of Projects 1–6. `datasets` is not needed: the data are
Project 1's token files, read directly. No database, vector index or KNN (the guide says they are unnecessary). One
command loads the teacher and runs a forward pass:
`python -c "import distill, torch; from distill.models import load_teacher; print(load_teacher()(torch.full((1,256),8192)).shape)"`
→ `torch.Size([1, 256, 8193])`.

---

## Phase 3 — Define the distillation target

**Learn:** "steps" here are **forward passes of the model** over the continuation. With a linear quota, a 32-pass
decoder fixes ~1/32 of the masked positions per pass, an 8-pass decoder ~1/8. Fewer passes = faster, but more tokens
are drawn at the same time without seeing each other, and quality drops. Two effects must be kept apart:

| System | Training | Passes |
|---|---|---|
| Teacher | Project 3 (none here) | 32, 16 |
| Teacher with fewer passes | none | 8, 4 |
| CE-only student | 1,500 more updates on real data, no teacher | 16, 8, 4 |
| Distilled (KD) student | the same 1,500 updates, with teacher targets | 16, 8, 4 |

(1) cutting passes with no retraining = teacher 32 → teacher 8; (2) what training adds at the same budget = student 8
vs teacher 8, and what the *teacher's supervision* adds = KD student 8 vs CE-only student 8. Primary comparison:
**32 → 8**. Output length, tokenizer, prompts, decoder, sampling rule and hardware are the same for every system.

**Files:** `configs/sampling.yaml` (systems, selection rule), `configs/hypotheses.md` (H1–H8, written before training)

---

## Phase 4 — Choose the student architecture

**Learn:** the student has the teacher's architecture and tokenizer and starts as an **exact copy** of the teacher.
That isolates *step* distillation from *model compression*: any difference comes from training, not from size. The
shared tokenizer means `[MASK]` = 8192, `<BOS>` = 2, `<EOS>` = 3, `<PAD>` = 0 for both. The optional smaller student
(`kd_small`) keeps 4 of the teacher's 6 layers (0, 2, 3, 5): 10.3 M parameters.

**Files:** `distill/models.py` (`make_student`, `save_student`, `load_student`)

**Check:** the student's output has shape (batch, 256, 8,193); its output layer stays tied to its embedding.

**Result:** ✅ (tests `test_shallow_student_keeps_chosen_layers`, `test_teacher_frozen_student_updates`).

---

## Phase 5 — Teacher denoising trajectories

**Learn:**
```
x(0)  = <BOS> prompt [MASK] [MASK] [MASK] ...      (pass 0: only the prompt)
  ↓ teacher pass 1: ~7 positions fixed
x(1)  = <BOS> prompt [MASK] the [MASK] ...
  ↓ ... 32 passes ...
x(32) = <BOS> prompt  finished window
```
Committed tokens never change, so a whole trajectory is stored as two numbers per position: the final token and the
pass at which it was fixed. The state after pass s = "tokens fixed at a pass ≤ s, `[MASK]` elsewhere". Teacher
*logits* are not stored (8,193 numbers per position would need ~50 GB); they are recomputed from the state when
needed, which is exact because the teacher is frozen. Teacher stories are imperfect (scorer perplexity ~40, real text
~4.5), so real TinyStories text stays in the loss as a separate signal.

**Files:** `distill/prompts.py` (training prompts), `distill/trajectories.py`

**Run:**
```
python -m distill.prompts         # 6,300 prompts from the TRAINING split, Project 4's rules, 4 groups
python -m distill.trajectories    # ~30 min
```

**Result:** ✅ 6,300 prompts from Project 1's **training** split (1,575 per group, the same four groups and rules as
the test prompts; the last 75 of each group = 300 held-out prompts for validation). Their prompt lengths match the
test prompts (median 13 tokens, range 5–80). The teacher generated all 6,300 windows with
`full-random-K32+noadj+lazy` (32 passes, top-k 20) in **1,775 s (0.28 s per window)**; ~7.1 tokens were fixed per
pass, and 77% of the windows contain an `<EOS>` (the story ended inside the window). Stored: final ids, pass number
and the probability of each drawn token: 1,280 bytes per trajectory, 5 MB in total
(`results/trajectories/train.npz`), with 40 decoded examples in `sample.jsonl`. Batch seeds 7000 + batch →
reproducible.

---

## Phase 6 — Teacher target pairs

**Learn:** a student with S passes (4, 8 or 16) sees, before its pass j+1, a window where a share j/S of the
continuation is fixed. One student pass covers 32/S teacher passes:
```
a = 32·j/S,   b = 32·(j+1)/S
state  x(a)  = what the student sees                          (input)
span         = positions the teacher fixed in passes a+1 .. b  (what ONE student pass has to fix)
rest         = positions still masked after teacher pass b
later  x(b)  = the teacher's state after pass b
```
Example (S = 8, j = 2): the student sees the teacher's state after pass 8; the span is what the teacher fixed in
passes 9–12; the rest is still masked after pass 12. Every batch has 8 such **trajectory states** and 8 **real
states**: a real training window, a prompt length drawn from the training prompts, and the share 1 − j/S of the
continuation masked at random (same quota as the decoder). S is drawn from {4, 8, 16} for each state, so one student
learns all three budgets. All randomness comes from one seeded generator, so every run sees exactly the same states.

**Files:** `distill/states.py` (`TrajectoryStates`, `RealStates`)

**Check (tests):** span and rest split the masked positions exactly; the prompt is never masked; the number of masks
equals the decoder's quota after j of S passes (checked against Project 6's `_counts`).

**Result:** ✅

---

## Phase 7 — Distillation loss

**Learn:**
```
L_KD = mean over masked positions i of
         span i:  CE(z_student,i , teacher token)                        "which token should this pass fix"
         rest i:  T² · KL( softmax(z_teacher(x(b)),i / T) || softmax(z_student,i / T) )
L_CE = mean over masked positions of the real states of CE(z_student,i , real token)
L    = λ_KD · L_KD + (1 − λ_KD) · L_CE            starter: T = 2, λ_KD = 0.5
```
This is the guide's "advanced extension": the student sees the state **before** a span of teacher passes and is
taught what the teacher did **during and after** that span. The guide's beginner target (run `kd_state`) is the
teacher's distribution at the **same** state. A key fact, worked out before training: a student that starts as an
exact copy of the teacher gets almost no gradient from same-state targets (its predictions already equal the
teacher's; only its dropout noise differs), so only trajectory targets can teach it something the teacher does not
already do in one pass.

KL direction: KL(teacher ‖ student): the student must cover what the teacher thinks possible. `[MASK]` and `<PAD>`
are removed from both distributions (the decoder never draws them). Means are over masked positions only; unmasked
positions and padding never enter the loss. The temperature only softens soft targets; teacher tokens are matched
with plain cross-entropy.

**Files:** `distill/losses.py`

**Check (tests):** KL is zero for identical logits, equals KL(teacher ‖ student) and not the reverse, scales with T²;
`[MASK]`/`<PAD>` get zero target probability; the loss is finite.

**Result:** ✅ In a 60-update smoke run all loss parts stayed finite (L_KD 1.57, L_CE 3.62 at update 50) and only the
student's weights changed.

---

## Phase 8 — Training loop

**Learn:**
```
for each update:
    trajectory states, real states = sample(...)
    with torch.no_grad(): teacher logits at the later state (rest positions only)
    student logits at the masked positions only (the output layer is ~1/4 of the model's compute)
    loss = λ_KD · L_KD + (1 − λ_KD) · L_CE
    AdamW (weight decay 0.1 on matrices), lr 3e-5, warm-up 100 → cosine → 3e-6, gradient clipping 1.0
```
The teacher's own training ended at a learning rate of 6e-5. A first **pilot** of the CE-only run at 1e-4 made every
held-out number worse (`val_sched_ce` 3.5675 → 3.6064, `val_ce` 3.0596 → 3.1152): the teacher is already converged and
1e-4 knocked it off its minimum. The pilot is kept in `results/metrics/train_pilot/` and all 9 runs use **3e-5**, at
which the CE-only run stays flat (3.5661 / 3.0646): a fair control that neither helps nor hurts. Dropout 0.1 as in
Project 3. Every 250 updates the student is checked on **fixed** held-out states (identical for every run and every
check; update 0 = the teacher itself): `val_traj` (L_KD at T = 1 on 400 states of the 300 held-out trajectories),
`val_span_ce` (its span part), `val_sched_ce` (CE on 400 real validation windows masked like a 4/8/16-pass decoder),
and Project 3's own reconstruction protocol (`val_ce`, `val_acc`; uniform masking level).

**Files:** `configs/distill.yaml` (9 runs), `distill/train_student.py`, `scripts/train_all.sh`

**Run:**
```
python -m distill.train_student --run kd       # one run, ~26 min
bash scripts/train_all.sh                      # all 9 runs, ~3.3 h
```

**Result:** ✅ All 9 runs finished 1,500 updates (peak memory ~2.6 GB; 0.50 s/update CE-only, 1.03 s/update with the teacher).
Held-out numbers (lower is better except acc; update 0 = the untouched teacher):

| Run | val_traj | val_span_ce | val_sched_ce | val_ce | val_acc | time |
|---|---|---|---|---|---|---|
| *teacher (update 0)* | 1.3268 | 2.4265 | 3.5675 | 3.0596 | 0.4154 | – |
| ce (λ=0) | 1.3458 | 2.4415 | 3.5661 | 3.0646 | 0.4157 | 12 min |
| kd (λ=0.5, T=2) | 1.3397 | 2.3910 | 3.5839 | 3.0773 | 0.4142 | 26 min |
| kd_l25 | 1.3358 | 2.4060 | 3.5725 | 3.0686 | 0.4154 | 26 min |
| kd_l75 | 1.3495 | 2.3820 | 3.6066 | 3.0939 | 0.4127 | 26 min |
| kd_l100 (λ=1) | 1.3755 | 2.3821 | 3.6881 | 3.1620 | 0.4092 | 15 min |
| kd_T1 | 1.3200 | 2.4050 | 3.5782 | 3.0729 | 0.4143 | 26 min |
| kd_T4 | 1.3313 | 2.4008 | 3.5752 | 3.0703 | 0.4148 | 26 min |
| kd_state | 1.3331 | 2.4274 | 3.5692 | 3.0657 | 0.4147 | 26 min |
| kd_small (4 layers) | 1.4496 | 2.5356 | 3.7153 | 3.2347 | 0.3869 | 19 min |

What this says: the KD runs learn the thing they are taught (span CE falls from 2.43 to 2.38–2.41: they get better at
guessing *which token the teacher will fix during the next few passes*), and the more weight on the teacher, the more
they lose on real text (`val_ce` 3.06 → 3.16 at λ = 1). Same-state targets (`kd_state`) change almost nothing, as
predicted in Phase 7. CE-only training is flat. The small student loses ~0.17 nats of CE: removing two layers costs
more than 1,500 updates can repair. Curves: `results/figures/fig3_training_curves.png`.

---

## Phase 9 — Low-step student sampling

**Learn:** a student is decoded by **exactly the decoder the teacher uses** (Project 6's `decode_ids` takes any model
with the same interface). At each pass it predicts every masked position, chooses which ones to fix (random order,
never two neighbours in the same pass, linear quota so that S passes finish the window), draws their tokens (top-k
20), keeps the prompt, and stops early once the story's `<EOS>` is fixed with no `[MASK]` before it. It never unmasks
more positions than remain; the last pass fixes whatever is left.

**Files:** `distill/run.py` (system names `model@decoder[@sampling]`, generation, scoring, summaries)

**Run:** `python -m distill.run --split dev` (all systems of the dev grid), `--split test` (the selected ones).

**Result:** ✅ Dev grid (200 validation prompts, seed 0; scorer perplexity, lower = more fluent; Distinct-2 higher = more varied):

| System | 4 passes | 8 passes | 16 passes | D2 at 8 |
|---|---|---|---|---|
| teacher | 113.3 | 60.9 | 48.3 | 0.535 |
| teacher, temperature 0.8 | 83.9 | 43.9 | – | 0.491 |
| ce | 108.0 | 59.2 | 45.2 | 0.516 |
| kd | 91.6 | 50.8 | 39.0 | 0.480 |
| kd_l25 | 94.3 | 52.1 | 40.3 | 0.488 |
| kd_l75 | 90.8 | 50.9 | 37.6 | 0.471 |
| **kd_l100** | **88.9** | **49.0** | **37.6** | 0.467 |
| kd_T1 | 100.7 | 54.2 | 41.8 | 0.497 |
| kd_T4 | 95.7 | 51.5 | 41.9 | 0.488 |
| kd_state | 103.6 | 56.9 | 42.8 | 0.512 |
| kd_small | 114.4 | 65.7 | 53.3 | 0.466 |

(teacher at 32 passes: 40.2, D2 0.517; real continuations 4.4.) Every system passes the selection rule (repetitive
≤ 10%, Distinct-2 ≥ 0.40), so **kd_best = kd_l100** (lowest perplexity at 8 passes). Every KD run lowers perplexity,
but it also lowers Distinct-2 by roughly the same amount as simply sampling the teacher at temperature 0.8: the first
warning that KD may mostly be *sharpening* the distribution (tested properly in Phase 12, H8).

---

## Phase 10 — Functional correctness

**Files:** `tests/test_masking.py`, `tests/test_loss.py`, `tests/test_sampling.py`

**Run:** `python -m pytest -q`

**Check:** no causal mask (changing a *later* token changes the prediction at an *earlier* position); the prompt is
never changed; the number of `[MASK]` tokens falls at every pass; the decoder completes the window at 4, 8 and 16
passes and never needs more passes than its budget; same seed → same output; KL direction, temperature scaling, loss
only at masked positions; teacher frozen (no gradients, weights unchanged after an optimizer step) while the student's
weights change; the 4-layer student keeps the chosen teacher layers; state construction (see Phase 6).

**Result:** ✅ `27 passed` (`python -m pytest -q`, 6 s).

---


## Phase 11 — Comparison baselines

**Learn:** a fast student only means something next to the right baselines. Each one answers a different question:

| Baseline | Question it answers |
|---|---|
| teacher, 32 passes | the quality we want to keep (the "slow" reference) |
| teacher, 16 / 8 / 4 passes | what cutting passes costs with **no** training |
| CE-only student (`ce`) | what 1,500 more updates on real text give **without** the teacher |
| teacher at temperature 0.8 | what simply making the distribution **sharper** gives (no training at all) |
| teacher / student with confidence order | the guide's suggested reveal rule (Projects 3–6: it fails on this model) |
| 4-layer KD student | step distillation **plus** compression |
| real continuations | the ceiling (scorer perplexity of real TinyStories text) |

The temperature-0.8 baseline was added before training: a model trained to copy the teacher's *later* (more decided)
predictions might simply learn to be more confident, and a temperature knob does that for free.

**Files:** `configs/sampling.yaml`, `distill/select.py` (`results/metrics/selection.json`)

**Run:** `python -m distill.select` → `kd_best` and the list of test systems.

**Result:** ✅ 16 test systems: teacher at 32/16/8/4 passes, teacher at 8 passes with temperature 0.8, teacher with
confidence order at 8; `ce` at 16/8/4; `kd_best` (= `kd_l100`, chosen on dev only) at 16/8/4 and with confidence order;
`kd`, `kd_state`, `kd_small` at 8. Plus the real continuations.

---

## Phase 12 — Generation quality

**Learn:** every system writes a story for the same 400 test prompts with 3 seeds (1,200 stories). The scorer
(TinyStories-33M, a separate model) measures fluency as perplexity; Distinct-2 measures variety (share of different
word pairs in 100 stories of one group and seed); "repetitive" = a story repeats itself badly. Comparisons are
**paired** (same prompt, same seed) with a bootstrap 95% confidence interval of the perplexity ratio.

**Run:** `python -m distill.run --split test` (~45 min), `python -m distill.compare`

**Result:** ✅ Test set, 1,200 stories per system (`results/metrics/test_summary.json`, figure `fig1_quality_vs_passes.png`):

| System | Passes | Perplexity [95% CI] | Distinct-2 | Repetitive | Ended with `<EOS>` |
|---|---|---|---|---|---|
| teacher | 32 | 39.4 [38.2, 40.7] | 0.430 | 1.8% | 79% |
| teacher | 16 | 45.3 [43.7, 47.0] | 0.437 | 1.0% | 78% |
| teacher | 8 | 59.9 [57.9, 62.0] | 0.441 | 0.5% | 75% |
| teacher | 4 | 109.1 [105.5, 112.8] | 0.453 | 0.2% | 69% |
| teacher, temperature 0.8 | 8 | 44.0 [42.5, 45.4] | 0.402 | 2.7% | 76% |
| ce | 16 | 44.0 [42.5, 45.5] | 0.422 | 0.8% | 78% |
| ce | 8 | 57.4 [55.6, 59.3] | 0.426 | 0.2% | 76% |
| ce | 4 | 105.0 [101.6, 108.4] | 0.439 | 0.0% | 72% |
| **kd_l100** | 16 | **36.5** [35.4, 37.6] | 0.370 | 5.1% | 80% |
| **kd_l100** | 8 | **48.3** [46.7, 49.9] | 0.383 | 2.8% | 80% |
| **kd_l100** | 4 | **87.4** [84.4, 90.4] | 0.395 | 0.8% | 75% |
| kd (λ = 0.5) | 8 | 50.1 [48.5, 51.7] | 0.391 | 1.4% | 79% |
| real continuations | – | 4.5 | 0.541 | 6.2% | 100% |

(Distinct-2 is lower on test than on dev for **every** system, real text included (0.62 → 0.54), because a test cell
has 100 stories instead of 50: more stories share more word pairs.)

Hypotheses (written before training, `configs/hypotheses.md`); ratio < 1 = first system better:

| # | Comparison | Ratio [95% CI] | Verdict |
|---|---|---|---|
| H1 | teacher 8 vs teacher 32 | 1.520 [1.482, 1.558] | ✅ supported: cutting passes alone costs 52% |
| H2 | ce 8 vs teacher 8 | 0.958 [0.940, 0.978] | ✅ supported, but small (−4%) |
| H3 | kd 8 vs ce 8 / kd_l100 8 vs ce 8 | 0.873 [0.854, 0.891] / 0.841 [0.822, 0.861] | ✅ supported: the teacher's supervision adds −13 to −16% beyond extra training |
| H4 | kd_l100 8 vs teacher 16 | 1.065 [1.034, 1.097] | ✅ supported (upper bound ≤ 1.10): 8 student passes ≈ 16 teacher passes |
| H5 | kd_l100 4: better than teacher 4, and acceptable | 0.801 [0.782, 0.820]; Distinct-2 0.395 | ❌ not supported: much better, but Distinct-2 is just under the 0.40 rule on test (0.496 on dev) |
| H6 | kd (trajectory) 8 vs kd_state 8 | 0.890 [0.872, 0.907] | ✅ supported |
| H7 | speed-up, kd_l100 8 vs teacher 32 | 3.44× (Phase 13) | ✅ supported (≥ 2.8×) |
| H8 | kd_l100 8 vs teacher 8 at temperature 0.8 | 1.098 [1.074, 1.122]; Distinct-2 −0.020 | ❌ **not supported** |

**What this means, honestly.** Distillation clearly works *as measured by perplexity*: the KD student at 8 passes is
20% better than the teacher at 8 passes and as good as the teacher at 16 (2× fewer passes), and at 16 passes it even
beats the teacher at 32 (0.926 [0.901, 0.950]). But it pays in variety: Distinct-2 falls by 0.04–0.07 and repetition
rises a little. **H8 is the key negative result:** turning the teacher's temperature down to 0.8, with no training at
all, gives *lower* perplexity (44.0 vs 48.3) *and* more variety (0.402 vs 0.383). So most of what this distillation
buys is a sharper (more confident) distribution, which a temperature knob gives for free. Sharpening alone
(temperature 0.8 vs 1.0 at 8 passes) is worth 0.734 [0.720, 0.749].

---

## Phase 13 — Latency, throughput, memory

**Learn:** fewer passes should mean proportionally less time, but each pass also has fixed costs (sampling, choosing
positions, Python), so the speed-up is a little smaller than the pass ratio. Measured on this CPU (24 threads, fp32),
batch 1, 40 prompts (10 per group), after 2 warm-up stories; a run counts only if other programs used ≤ 15% of the CPU.

**Files:** `distill/benchmark.py`, `scripts/bench.sh`

**Run:** `bash scripts/bench.sh` (~15 min, on a quiet machine)

**Result:** ✅ (`results/metrics/timing.json`, `throughput.json`, `memory.json`; figure `fig2_latency_vs_quality.png`)

| System | Passes | Median latency (p95) | Speed-up vs teacher 32 | ms per pass | Perplexity (test) |
|---|---|---|---|---|---|
| teacher | 32 | 363 ms (374) | 1.00× | 11.2 | 39.4 |
| teacher | 16 | 190 ms (201) | 1.91× | 11.9 | 45.3 |
| teacher | 8 | 104 ms (112) | 3.47× | 13.0 | 59.9 |
| teacher | 4 | 63 ms (68) | 5.78× | 15.6 | 109.1 |
| ce | 8 | 103 ms (115) | 3.52× | 13.0 | 57.4 |
| **kd_l100** | 8 | **106 ms** (113) | **3.44×** | 13.1 | **48.3** |
| kd_l100 | 4 | 61 ms (67) | 5.93× | 15.4 | 87.4 |
| kd_small (4 layers) | 8 | 83 ms (91) | 4.39× | 10.3 | 65.6 |

A student pass costs exactly as much as a teacher pass (same network): the speed comes only from doing fewer passes.
3.44× of the ideal 4× (86%); the rest is fixed per-pass cost. Throughput (stories per second, 32 prompts): teacher 32
passes 2.8 / 3.7 / 3.3 at batch 1 / 8 / 32; kd_l100 8 passes 9.6 / 11.8 / 11.0; kd_l100 4 passes 15.7 / 18.3 / 17.5.
Batching helps only up to ~8 on this CPU. Memory: weights 65.0 MB (13.9 M parameters) for teacher and student,
51.5 MB (10.3 M) for the 4-layer student; whole process ~340 MB at batch 1, ~580–600 MB at batch 16; the student
uses less extra memory than the 32-pass teacher at batch 16 (143 vs 206 MB) only because its stories end in fewer
passes. Two batch-1 throughput runs stayed at 15.1–16.0% background load after a repeat; their values agree with the
first attempt within 3% and are kept (marked `reliable: false` in the file).

---

## Phase 14 — Ablations

**Learn:** change one thing at a time and see what moves. All ablation runs share the same seed and the same 1,500
updates; they are compared on the 200 **validation** prompts (the test set is used once, Phase 12).

**Result:** ✅ (figure `fig4_ablations.png`; dev perplexity at 8 passes; teacher 60.9, CE-only 59.2)

| Ablation | Setting → dev perplexity at 8 passes | Distinct-2 | Reading |
|---|---|---|---|
| KD weight λ (T = 2) | 0 → 59.2, 0.25 → 52.1, 0.5 → 50.8, 0.75 → 50.9, 1.0 → **49.0** | 0.516 → 0.467 | more teacher, lower perplexity, less variety |
| Temperature T (λ = 0.5) | 1 → 54.2, **2 → 50.8**, 4 → 51.5 | 0.497 / 0.480 / 0.488 | T = 2 best, as the guide suggests |
| Target type | same-state → 56.9, trajectory → **50.8** | 0.512 / 0.480 | test: 0.890 [0.872, 0.907] for trajectory (H6) |
| Student size | 6 layers → 50.8, 4 layers → 65.7 | 0.480 / 0.466 | 4 layers is worse than the untouched teacher (test 1.094 [1.069, 1.120]) |
| Reveal rule | random + no-adjacent 48.3 vs confidence order 2,124 (test) | | confidence order fails for students too (44× worse) |

Reconstruction (`python -m distill.recon`, 800 test windows; `results/metrics/reconstruction.json`): masked-token CE
on **real** text, masked like a 8-pass decoder: teacher 3.532, ce 3.531, kd 3.545, kd_l100 3.644, kd_small 3.683.
The KD students are **worse** at predicting real hidden words, yet write stories the scorer prefers: they have moved
toward the teacher's own (more predictable) style. A useful reminder that reconstruction loss and generation quality
are not the same thing.

Per prompt group (figure `fig5_divergence.png`, ratio to teacher at 32 passes): at 8 passes the teacher loses most on
**short** prompts (1.58) and KD helps there most (1.25); on **longer** prompts the gain is smallest (1.47 → 1.29).

---

## Phase 15 — Error analysis

**Learn:** numbers hide *how* stories go wrong. 20 test prompts (5 per group) × 4 systems = 80 stories were shuffled,
given neutral ids, and labelled **blind** (the labeller saw no system names) on a 1–5 rubric and 7 failure types.

**THE LABELS WERE PRODUCED BY AN AI ASSISTANT (Claude, run as a separate agent that saw only `blind_stories.md`), not
by human annotators, and are not human-verified. One annotator, so there is no inter-annotator agreement; 20 stories
per system is a small sample.**

**Run:** `python -m distill.errors --stage sample` → labels → `--stage count` → `results/metrics/error_analysis.md`

**Result:** ✅ (figure `fig6_errors.png`)

| Measure | teacher 32 | teacher 8 | ce 8 | kd_l100 8 |
|---|---|---|---|---|
| Rubric overall (1–5) | 2.33 | 1.90 | 1.78 | 1.97 |
| coherence / grammar | 1.75 / 1.95 | 1.25 / 1.30 | 1.15 / 1.20 | 1.20 / 1.30 |
| completeness / consistency | 2.75 / 2.10 | 2.25 / 1.85 | 2.15 / 1.75 | 2.50 / 2.10 |
| repetition | 6/20 | 10/20 | 12/20 | 11/20 |
| early wrong turn | 4/20 | 4/20 | 6/20 | 7/20 |
| premature ending | 2/20 | 6/20 | 4/20 | 1/20 |
| entity changes | 15/20 | 14/20 | 15/20 | 12/20 |
| vs teacher 32 on the same prompt: worse / same / better | – | 14/3/3 | 15/2/3 | 14/2/4 |

All four systems write weak stories by human standards (a 13.9 M model; most stories contain a contradiction and an
ungrammatical sentence). Cutting 32 → 8 passes adds repetition and unfinished endings. The KD student finishes stories
better (1 premature ending vs 6) and keeps characters more consistent, but it does not repair coherence or grammar,
and on the same prompt it is still worse than the 32-pass teacher 14 times out of 20. This fits Phase 12: the scorer's
gain is real, but it is a gain in confidence and tidiness more than in story logic.

---

## Phase 16 — Repository and reproducibility

**Files:** `README.md` (overview, results, how to reproduce), this file, `configs/` (all settings and the
hypotheses, written before training), `requirements.txt` + `requirements.lock.txt`, `scripts/` (`train_all.sh`,
`stage2.sh`, `bench.sh`, `check_reproduce.py`), `results/` (metrics, figures, error labels; the stories and scores of
every system).

**Reproduce everything** (CPU, ~7 h in total):
```
python -m distill.prompts && python -m distill.trajectories       # Phase 5   ~30 min
bash scripts/train_all.sh                                        # Phase 8   ~3.3 h
python -m distill.run --split dev && python -m distill.recon     # Phase 9, 14   ~1 h
python -m distill.select && python -m distill.run --split test   # Phase 11, 12  ~45 min
python -m distill.compare
bash scripts/bench.sh                                            # Phase 13  ~15 min (quiet machine)
python -m distill.errors --stage sample   (labels)   --stage count   --stage divergence
python -m distill.plots
python -m pytest -q
```
Checkpoints (`checkpoints/<run>/student.pt`, 9 × 54 MB) are kept locally and are in `.gitignore`.

**Result:** ✅ Every random choice has a fixed seed (prompts 7070, trajectories 7000 + batch, training 7007,
generation 6000 + 100·seed + batch, error sample 2029); the teacher's reproduction check gives 200/200 identical
stories to Project 6; 27 tests pass.
