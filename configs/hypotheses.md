# Project 7 hypotheses (written before any student was trained)

All quality comparisons are paired over the 400 test prompts × 3 seeds, with the paired bootstrap of Project 4
(`bench.evaluate_quality.paired_diff`, 10,000 resamples). Quality = scorer perplexity (TinyStories-33M); a ratio < 1
means the first system is better. "Better" = the 95% CI of the ratio lies below 1.

| # | Hypothesis | Test | Supported if |
|---|---|---|---|
| H1 | Cutting passes without retraining costs quality | teacher K8 vs teacher K32 | ratio > 1, CI above 1 |
| H2 | Extra training alone (CE on real data, student-pass masking) helps at 8 passes | ce K8 vs teacher K8 | CI below 1 |
| H3 | Teacher supervision helps beyond extra training (main question) | kd K8 vs ce K8 (pre-registered run), and kd_best K8 vs ce K8 | CI below 1 |
| H4 | A distilled student at 8 passes is as good as the teacher at 16 | kd_best K8 vs teacher K16 | CI upper bound ≤ 1.10 (half: point estimate ≤ 1.10) |
| H5 | 4 passes become usable | kd_best K4: acceptable (repetitive ≤ 10%, Distinct-2 ≥ 0.40) and better than teacher K4 | both |
| H6 | Trajectory targets (teacher's later state) beat same-state targets | kd K8 vs kd_state K8 | CI below 1 |
| H7 | Fewer passes give a proportional wall-clock speed-up | median latency, student K8 vs teacher K32 (batch 1) | speed-up ≥ 2.8× (70% of the ideal 4×) |
| H8 | Distillation does more than sharpen the distribution | kd_best K8 vs teacher K8 at temperature 0.8 | ppl CI below 1 and Distinct-2 not lower by more than 0.005 |

Expected from theory (stated in advance): a same-size student that starts as an exact copy of the teacher gets
**almost no gradient** from same-state soft targets at the start of training: its predictions equal the teacher's, and
the only difference comes from the student's dropout noise. So `kd_state` can mostly drift only through the CE half of
the loss. Trajectory targets differ from the teacher's own one-pass
prediction, so they are the part that can change the student.
