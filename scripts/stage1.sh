#!/usr/bin/env bash
# Teacher dev baselines + Phase 1 reproduction check, then all training runs.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
$PY -m distill.run --split dev --names "teacher@full-random-K16+noadj+stop" "teacher@full-random-K4+noadj+stop+lazy" "teacher@full-random-K8+noadj+stop+lazy" "teacher@full-random-K16+noadj+stop+lazy" "teacher@full-random-K32+noadj+stop+lazy" "teacher@full-random-K8+noadj+stop+lazy@topk20t0.8" "teacher@full-random-K4+noadj+stop+lazy@topk20t0.8" "teacher@full-conf-K8+stop+lazy" > results/metrics/dev_teacher.log 2>&1
$PY scripts/check_reproduce.py >> results/metrics/dev_teacher.log 2>&1
bash scripts/train_all.sh
