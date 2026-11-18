#!/usr/bin/env bash
# Train every run of configs/distill.yaml one after the other (CPU, ~30 min each). Logs: results/metrics/train/<run>.log
# Usage (from project7):  bash scripts/train_all.sh [run ...]
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python
RUNS="${*:-ce kd kd_l25 kd_l75 kd_l100 kd_T1 kd_T4 kd_state kd_small}"
mkdir -p results/metrics/train
for r in $RUNS; do
  if [ -f "checkpoints/$r/student.pt" ]; then echo "skip $r (done)"; continue; fi
  echo "=== $r $(date +%H:%M:%S)"
  PYTHONIOENCODING=utf-8 "$PY" -m distill.train_student --run "$r" > "results/metrics/train/$r.log" 2>&1
  tail -n 8 "results/metrics/train/$r.log"
done
echo "=== all done $(date +%H:%M:%S)"
