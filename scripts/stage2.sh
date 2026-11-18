#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
bash scripts/train_all.sh
.venv/Scripts/python.exe -m distill.run --split dev > results/metrics/dev.log 2>&1
.venv/Scripts/python.exe -m distill.recon > results/metrics/recon.log 2>&1
echo STAGE2 DONE
