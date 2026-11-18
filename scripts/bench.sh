#!/usr/bin/env bash
# Phase 13: speed and memory. Run on a quiet machine; runs with other CPU load > 15% are marked unreliable and redone.
set -e
cd "$(dirname "$0")/.."
source .venv/Scripts/activate
R="full-random"
NAMES="teacher@$R-K32+noadj+stop+lazy teacher@$R-K16+noadj+stop+lazy teacher@$R-K8+noadj+stop+lazy teacher@$R-K4+noadj+stop+lazy ce@$R-K8+noadj+stop+lazy kd_l100@$R-K8+noadj+stop+lazy kd_l100@$R-K4+noadj+stop+lazy kd_small@$R-K8+noadj+stop+lazy"
python -m distill.benchmark --timing --names $NAMES
python -m distill.benchmark --throughput --names teacher@$R-K32+noadj+stop+lazy kd_l100@$R-K8+noadj+stop+lazy kd_l100@$R-K4+noadj+stop+lazy kd_small@$R-K8+noadj+stop+lazy
python -m distill.benchmark --memory --names teacher@$R-K32+noadj+stop+lazy kd_l100@$R-K8+noadj+stop+lazy kd_small@$R-K8+noadj+stop+lazy
echo BENCH DONE
