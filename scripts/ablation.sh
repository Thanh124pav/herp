#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-/home/pavt1024/miniconda3/envs/deeplearning/bin/python}
TASK=${TASK:-PickCube-v1}
STEPS=${STEPS:-300000}
for seed in 0 1 2; do
  for estimator in cosine dot fisher occupancy; do
    "$PY" train.py --env-id "$TASK" --method herp --p-estimator "$estimator" \
      --seed "$seed" --total-timesteps "$STEPS" --output-dir "outputs/ablation_p/$estimator" "$@"
  done
  for estimator in pairwise branch return; do
    "$PY" train.py --env-id "$TASK" --method herp --sigma-estimator "$estimator" \
      --seed "$seed" --total-timesteps "$STEPS" --output-dir "outputs/ablation_sigma/$estimator" "$@"
  done
done
