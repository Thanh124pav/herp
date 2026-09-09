#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-/home/pavt1024/miniconda3/envs/deeplearning/bin/python}
"$PY" -m pytest tests/test_herp_core.py tests/test_herp_regressions.py -q
"$PY" scripts/check_snapshot.py
exec "$PY" train.py --method herp --total-timesteps 2048 --rollout-horizon 1024 \
  --reference-horizon 64 --max-candidates 2 --eval-episodes 2 --eval-interval 2048 \
  --output-dir outputs/herp_smoke "$@"
