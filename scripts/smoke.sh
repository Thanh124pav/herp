#!/usr/bin/env bash
# Fast local smoke: unit tests + ManiSkill snapshot replay + a short HERP run.
# Override PY=/path/to/python if the default `python` on PATH is wrong.
set -euo pipefail
PY=${PY:-python}
BENCHMARK=${BENCHMARK:-maniskill}
ENV_ID=${ENV_ID:-PushCube-v1}
"$PY" -m pytest tests/test_herp_core.py tests/test_herp_regressions.py -q
"$PY" scripts/check_restore.py --benchmark "$BENCHMARK" --env-id "$ENV_ID"
exec "$PY" train.py --benchmark "$BENCHMARK" --env-id "$ENV_ID" --method herp \
  --total-timesteps 2048 --rollout-horizon 1024 --reference-horizon 64 \
  --max-candidates 2 --eval-episodes 2 --eval-interval 2048 \
  --output-dir outputs/herp_smoke "$@"
