#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-/home/pavt1024/miniconda3/envs/deeplearning/bin/python}
exec "$PY" scripts/run_herp_suite.py "$@"
