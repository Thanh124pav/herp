#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-/home/pavt1024/miniconda3/envs/deeplearning/bin/python}
exec "$PY" analysis/mechanisms.py --kind sigma "$@"
