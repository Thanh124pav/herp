#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python}
exec "$PY" scripts/run_mechanisms.py --kind p "$@"
