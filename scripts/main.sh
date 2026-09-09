#!/usr/bin/env bash
# Launch the full benchmark suite. See scripts/run_suite.py for options.
set -euo pipefail
PY=${PY:-python}
exec "$PY" scripts/run_suite.py "$@"
