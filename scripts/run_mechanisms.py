"""Thin wrapper around ``analysis/mechanisms.py`` for the IMPL §32 CLI shape."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "analysis" / "mechanisms.py"), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
