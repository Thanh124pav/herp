from __future__ import annotations

import csv
from pathlib import Path


class CsvLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: dict) -> None:
        row = dict(row)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})
