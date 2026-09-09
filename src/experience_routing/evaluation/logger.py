"""Run logging (PLAN.md section 21).

A thin logger that mirrors every metric to **Weights & Biases** (preferred) and,
regardless of W&B availability, to a local **JSONL** stream + a final
``summary.json`` -- so a run is always reviewable later even offline or without a
W&B account.

Keys follow the section-21 taxonomy with ``group/name`` naming
(``return/policy_0``, ``routing/route_count``, ``compute/routing_time`` ...), so
the W&B UI groups them automatically.

Usage::

    logger = RunLogger(project="experience-routing", run_name="uot_seed0",
                       config=vars(cfg), use_wandb=True)
    logger.log({"success/policy_0": 0.3}, step=10_000)
    logger.summary({"final/mean_success": 0.25})
    logger.finish()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(
        self,
        project: str = "experience-routing",
        run_name: str | None = None,
        config: dict | None = None,
        use_wandb: bool = True,
        wandb_mode: str = "online",
        outdir: str | Path = "outputs",
        group: str | None = None,
    ):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or f"run_{int(time.time())}"
        self._jsonl_path = self.outdir / "log.jsonl"
        self._jsonl = self._jsonl_path.open("w")
        self._summary: dict[str, Any] = {}
        self.wandb = None

        if use_wandb:
            try:
                import wandb

                self.wandb = wandb.init(
                    project=project, name=self.run_name, config=config or {},
                    mode=wandb_mode, group=group, dir=str(self.outdir),
                    reinit=True,
                )
            except Exception as exc:  # pragma: no cover - env/network dependent
                print(f"[logger] W&B disabled ({exc}); logging locally to "
                      f"{self._jsonl_path}")
                self.wandb = None
        # always persist the config for later review
        if config is not None:
            (self.outdir / "config.json").write_text(
                json.dumps(config, indent=2, default=str)
            )

    def log(self, data: dict, step: int | None = None) -> None:
        """Log a flat ``group/name -> value`` dict at ``step`` (env steps)."""
        clean = {k: _to_scalar(v) for k, v in data.items()}
        if self.wandb is not None:
            self.wandb.log(clean, step=step)
        record = {"step": step, **clean}
        self._jsonl.write(json.dumps(record, default=float) + "\n")
        self._jsonl.flush()

    def summary(self, data: dict) -> None:
        """Set run-level summary metrics (final results, PLAN.md section 28)."""
        for k, v in data.items():
            self._summary[k] = _to_scalar(v)
            if self.wandb is not None:
                self.wandb.summary[k] = self._summary[k]
        (self.outdir / "summary.json").write_text(
            json.dumps(self._summary, indent=2, default=float)
        )

    def log_artifact_file(self, path: str | Path) -> None:
        """Attach a produced artifact (png/json) to the W&B run if available."""
        if self.wandb is None:
            return
        try:  # pragma: no cover - optional
            import wandb

            art = wandb.Artifact(f"{self.run_name}-artifacts", type="run-output")
            art.add_file(str(path))
            self.wandb.log_artifact(art)
        except Exception:
            pass

    def finish(self) -> None:
        self._jsonl.close()
        if self.wandb is not None:
            self.wandb.finish()


def _to_scalar(v: Any) -> Any:
    """Coerce numpy scalars/arrays to plain Python for JSON + W&B."""
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    return v
