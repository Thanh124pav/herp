"""Population evaluation helpers (PLAN.md sections 16, 19).

Thin wrappers producing population mean / worst / best performance (the metrics
the routing comparison in section 19.4 reports).
"""

from __future__ import annotations

import numpy as np


def summarize_population(eval_results: dict[int, dict[str, float]]) -> dict:
    succ = np.array([v["success_rate"] for v in eval_results.values()])
    ret = np.array([v["mean_return"] for v in eval_results.values()])
    return {
        "mean_success": float(succ.mean()),
        "worst_success": float(succ.min()),
        "best_success": float(succ.max()),
        "mean_return": float(ret.mean()),
        "worst_return": float(ret.min()),
        "best_return": float(ret.max()),
    }
