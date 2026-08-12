"""Running feature normalization (PLAN.md sections 6.1, 7.1).

State-transition features and experience descriptors are normalized per-dimension
over the replay dataset. We keep a Welford-style running estimator so the
segmenter/grouper can normalize online without re-scanning all data.
"""

from __future__ import annotations

import numpy as np


class RunningNormalizer:
    """Per-dimension running mean/std normalizer (Welford's algorithm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = int(dim)
        self.eps = float(eps)
        self.count = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._m2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a batch ``x`` of shape [N, dim] or [dim]."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        for row in x:
            self.count += 1
            delta = row - self._mean
            self._mean += delta / self.count
            self._m2 += delta * (row - self._mean)

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.dim, dtype=np.float64)
        return np.sqrt(self._m2 / (self.count - 1)) + self.eps

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "eps": self.eps,
            "count": self.count,
            "mean": self._mean.copy(),
            "m2": self._m2.copy(),
        }
