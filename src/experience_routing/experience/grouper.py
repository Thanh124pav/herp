"""State-transition experience descriptors and distance (PLAN.md Module C, section 7).

Grouping is separate from segmentation: segmentation says *where* a candidate
experience starts/ends; grouping says *which chunks instantiate the same
functional experience*. The descriptor is (precondition, effect) on normalized
task-relevant state -- no learned encoder (PLAN.md section 7.1).
"""

from __future__ import annotations

import numpy as np

from .trajectory import Chunk, Experience


def precondition_effect(chunk: Chunk) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(chunk.pre_state), np.asarray(chunk.effect)


def experience_distance(
    pre_a: np.ndarray,
    eff_a: np.ndarray,
    pre_b: np.ndarray,
    eff_b: np.ndarray,
    alpha_pre: float = 0.5,
    alpha_eff: float = 0.5,
) -> float:
    """Weighted precondition + effect L2 distance (PLAN.md section 7.2)."""
    d_pre = float(np.linalg.norm(pre_a - pre_b))
    d_eff = float(np.linalg.norm(eff_a - eff_b))
    return alpha_pre * d_pre + alpha_eff * d_eff


def chunk_experience_distance(chunk: Chunk, exp: Experience, **kw) -> float:
    pre, eff = precondition_effect(chunk)
    return experience_distance(pre, eff, exp.precondition_center, exp.effect_center, **kw)
