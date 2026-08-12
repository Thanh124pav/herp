"""Mild controlled per-policy environment/dynamics perturbations (PLAN.md 1.1).

Policies in the population differ by seed and by *mild* dynamics perturbation.
Keeping perturbations small but distinct is what produces complementary
competence (Gate C, PLAN.md section 25) without changing the task identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PerturbationSpec:
    """Per-policy dynamics knobs. Defaults = the unperturbed nominal dynamics."""

    move_speed: float = 1.0  # control gain on end-effector motion
    grasp_ease: float = 1.0  # multiplier on effective grasp radius

    @staticmethod
    def for_policy(policy_id: int, n_policies: int, strength: float = 0.25) -> "PerturbationSpec":
        """Deterministically spread policies across a small perturbation range.

        Policy 0 stays near nominal; others get mild, distinct biases toward being
        better at either the reaching/transport phase (higher move_speed) or the
        grasping phase (higher grasp_ease).
        """
        if n_policies <= 1:
            return PerturbationSpec()
        # phase in [-1, 1] across the population
        phase = 2.0 * policy_id / (n_policies - 1) - 1.0
        return PerturbationSpec(
            move_speed=float(1.0 + strength * phase),
            grasp_ease=float(1.0 - strength * phase),
        )
