"""Valid / useful experience filter (PLAN.md Module D, section 8).

Hypothesis: a functional experience is more likely to be useful if it repeatedly
occurs in successful trajectories. An experience is *valid* when it has both
enough successful-episode support and a high enough success-support fraction.

Terminology per PLAN.md section 8: "experience validity / success support" --
this is NOT called receiver-specific utility.
"""

from __future__ import annotations

from dataclasses import dataclass

from .vocabulary import ExperienceVocabulary


@dataclass
class ValidityConfig:
    min_success_support: int = 5
    success_support_threshold: float = 0.05


class ValidityFilter:
    def __init__(self, config: ValidityConfig | None = None, latch: bool = True):
        self.config = config or ValidityConfig()
        self.latch = bool(latch)  # once valid, stay valid (stable routed set)
        # experience_id -> set of successful episode keys (policy_id, episode_id)
        self._success_episodes: dict[int, set[tuple[int, int]]] = {}
        self._total_success_episodes: set[tuple[int, int]] = set()
        self._latched: set[int] = set()

    def observe_chunk(self, experience_id: int, policy_id: int, episode_id: int, success: bool) -> None:
        if not success:
            return
        key = (policy_id, episode_id)
        self._success_episodes.setdefault(experience_id, set()).add(key)
        self._total_success_episodes.add(key)

    def success_support(self, experience_id: int) -> float:
        """Fraction of successful episodes that contain the experience (section 8)."""
        total = len(self._total_success_episodes)
        if total == 0:
            return 0.0
        return len(self._success_episodes.get(experience_id, set())) / total

    def n_success_support(self, experience_id: int) -> int:
        return len(self._success_episodes.get(experience_id, set()))

    def is_valid(self, experience_id: int) -> bool:
        cfg = self.config
        return (
            self.n_success_support(experience_id) >= cfg.min_success_support
            and self.success_support(experience_id) >= cfg.success_support_threshold
        )

    def update(self, vocabulary: ExperienceVocabulary) -> list[int]:
        """Refresh the ``valid`` flag on every experience; return valid ids.

        With ``latch`` (default), an experience that once met the criteria stays
        valid -- a repeatedly-useful functional experience does not stop being one
        just because the pool of successful episodes later grows.
        """
        valid_ids = []
        for eid, exp in vocabulary.experiences.items():
            exp.n_success_episodes = self.n_success_support(eid)
            currently = self.is_valid(eid)
            if currently:
                self._latched.add(eid)
            exp.valid = currently or (self.latch and eid in self._latched)
            if exp.valid:
                valid_ids.append(eid)
        return valid_ids
