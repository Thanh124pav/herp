"""Per-(donor, experience) exemplar chunk bank (PLAN.md Module G, section 11).

For each ``(donor policy d, experience e)`` we keep the successful exemplar
chunks the donor produced. When the router asks for ``m`` units from ``(d, e)``,
we select exemplars by:

    1. successful episode only;
    2. closest to the experience prototype;
    3. (tie-break) higher return-to-go.
"""

from __future__ import annotations

import numpy as np

from .grouper import chunk_experience_distance
from .trajectory import Chunk
from .vocabulary import ExperienceVocabulary


class ExperienceBank:
    def __init__(self, max_chunks_per_experience: int = 200):
        self.max_per = int(max_chunks_per_experience)
        # (donor_id, experience_id) -> list[Chunk]
        self._store: dict[tuple[int, int], list[Chunk]] = {}
        self._by_id: dict[int, Chunk] = {}

    def add(self, experience_id: int, chunk: Chunk) -> None:
        chunk.experience_id = experience_id
        self._by_id[chunk.chunk_id] = chunk
        key = (chunk.policy_id, experience_id)
        bucket = self._store.setdefault(key, [])
        bucket.append(chunk)
        if len(bucket) > self.max_per:
            bucket.pop(0)

    def get(self, chunk_id: int) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def successful_chunks(self, donor_id: int, experience_id: int) -> list[Chunk]:
        return [c for c in self._store.get((donor_id, experience_id), []) if c.success_episode]

    def n_successful(self, donor_id: int, experience_id: int) -> int:
        return len(self.successful_chunks(donor_id, experience_id))

    def select(
        self,
        donor_id: int,
        experience_id: int,
        vocabulary: ExperienceVocabulary,
        m: int,
    ) -> list[Chunk]:
        """Choose the ``m`` best exemplar chunks from ``(donor, experience)``."""
        cands = self.successful_chunks(donor_id, experience_id)
        if not cands or m <= 0:
            return []
        exp = vocabulary.experiences.get(experience_id)
        if exp is None:
            return cands[:m]

        def score(c: Chunk):
            dist = chunk_experience_distance(c, exp)
            rtg = float(np.sum(c.rewards)) if len(c.rewards) else 0.0
            return (dist, -rtg)  # nearest first, then higher return

        return sorted(cands, key=score)[:m]
