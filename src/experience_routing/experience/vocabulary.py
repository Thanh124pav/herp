"""Online dynamic experience vocabulary (PLAN.md Module C, section 7.3).

Threshold-based online clustering of chunks into experience prototypes:

    * assign a chunk to the nearest prototype if within ``eps_experience``,
      else create a new experience;
    * update prototypes with incremental means;
    * periodically merge prototypes closer than ``eps_merge`` to avoid
      vocabulary fragmentation.

No learned latent -- prototypes are running means of (precondition, effect).
"""

from __future__ import annotations

import numpy as np

from .grouper import experience_distance
from .trajectory import Chunk, Experience


class ExperienceVocabulary:
    def __init__(
        self,
        eps_experience: float,
        eps_merge: float | None = None,
        alpha_pre: float = 0.5,
        alpha_eff: float = 0.5,
    ):
        self.eps_experience = float(eps_experience)
        self.eps_merge = float(eps_merge if eps_merge is not None else 0.7 * eps_experience)
        self.alpha_pre = float(alpha_pre)
        self.alpha_eff = float(alpha_eff)
        self.experiences: dict[int, Experience] = {}
        self.merge_count = 0
        self._next_id = 0
        # remap of merged experience ids -> surviving id (for external references)
        self.remap: dict[int, int] = {}
        # parallel arrays for vectorized nearest-prototype search (hot path)
        self._ids: list[int] = []  # row index -> experience_id
        self._row: dict[int, int] = {}  # experience_id -> row index
        self._pre: np.ndarray | None = None  # [K, d]
        self._eff: np.ndarray | None = None  # [K, d]

    def __len__(self) -> int:
        return len(self.experiences)

    def _kw(self) -> dict:
        return {"alpha_pre": self.alpha_pre, "alpha_eff": self.alpha_eff}

    def _create(self, chunk: Chunk) -> int:
        eid = self._next_id
        self._next_id += 1
        pre = np.asarray(chunk.pre_state, dtype=np.float64).copy()
        eff = np.asarray(chunk.effect, dtype=np.float64).copy()
        exp = Experience(
            experience_id=eid,
            precondition_center=pre,
            effect_center=eff,
            n_chunks=1,
            n_success_chunks=int(chunk.success_episode),
            donor_chunk_ids={chunk.policy_id: [chunk.chunk_id]},
        )
        self.experiences[eid] = exp
        # append to parallel arrays
        self._row[eid] = len(self._ids)
        self._ids.append(eid)
        self._pre = pre[None, :] if self._pre is None else np.vstack([self._pre, pre])
        self._eff = eff[None, :] if self._eff is None else np.vstack([self._eff, eff])
        chunk.experience_id = eid
        return eid

    def _update_prototype(self, exp: Experience, chunk: Chunk) -> None:
        """Incremental-mean prototype update (PLAN.md section 7.3)."""
        n = exp.n_chunks
        w = 1.0 / (n + 1)
        exp.precondition_center += w * (np.asarray(chunk.pre_state) - exp.precondition_center)
        exp.effect_center += w * (np.asarray(chunk.effect) - exp.effect_center)
        exp.n_chunks += 1
        exp.n_success_chunks += int(chunk.success_episode)
        exp.donor_chunk_ids.setdefault(chunk.policy_id, []).append(chunk.chunk_id)
        r = self._row[exp.experience_id]
        self._pre[r] = exp.precondition_center
        self._eff[r] = exp.effect_center

    def _rebuild_arrays(self) -> None:
        self._ids = list(self.experiences.keys())
        self._row = {eid: r for r, eid in enumerate(self._ids)}
        if self._ids:
            self._pre = np.array([self.experiences[i].precondition_center for i in self._ids])
            self._eff = np.array([self.experiences[i].effect_center for i in self._ids])
        else:
            self._pre = self._eff = None

    def assign(self, chunk: Chunk) -> int:
        """Assign a chunk to an experience, creating one if needed (section 7.3).

        Vectorized nearest-prototype search over the parallel center arrays.
        """
        if not self.experiences:
            return self._create(chunk)
        pre = np.asarray(chunk.pre_state)
        eff = np.asarray(chunk.effect)
        d = self.alpha_pre * np.linalg.norm(self._pre - pre, axis=1) + \
            self.alpha_eff * np.linalg.norm(self._eff - eff, axis=1)
        j = int(np.argmin(d))
        if d[j] <= self.eps_experience:
            exp = self.experiences[self._ids[j]]
            self._update_prototype(exp, chunk)
            chunk.experience_id = exp.experience_id
            return exp.experience_id
        return self._create(chunk)

    def merge_close(self) -> int:
        """Merge prototype pairs closer than ``eps_merge``. Returns #merges done."""
        merged_now = 0
        changed = True
        while changed:
            changed = False
            ids = list(self.experiences.keys())
            for a_i in range(len(ids)):
                for b_i in range(a_i + 1, len(ids)):
                    ea, eb = self.experiences.get(ids[a_i]), self.experiences.get(ids[b_i])
                    if ea is None or eb is None:
                        continue
                    d = experience_distance(
                        ea.precondition_center, ea.effect_center,
                        eb.precondition_center, eb.effect_center, **self._kw(),
                    )
                    if d <= self.eps_merge:
                        self._merge(ea, eb)
                        merged_now += 1
                        self.merge_count += 1
                        changed = True
                        break
                if changed:
                    break
        if merged_now:
            self._rebuild_arrays()
        return merged_now

    def _merge(self, keep: Experience, drop: Experience) -> None:
        na, nb = keep.n_chunks, drop.n_chunks
        tot = max(na + nb, 1)
        keep.precondition_center = (na * keep.precondition_center + nb * drop.precondition_center) / tot
        keep.effect_center = (na * keep.effect_center + nb * drop.effect_center) / tot
        keep.n_chunks += drop.n_chunks
        keep.n_success_chunks += drop.n_success_chunks
        keep.n_success_episodes += drop.n_success_episodes
        for pid, cids in drop.donor_chunk_ids.items():
            keep.donor_chunk_ids.setdefault(pid, []).extend(cids)
        self.remap[drop.experience_id] = keep.experience_id
        del self.experiences[drop.experience_id]

    def resolve(self, eid: int) -> int:
        """Follow merge remaps to the surviving experience id."""
        seen = set()
        while eid in self.remap and eid not in seen:
            seen.add(eid)
            eid = self.remap[eid]
        return eid
