"""EXTRACT-style state-transition trajectory segmentation (PLAN.md Module B, section 6).

Pipeline (per PLAN.md section 6.2), a naive baseline -- *not* a research
contribution:

    per-step state-transition features g_t
        -> K-means labels
        -> median temporal smoothing
        -> split where the smoothed label changes
        -> merge chunks shorter than min_chunk_len

We fit K-means on a pool of features and then segment individual trajectories.
Only task-relevant state dims (``EnvSpec.task_features``) feed the features, and
features are normalized per-dim over the dataset (PLAN.md section 6.1).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt
from sklearn.cluster import KMeans

from ..envs.base import EnvSpec
from ..utils.normalization import RunningNormalizer
from .trajectory import Chunk, Trajectory


def step_features(task_feats: np.ndarray) -> np.ndarray:
    """Per-timestep state-transition feature g_t (PLAN.md section 6.1).

    ``task_feats`` is [T+1, task_dim]; returns [T, task_dim] of consecutive
    deltas (delta ee / object / gripper / object-goal / contact).
    """
    task_feats = np.asarray(task_feats, dtype=np.float64)
    return np.diff(task_feats, axis=0)


def _boundaries_from_labels(labels: np.ndarray) -> np.ndarray:
    """Indices (into the T-step axis) where the smoothed label changes."""
    if len(labels) <= 1:
        return np.array([], dtype=int)
    return np.where(labels[1:] != labels[:-1])[0] + 1


def merge_short_chunks(boundaries: list[int], n_steps: int, min_chunk_len: int) -> list[int]:
    """Drop boundaries that would create a chunk shorter than ``min_chunk_len``.

    Returns interior boundaries (exclusive of 0 and n_steps). Guarantees every
    resulting chunk has length >= min_chunk_len (except possibly a single whole
    trajectory shorter than that).
    """
    segs = [0, *sorted(b for b in boundaries if 0 < b < n_steps), n_steps]
    merged = [0]
    for b in segs[1:]:
        if b - merged[-1] < min_chunk_len and b != n_steps:
            continue  # skip this boundary: too short a chunk
        if b == n_steps and b - merged[-1] < min_chunk_len and len(merged) > 1:
            merged.pop()  # absorb trailing short chunk into previous one
        merged.append(b)
    return merged[1:-1]  # interior boundaries only


class Segmenter:
    def __init__(
        self,
        spec: EnvSpec,
        k_seg: int = 6,
        median_window: int = 7,
        min_chunk_len: int = 4,
        seed: int = 0,
    ):
        self.spec = spec
        self.k_seg = int(k_seg)
        self.median_window = int(median_window) | 1  # medfilt needs odd window
        self.min_chunk_len = int(min_chunk_len)
        self.seed = int(seed)
        # Two distinct normalizers (fit on different quantities):
        #   feat_normalizer -> per-step deltas g_t, feeds K-means (section 6.1);
        #   normalizer      -> absolute task-relevant states, feeds the
        #                      (precondition, effect) descriptors and competence
        #                      opportunity/effect matching (sections 7.1, 9).
        self.feat_normalizer = RunningNormalizer(spec.task_dim)
        self.normalizer = RunningNormalizer(spec.task_dim)
        self.kmeans: KMeans | None = None
        self._next_chunk_id = 0

    # -- fitting -----------------------------------------------------------
    def fit(self, trajectories: list[Trajectory]) -> "Segmenter":
        """Fit both normalizers + K-means from a pool of trajectories."""
        feats, states = [], []
        for traj in trajectories:
            tf = self.spec.task_features(traj.states)
            states.append(tf)
            g = step_features(tf)
            if len(g):
                feats.append(g)
        if not feats:
            return self
        pooled = np.concatenate(feats, axis=0)
        self.feat_normalizer.update(pooled)
        self.normalizer.update(np.concatenate(states, axis=0))
        norm = self.feat_normalizer.normalize(pooled)
        k = min(self.k_seg, len(norm))
        self.kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init=10).fit(norm)
        return self

    # -- segmentation ------------------------------------------------------
    def _labels_for(self, traj: Trajectory) -> np.ndarray:
        tf = self.spec.task_features(traj.states)
        g = step_features(tf)
        if self.kmeans is None or len(g) == 0:
            return np.zeros(len(g), dtype=int)
        labels = self.kmeans.predict(self.feat_normalizer.normalize(g))
        win = min(self.median_window, len(labels) if len(labels) % 2 == 1 else len(labels) - 1)
        if win >= 3:
            labels = medfilt(labels, kernel_size=win).astype(int)
        return labels

    def segment(self, traj: Trajectory) -> list[Chunk]:
        """Split one trajectory into contiguous, non-overlapping chunks.

        Every timestep is covered exactly once and every chunk respects
        ``min_chunk_len`` (PLAN.md section 22 segmenter tests).
        """
        n_steps = traj.length
        if n_steps == 0:
            return []
        labels = self._labels_for(traj)
        boundaries = list(_boundaries_from_labels(labels))
        boundaries = merge_short_chunks(boundaries, n_steps, self.min_chunk_len)

        edges = [0, *boundaries, n_steps]
        tf_norm = self.normalizer.normalize(self.spec.task_features(traj.states))
        chunks: list[Chunk] = []
        for a, b in zip(edges[:-1], edges[1:]):
            pre = tf_norm[a]
            post = tf_norm[b]
            chunks.append(
                Chunk(
                    chunk_id=self._next_chunk_id,
                    policy_id=traj.policy_id,
                    episode_id=traj.episode_id,
                    start_t=a,
                    end_t=b,
                    states=traj.states[a : b + 1],
                    actions=traj.actions[a:b],
                    rewards=traj.rewards[a:b],
                    pre_state=pre,
                    post_state=post,
                    effect=post - pre,
                    success_episode=traj.success,
                )
            )
            self._next_chunk_id += 1
        return chunks
