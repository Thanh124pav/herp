"""Segmenter unit tests (PLAN.md section 22)."""

import numpy as np

from experience_routing.experience.segmenter import Segmenter, merge_short_chunks
from tests.conftest import make_phase_trajectory


def _fit_segmenter(spec, min_chunk_len=4):
    trajs = [make_phase_trajectory(seed=s) for s in range(8)]
    seg = Segmenter(spec, k_seg=3, median_window=5, min_chunk_len=min_chunk_len, seed=0)
    seg.fit(trajs)
    return seg


def test_boundaries_sorted_and_full_cover(spec):
    seg = _fit_segmenter(spec)
    traj = make_phase_trajectory(seed=100)
    chunks = seg.segment(traj)
    # chunks tile [0, T] contiguously with no gaps/overlaps
    assert chunks[0].start_t == 0
    assert chunks[-1].end_t == traj.length
    for a, b in zip(chunks[:-1], chunks[1:]):
        assert a.end_t == b.start_t  # contiguous, sorted, no overlap
    covered = sum(c.length for c in chunks)
    assert covered == traj.length  # every timestep covered exactly once


def test_min_chunk_len_respected(spec):
    seg = _fit_segmenter(spec, min_chunk_len=4)
    traj = make_phase_trajectory(seed=101)
    chunks = seg.segment(traj)
    # every chunk except possibly a whole-short-trajectory respects the minimum
    if traj.length >= 4:
        assert all(c.length >= 4 for c in chunks)


def test_not_degenerate(spec):
    seg = _fit_segmenter(spec)
    traj = make_phase_trajectory(seed=102)
    chunks = seg.segment(traj)
    # neither one-chunk-per-step nor a single chunk for the whole trajectory
    assert 1 < len(chunks) < traj.length


def test_merge_short_chunks_helper():
    # boundaries at 2 and 5 on a length-10 trajectory, min length 4
    interior = merge_short_chunks([2, 5], n_steps=10, min_chunk_len=4)
    edges = [0, *interior, 10]
    lengths = [b - a for a, b in zip(edges[:-1], edges[1:])]
    assert all(length >= 4 for length in lengths)
    assert edges[0] == 0 and edges[-1] == 10
