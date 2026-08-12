"""Shared helper: turn abstract Route mass into real chunks in receiver buffers.

Used by the experience-level routers (greedy, UOT). The router decides
donor->receiver->experience mass; this converts a per-route chunk count into
actual exemplar chunks from the bank and pushes their transitions into the
receiver's routed buffer (PLAN.md sections 4.5, 11, 14).
"""

from __future__ import annotations

from ..experience.trajectory import Route
from .base import RoutingContext, add_chunks_to_receiver


def materialize(population, ctx: RoutingContext, routes: list[Route]) -> list[Route]:
    """Fill ``route.chunk_ids`` from the bank and apply to receiver buffers.

    Returns the routes that actually moved at least one chunk.
    """
    workers = {w.policy_id: w for w in population}
    applied: list[Route] = []
    for route in routes:
        n_chunks = int(round(route.mass))
        if n_chunks <= 0:
            continue
        chunks = ctx.bank.select(route.donor_id, route.experience_id, ctx.vocabulary, n_chunks)
        if not chunks:
            continue
        receiver = workers.get(route.receiver_id)
        if receiver is None:
            continue
        add_chunks_to_receiver(receiver, chunks)
        route.chunk_ids = [c.chunk_id for c in chunks]
        applied.append(route)
    return applied
