"""B4 -- SUPER-style global TD-priority sharing, adapted to SAC (PLAN.md 17, 2.3).

Select the top-|TD-error| transitions globally across all donors and share them
with the other policies. This is the "global experience importance" contrast to
receiver-specific deficit routing. Implemented inside our own SAC backbone so
architecture, replay budget, and population size are matched (PLAN.md section 2.3).
"""

from __future__ import annotations

import numpy as np
import torch

from ..experience.trajectory import Route
from .base import Router, RoutingContext


class TDPriorityRouter(Router):
    name = "td_priority"

    def __init__(self, candidate_pool: int = 512):
        self.candidate_pool = int(candidate_pool)

    def _td_errors(self, worker, batch) -> np.ndarray:
        agent = worker.agent
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        act = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rew = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        nobs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        done = torch.as_tensor(batch["dones"], dtype=torch.float32)
        with torch.no_grad():
            na, nlp = agent.actor.sample(nobs)
            tq1, tq2 = agent.critic_target(nobs, na)
            target = rew + agent.cfg.gamma * (1 - done) * (torch.min(tq1, tq2) - agent.alpha * nlp)
            q1, q2 = agent.critic(obs, act)
            td = torch.abs(target - torch.min(q1, q2)).squeeze(-1)
        return td.cpu().numpy()

    def route(self, population, ctx: RoutingContext) -> list[Route]:
        workers = list(population)
        n = len(workers)
        if n <= 1:
            return []
        # gather a global candidate pool tagged by donor
        pooled = []  # (td, donor_idx, transition dict-of-arrays index)
        donor_batches = {}
        for wi, w in enumerate(workers):
            if len(w.local_buffer) == 0:
                continue
            take = min(self.candidate_pool, len(w.local_buffer))
            batch = w.local_buffer.sample(take)
            td = self._td_errors(w, batch)
            donor_batches[wi] = batch
            for j in range(take):
                pooled.append((td[j], wi, j))
        if not pooled:
            return []
        pooled.sort(key=lambda x: -x[0])
        top = pooled[: ctx.budget_transitions]

        routes: list[Route] = []
        for td_val, wi, j in top:
            batch = donor_batches[wi]
            transition = (
                [batch["obs"][j]], [batch["actions"][j]], [batch["rewards"][j]],
                [batch["next_obs"][j]], [batch["dones"][j]],
            )
            donor = workers[wi]
            for receiver in workers:
                if receiver.policy_id == donor.policy_id:
                    continue
                receiver.add_routed_transitions(*transition)
                routes.append(Route(donor.policy_id, receiver.policy_id, -1, [], 1.0))
        return routes
