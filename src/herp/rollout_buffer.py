from __future__ import annotations

from dataclasses import dataclass

import torch


NORMAL = 0
PROBE = 1
ALLOCATED = 2


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_value: torch.Tensor
    source: torch.Tensor
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    def flatten(self) -> "RolloutBatch":
        return RolloutBatch(
            obs=self.obs.reshape(-1, *self.obs.shape[2:]),
            actions=self.actions.reshape(-1, *self.actions.shape[2:]),
            logprobs=self.logprobs.reshape(-1),
            rewards=self.rewards.reshape(-1),
            dones=self.dones.reshape(-1),
            values=self.values.reshape(-1),
            next_value=self.next_value.reshape(-1),
            source=self.source.reshape(-1),
            advantages=None if self.advantages is None else self.advantages.reshape(-1),
            returns=None if self.returns is None else self.returns.reshape(-1),
        )


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    next_done: torch.Tensor | None = None,
    final_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE with upstream-style boundary + truncation-bootstrap handling.

    Two things this handles that a naive GAE does not:

    1. **Off-by-one on ``dones``.** ``dones[t]`` records whether **step t-1**
       terminated the episode (the buffer stores ``next_done`` at the start
       of iteration t). To ask "did step t end?" we read ``dones[t+1]``, or
       the caller-supplied ``next_done`` at the last step.
    2. **Time-limit truncation bootstrap.** In ManiSkill / most robotics envs
       episodes usually end by truncation, not a true terminal. In that case
       ``V(s_T)`` is a valid target for bootstrapping the return of step T-1.
       ``final_values[t]`` should be ``V(final_observation)`` for slots that
       ended at step t (zero elsewhere). Without this, the value function
       learns ``V(s_near_end) ≈ 0`` and the whole PPO update goes off — this
       was the bug behind the "return climbs then collapses" pattern in the
       HERP pilot suite.

    Matches the pattern in the upstream ManiSkill/CleanRL PPO reference.
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(next_value)
    if next_done is None:
        next_done = torch.zeros_like(next_value)
    if final_values is None:
        final_values = torch.zeros_like(rewards)
    for t in reversed(range(T)):
        if t == T - 1:
            next_not_done = 1.0 - next_done
            nextvalues = next_value
        else:
            next_not_done = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        # When step t ended the episode (next_not_done == 0), bootstrap with
        # V(final_obs) — the value of the actual last state visited — rather
        # than V(reset_obs). ``final_values[t]`` is 0 for slots that did NOT
        # end at step t, so this collapses to the standard formula there.
        real_next_values = next_not_done * nextvalues + final_values[t]
        delta = rewards[t] + gamma * real_next_values - values[t]
        lastgaelam = delta + gamma * gae_lambda * next_not_done * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns
