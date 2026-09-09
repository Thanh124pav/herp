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
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(next_value)
    for t in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[t]
        next_values = next_value if t == rewards.shape[0] - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns
