import torch
from torch import nn


def mlp(n_in, n_out, hidden=128):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                         nn.ReLU(), nn.Linear(hidden, n_out))


class RND(nn.Module):
    """Random network distillation for privileged state observations."""
    def __init__(self, obs_dim, action_dim=0):
        super().__init__()
        self.target = mlp(obs_dim, 64).requires_grad_(False)
        self.predictor = mlp(obs_dim, 64)

    def errors(self, obs, actions, next_obs):
        return (self.predictor(next_obs) - self.target(next_obs).detach()).square().mean(-1)

    def bonus(self, obs, actions, next_obs):
        return self.errors(obs, actions, next_obs)

    def loss(self, obs, actions, next_obs):
        return self.errors(obs, actions, next_obs).mean()
