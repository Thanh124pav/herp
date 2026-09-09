import torch
from torch import nn
from .rnd import mlp


class Disagreement(nn.Module):
    """Five independently initialized, bootstrap-trained forward models."""
    def __init__(self, obs_dim, action_dim, ensemble_size=5):
        super().__init__()
        self.models = nn.ModuleList([mlp(obs_dim + action_dim, obs_dim) for _ in range(ensemble_size)])

    def predictions(self, obs, actions):
        return torch.stack([model(torch.cat((obs, actions), -1)) for model in self.models])

    def bonus(self, obs, actions, next_obs):
        return self.predictions(obs, actions).var(0, unbiased=False).mean(-1)

    def loss(self, obs, actions, next_obs):
        errors = (self.predictions(obs, actions) - next_obs[None]).square().mean(-1)
        masks = (torch.rand_like(errors) < .8).float()
        return (errors * masks).sum() / masks.sum().clamp_min(1)
