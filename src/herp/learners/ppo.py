"""Adapter around the existing PPO optimizer, with no objective changes."""
import torch
from herp.gradient_signature import policy_gradient_signature, signature_parameters

class PPOLearner:
    def __init__(self, agent, optimizer, args, update_fn):
        self.agent, self.optimizer, self.args, self.update_fn = agent, optimizer, args, update_fn
        self.batches = []

    def act(self, obs, deterministic=False):
        return self.agent.act(obs, deterministic=deterministic)

    def observe(self, transition):
        self.batches.append(transition)

    def update(self):
        if not self.batches:
            raise ValueError('No rollout batches observed')
        batch = {k: torch.cat([b[k] for b in self.batches]) for k in self.batches[0]}
        result = self.update_fn(self.agent, self.optimizer, batch, self.args)
        self.batches.clear()
        return result

    def policy_parameters(self):
        return signature_parameters(self.agent)

    def gradient_signature(self, batch):
        return policy_gradient_signature(self.agent, batch['obs'], batch['actions'],
                                         batch['advantages'], batch.get('adv_scale'))

    def save(self, path):
        torch.save(dict(policy=self.agent.state_dict(), optimizer=self.optimizer.state_dict(),
                        batches=self.batches), path)

    def load(self, path):
        state = torch.load(path, weights_only=False, map_location=next(self.agent.parameters()).device)
        self.agent.load_state_dict(state['policy'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.batches = state['batches']
