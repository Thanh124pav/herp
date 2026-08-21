"""A single population member: one SAC policy + its env + its buffers.

(PLAN.md Module A, sections 5, 14.) Each worker collects data independently into
its own local buffer and trains from a local/routed mix. Workers are the unit the
rollout manager and routers address.
"""

from __future__ import annotations

import numpy as np

from ..envs.base import BaseEnv
from ..experience.trajectory import Trajectory
from ..rl.replay_buffer import ReplayBuffer, mixed_batch
from ..rl.sac import SACAgent, SACConfig


class Worker:
    def __init__(
        self,
        policy_id: int,
        env: BaseEnv,
        buffer_capacity: int = 100_000,
        route_batch_fraction: float = 0.25,
        seed: int = 0,
        hidden: int = 128,
        utd: int = 1,
        device: str = "cpu",
        obs_norm: bool = False,
        lr: float = 3e-4,
    ):
        self.policy_id = policy_id
        self.env = env
        self.route_batch_fraction = float(route_batch_fraction)
        self.utd = int(utd)  # gradient updates per env step (PLAN.md backbone strength)
        spec = env.spec
        self.agent = SACAgent(
            SACConfig(obs_dim=spec.obs_dim, action_dim=spec.action_dim, seed=seed,
                      hidden=hidden, device=device, obs_norm=obs_norm, lr=lr)
        )
        self.local_buffer = ReplayBuffer(buffer_capacity, spec.obs_dim, spec.action_dim, seed)
        self.routed_buffer = ReplayBuffer(buffer_capacity, spec.obs_dim, spec.action_dim, seed + 1)

        self._rng = np.random.default_rng(seed)
        self._episode_id = 0
        self.env_steps = 0  # per-policy interaction count (PLAN.md section 20)
        self.last_losses: dict[str, float] = {}  # latest SAC diagnostics (PLAN.md section 21)
        # rolling episode buffers
        self._reset_episode(seed)

    @property
    def current_obs(self) -> np.ndarray:
        """Observation the worker is about to act on (for external selectors)."""
        return self._obs

    def _reset_episode(self, seed: int | None = None) -> None:
        self._obs = self.env.reset(seed=seed)
        self.agent.observe_obs(self._obs)
        self._states = [self._obs.copy()]
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._dones: list[float] = []
        self._ep_success = False

    def collect_step(self, warmup: bool = False, action=None) -> Trajectory | None:
        """Take one env step. Returns a finished ``Trajectory`` when the episode
        ends, else ``None``.

        ``action`` may be supplied by the rollout manager (e.g. QMP-style
        receiver-Q behavior selection, PLAN.md B5); otherwise the worker's own
        policy acts (or uniform noise during warmup).
        """
        spec = self.env.spec
        if warmup:
            action = self._rng.uniform(-1, 1, size=spec.action_dim)
        elif action is None:
            action = self.agent.act(self._obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated)
        self.local_buffer.add(self._obs, action, reward, next_obs, done)

        self._states.append(next_obs.copy())
        self._actions.append(np.asarray(action, dtype=np.float64))
        self._rewards.append(float(reward))
        self._dones.append(float(done))
        self._ep_success = self._ep_success or bool(info.get("success", False))
        self._obs = next_obs
        self.agent.observe_obs(next_obs)
        self.env_steps += 1

        if terminated or truncated:
            traj = Trajectory(
                policy_id=self.policy_id,
                episode_id=self._episode_id,
                states=np.asarray(self._states),
                actions=np.asarray(self._actions),
                rewards=np.asarray(self._rewards),
                dones=np.asarray(self._dones),
                success=self._ep_success,
                env_context={"perturbation": getattr(self.env, "pert", None)},
            )
            self._episode_id += 1
            self._reset_episode()
            return traj
        return None

    def train_step(self, batch_size: int = 128) -> dict[str, float] | None:
        losses = None
        for _ in range(self.utd):  # UTD: multiple gradient steps per env step
            batch = mixed_batch(
                self.local_buffer, self.routed_buffer, batch_size, self.route_batch_fraction
            )
            if batch is None:
                return losses
            losses = self.agent.update(batch)
        if losses is not None:
            self.last_losses = losses
        return losses

    def add_routed_transitions(self, obs, actions, rewards, next_obs, dones) -> None:
        self.routed_buffer.add_many(obs, actions, rewards, next_obs, dones)

    def reset_routed_buffer(self) -> None:
        """Empty the routed buffer (controlled-transfer isolation, PLAN.md 19.3)."""
        spec = self.env.spec
        self.routed_buffer = ReplayBuffer(
            self.routed_buffer.capacity, spec.obs_dim, spec.action_dim, self.policy_id + 1
        )

    def snapshot_policy(self) -> dict:
        """Deep copy of this worker's SAC state (checkpoint, PLAN.md 19.3)."""
        return self.agent.snapshot()

    def restore_policy(self, snap: dict) -> None:
        self.agent.load_state_dict(snap)

    def evaluate(self, episodes: int = 20, base_seed: int = 1_000_000) -> dict[str, float]:
        succ, returns = 0, []
        for ep in range(episodes):
            obs = self.env.reset(seed=base_seed + ep)
            done = trunc = False
            ret = 0.0
            info: dict = {}
            while not (done or trunc):
                action = self.agent.act(obs, deterministic=True)
                obs, r, done, trunc, info = self.env.step(action)
                ret += r
            succ += int(info.get("success", False))
            returns.append(ret)
        return {"success_rate": succ / episodes, "mean_return": float(np.mean(returns))}
