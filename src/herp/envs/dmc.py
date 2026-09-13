"""DeepMind Control Suite adapter — required by THEORY §8 / EXPERIMENTS §8.

Uses the DMC Python bindings directly (``dm_control.suite``) rather than
going through gymnasium, because DMC's snapshot semantics are exposed as
raw physics state (``env.physics.get_state()`` / ``set_state()``) and we
would lose that clean round-trip through a compatibility layer.

Env id format: ``"<domain>-<task>"`` — e.g. ``"walker-run"``,
``"humanoid-walk"``, ``"quadruped-run"``, ``"acrobot-swingup"``,
``"finger-turn_hard"``, ``"hopper-hop"``.

Vectorization: serial list-of-envs (same pattern as MetaWorld / Fetch);
DMC episodes are cheap enough that the paper does not need per-process
parallelism. HERP allocation is still identical to the multi-env
adapters — snapshot save/restore is per-slot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .base import EnvAdapter
from ._serial import SerialVectorState, close_all, collect_success, serial_reset, serial_step


@dataclass
class DMCSnapshot:
    """Full sim state — DMC's ``physics.get_state()`` bundles qpos+qvel+act."""
    physics_state: np.ndarray
    time: float
    elapsed_steps: int = 0


def _obs_to_flat(obs) -> np.ndarray:
    """DMC returns an OrderedDict of named views; flatten to a single vector.

    Component order is preserved by the underlying ``obs.values()`` iteration,
    so the flattened layout is deterministic per task.
    """
    if isinstance(obs, dict):
        parts = [np.asarray(v, dtype=np.float32).flatten() for v in obs.values()]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return np.asarray(obs, dtype=np.float32).flatten()


def _parse_env_id(env_id: str) -> tuple[str, str]:
    """Accepts "domain-task" or "domain/task"; returns (domain, task)."""
    for sep in ("-", "/"):
        if sep in env_id:
            domain, _, task = env_id.partition(sep)
            if domain and task:
                return domain, task
    raise ValueError(f"DMC env_id must be 'domain-task' (got {env_id!r})")


def _build_env(env_id: str, seed: int):
    """Load a single DMC task. Import at call time so `import herp.envs.dmc`
    doesn't require dm_control on installs that only need ManiSkill."""
    from dm_control import suite
    domain, task = _parse_env_id(env_id)
    return suite.load(domain_name=domain, task_name=task, task_kwargs={"random": seed})


class _DMCEnvShim:
    """Thin gymnasium-flavored wrapper for one DMC env so the shared
    ``_serial.py`` helpers (``serial_reset``/``serial_step``) work unchanged."""

    def __init__(self, dmc_env):
        self._env = dmc_env
        # Cache action spec + inferred observation size.
        spec = dmc_env.action_spec()
        self._action_low = np.asarray(spec.minimum, dtype=np.float32)
        self._action_high = np.asarray(spec.maximum, dtype=np.float32)
        self._action_dim = int(np.prod(spec.shape))
        ts = dmc_env.reset()
        self._obs_dim = int(_obs_to_flat(ts.observation).size)
        self._last_obs = ts.observation
        # Store a lightweight spec so ``serial_step`` can bump elapsed steps.
        self.action_space = _BoxSpace(self._action_low, self._action_high)
        self.observation_space = _BoxSpace(-np.inf * np.ones(self._obs_dim),
                                           np.inf * np.ones(self._obs_dim))
        # DMC control_timestep varies per suite; expose the natural episode limit.
        control_ts = float(dmc_env.control_timestep())
        # Default DMC episode = 1000 physics steps @ control_timestep ~ 0.025s = 25 s.
        self.spec = _SpecShim(max_episode_steps=1000)

    def reset(self, *, seed: int | None = None, **_):
        ts = self._env.reset()
        self._last_obs = ts.observation
        return ts.observation, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        ts = self._env.step(action)
        self._last_obs = ts.observation
        reward = float(ts.reward if ts.reward is not None else 0.0)
        terminated = bool(ts.last() and ts.discount == 0.0)  # true task termination
        truncated = bool(ts.last() and not terminated)       # time-limit
        info: dict = {}
        return ts.observation, reward, terminated, truncated, info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    @property
    def physics(self):
        return self._env.physics


class _BoxSpace:
    """Minimal gymnasium.spaces.Box replica so shared helpers stay happy."""
    def __init__(self, low, high):
        self.low = np.asarray(low)
        self.high = np.asarray(high)
        self.shape = self.low.shape


class _SpecShim:
    __slots__ = ("max_episode_steps",)

    def __init__(self, max_episode_steps: int):
        self.max_episode_steps = max_episode_steps


class DMCAdapter(EnvAdapter):
    benchmark = "dmc"

    def __init__(self, env_id: str, device: str | torch.device = "cpu", **_):
        self.env_id = env_id
        self.device = torch.device(device)
        self.env: _DMCEnvShim | None = None
        self._state: SerialVectorState | None = None
        self.max_episode_steps = 1000

    # ------------------------------------------------------------------ life-cycle

    def make(self, num_envs: int = 1, seed: int = 0, **_) -> "DMCAdapter":
        if self._state is not None:
            return self
        envs = [_DMCEnvShim(_build_env(self.env_id, seed + i)) for i in range(num_envs)]
        # Prime spaces from the first env.
        self.num_envs = num_envs
        self._state = SerialVectorState(envs, self.device)
        self.env = envs[0]
        self.obs_dim = envs[0]._obs_dim
        self.action_dim = envs[0]._action_dim
        self.max_episode_steps = int(getattr(envs[0].spec, "max_episode_steps", 1000) or 1000)
        return self

    def close(self) -> None:
        if self._state is not None:
            close_all(self._state)
            self._state = None
            self.env = None

    # ------------------------------------------------------------------ stepping

    def reset(self, seed: int | None = None):
        obs_np, info = serial_reset(self._state, _obs_to_flat)
        return torch.as_tensor(obs_np, device=self.device).float(), info

    def step(self, actions: torch.Tensor):
        obs_np, rew, term, trunc, info = serial_step(
            self._state, actions, _obs_to_flat, self.max_episode_steps
        )
        return (
            torch.as_tensor(obs_np, device=self.device).float(),
            torch.as_tensor(rew, device=self.device).float(),
            torch.as_tensor(term, device=self.device).bool(),
            torch.as_tensor(trunc, device=self.device).bool(),
            info,
        )

    # ------------------------------------------------------------------ snapshots

    def _snap_one(self, i: int) -> DMCSnapshot:
        env = self._state.envs[i]
        physics = env.physics
        return DMCSnapshot(
            physics_state=np.array(physics.get_state(), dtype=np.float64, copy=True),
            time=float(physics.time()),
            elapsed_steps=int(self._state.elapsed[i]),
        )

    def save_state(self, env_ids: torch.Tensor) -> list[DMCSnapshot]:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu").tolist()
        return [self._snap_one(i) for i in env_ids]

    def _restore_one(self, i: int, snap: DMCSnapshot) -> np.ndarray:
        env = self._state.envs[i]
        physics = env.physics
        with physics.reset_context():
            physics.set_state(snap.physics_state)
            physics.data.time = snap.time
        # Refresh derived quantities (contact, sensordata) before reading obs.
        physics.forward()
        obs = env._env.task.get_observation(physics)
        env._last_obs = obs
        self._state.last_obs[i] = obs
        self._state.elapsed[i] = int(snap.elapsed_steps)
        return _obs_to_flat(obs)

    def restore_state(
        self, env_ids: torch.Tensor, snapshots: list[DMCSnapshot]
    ) -> torch.Tensor:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu").tolist()
        obs = [self._restore_one(i, s) for i, s in zip(env_ids, snapshots)]
        return torch.as_tensor(np.stack(obs, axis=0), device=self.device).float()

    # ------------------------------------------------------------------ features

    def obs_tensor(self, raw_obs) -> torch.Tensor:
        if isinstance(raw_obs, dict):
            return torch.as_tensor(_obs_to_flat(raw_obs), device=self.device).float()
        if torch.is_tensor(raw_obs):
            return raw_obs.to(self.device).float()
        return torch.as_tensor(np.asarray(raw_obs, dtype=np.float32), device=self.device).float()

    def elapsed_steps(self) -> torch.Tensor:
        return self._state.elapsed.detach().clone()

    def success_from_info(self, info) -> torch.Tensor:
        """DMC has no native success bit; return an all-zero mask.

        Downstream reporting should not surface ``success_once``/
        ``success_at_end`` for DMC tasks — they aren't defined. The
        merge_cross_backbone_results.py table renders these as N/A.
        """
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
