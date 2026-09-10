"""HERP training entrypoint — vectorized GPU sim + official PPO backbone.

This is the §4.5 rewrite of the single-env prototype: it runs one big
``ManiSkillVectorEnv`` (``num_envs≥1024``, ``physx_cuda``) and treats each env
slot as one of four modes at every step — NORMAL, PROBE, ALLOCATED, REFERENCE
(§4.5). Slot-mode counters roll into a strict interaction-budget assertion
(§4.6). The inner PPO loop matches the upstream ManiSkill baseline
(3-layer 256-wide MLP, orthogonal init, clip-vloss, ``target_kl=0.1``,
optional linear LR anneal) so the pilot-v3 curves are comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.distributions.normal import Normal

from herp.allocator import AllocationConfig, priority_distribution
from herp.archive import RegionArchive
from herp.envs import make_adapter
from herp.gradient_signature import (
    empirical_fisher_diagonal,
    policy_gradient_signature,
    signature_parameters,
)
from herp.logging import CsvLogger
from herp.probe import probe_rollout, sample_probe_assignment, sigma_from_features
from herp.reference import collect_reference
from herp.regions import OnlineRegionizer, RegionizerConfig, RunningNormalizer
from herp.relevance import cosine_relevance
from herp.rollout_buffer import ALLOCATED, NORMAL, compute_gae
from herp.baselines import RND, Disagreement


# Slot-mode identifiers -----------------------------------------------------
MODE_NORMAL = 0
MODE_PROBE = 1
MODE_ALLOCATED = 2
MODE_REFERENCE = 3


@dataclass
class Args:
    # --- benchmark / task ---
    benchmark: str = "maniskill"
    env_id: str = "PickCube-v1"
    method: str = "herp"
    seed: int = 0
    # --- budget ---
    total_timesteps: int = 5_000_000
    num_envs: int = 1024
    num_envs_ref: int = 64
    num_steps: int = 32
    reference_horizon: int = 16
    reference_interval: int = 4
    # --- ppo ---
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 8
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    reward_scale: float = 1.0
    norm_adv: bool = True
    hidden: int = 256
    # --- herp ---
    probe_frac: float = 0.15
    allocated_frac: float = 0.10
    probe_horizon: int = 8
    probes_per_region: int = 4
    max_candidates: int = 32
    max_regions: int = 128
    max_snapshots_per_region: int = 8
    region_radius: float = 0.5
    archive_interval: int = 4
    scoring_interval: int = 2
    sigma_ema_tau: float = 0.9  # weight on OLD estimate
    p_ema_tau: float = 0.9
    uniform_mix: float = 0.10
    staleness_mix: float = 0.05
    sigma_estimator: str = "pairwise"
    p_estimator: str = "cosine"
    lambda_dyn: float = 0.0
    hybrid_eta: float = 0.5
    eps_hybrid: float = 1e-6
    fisher_damping: float = 1e-3
    probe_scale: float = 1.0
    # --- baselines ---
    intrinsic_coef: float = 0.01
    intrinsic_epochs: int = 4
    # --- eval ---
    eval_interval: int = 250_000
    eval_episodes: int = 100
    num_eval_envs: int = 8
    # --- sim / device ---
    control_mode: str = "pd_joint_delta_pos"
    obs_mode: str = "state"
    reward_mode: str = "normalized_dense"
    sim_backend: str = "physx_cuda"
    render_backend: str = "gpu"
    device: str = "cuda"
    torch_deterministic: bool = True
    # --- io ---
    output_dir: str = "outputs/herp"
    save_model: bool = True
    resume_from: str = ""
    wandb_mode: str = "disabled"
    wandb_project: str = "herp"
    wandb_entity: str = ""
    wandb_group: str = ""
    wandb_run_name: str = ""
    wandb_tags: str = ""
    wandb_log_every: int = 1


VALID_METHODS = ("ppo", "herp", "herp_sigma", "herp_p", "rnd", "disagreement", "go_explore", "plr")
VALID_P_ESTIMATORS = ("cosine", "dot", "fisher", "occupancy", "hybrid")
VALID_SIGMA_ESTIMATORS = ("pairwise", "branch", "return")


def _default_yaml_type(value):
    if isinstance(value, bool):
        return lambda s: str(s).lower() in ("1", "true", "yes", "y")
    return type(value)


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", action="append", default=[])
    known, _ = pre.parse_known_args(argv)
    defaults = asdict(Args())
    for path in known.config:
        cfg = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(cfg) - set(defaults)
        if unknown:
            pre.error(f"Unknown configuration keys: {sorted(unknown)}")
        defaults.update(cfg)
    parser = argparse.ArgumentParser(parents=[pre])
    for name, value in asdict(Args()).items():
        parser.add_argument(
            "--" + name.replace("_", "-"), type=_default_yaml_type(value), default=defaults[name]
        )
    parsed = vars(parser.parse_args(argv))
    parsed.pop("config")
    args = Args(**parsed)
    if args.method not in VALID_METHODS:
        parser.error(f"Unknown method {args.method!r}; expected one of {VALID_METHODS}")
    if args.p_estimator not in VALID_P_ESTIMATORS:
        parser.error(f"Unknown p estimator {args.p_estimator!r}")
    if args.sigma_estimator not in VALID_SIGMA_ESTIMATORS:
        parser.error(f"Unknown sigma estimator {args.sigma_estimator!r}")
    if args.wandb_mode not in ("disabled", "offline", "online"):
        parser.error("wandb_mode must be one of: disabled, offline, online")
    if args.wandb_log_every < 1:
        parser.error("wandb_log_every must be >= 1")
    if not 0 <= args.allocated_frac + args.probe_frac < 1:
        parser.error("probe_frac + allocated_frac must be in [0, 1)")
    return args


# ---------------------------------------------------------------------------
# Agent — copied from the ManiSkill upstream ppo.py (§4.14). Same 3xMLP-256
# tanh, orthogonal init, actor head std = 0.01*sqrt(2), initial logstd = -0.5.
# ---------------------------------------------------------------------------


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, 1)),
        )
        self.actor = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.actor_head = _layer_init(nn.Linear(hidden, action_dim), std=0.01 * np.sqrt(2))
        self.logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_distribution(self, x):
        hidden = self.actor(x.float())
        mean = self.actor_head(hidden)
        std = self.logstd.expand_as(mean).exp()
        return Normal(mean, std)

    def get_value(self, x):
        return self.critic(x.float()).squeeze(-1)

    value = get_value

    def act(self, x, deterministic: bool = False):
        dist = self.get_distribution(x)
        return dist.mean if deterministic else dist.sample()

    def get_action_and_value(self, x, action=None, deterministic: bool = False):
        dist = self.get_distribution(x)
        if action is None:
            action = dist.mean if deterministic else dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, self.get_value(x)


# ---------------------------------------------------------------------------
# Vectorized PPO update
# ---------------------------------------------------------------------------


def ppo_update(agent, optimizer, rollout, args):
    """Upstream-shaped PPO update; ``rollout`` is a flat dict of tensors on device."""
    b_obs = rollout["obs"]
    b_actions = rollout["actions"]
    b_logprobs = rollout["logprobs"]
    b_advantages = rollout["advantages"]
    b_returns = rollout["returns"]
    b_values = rollout["values"]
    batch_size = b_obs.shape[0]
    minibatch = max(1, batch_size // args.num_minibatches)

    # Use numpy for minibatch shuffling — matches the upstream ManiSkill PPO
    # recipe verified via mini_train_direct2.py. Prior attempt using GPU
    # torch.randperm gave a training curve that consistently diverged from
    # the upstream baseline (return climbs then collapses).
    b_inds = np.arange(batch_size)
    metrics = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "approx_kl": 0.0, "clipfrac": 0.0, "grad_norm": 0.0,
    }
    stop = False
    steps = 0
    epochs_completed = 0
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch):
            mb = b_inds[start : start + minibatch]
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
            logratio = newlogprob - b_logprobs[mb]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
            # Match upstream: bail BEFORE the update when the current policy is
            # already too far from the sampling policy on this minibatch. The
            # pre-refactor code did the update first and then broke, which
            # allowed one final blow-out update per iteration.
            if args.target_kl and approx_kl > args.target_kl:
                stop = True
                break
            mb_adv = b_advantages[mb]
            if args.norm_adv and mb_adv.numel() > 1:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            pg1 = -mb_adv * ratio
            pg2 = -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg1, pg2).mean()
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_unc = (newvalue - b_returns[mb]).pow(2)
                v_clip = b_values[mb] + (newvalue - b_values[mb]).clamp(-args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max(v_unc, (v_clip - b_returns[mb]).pow(2)).mean()
            else:
                v_loss = 0.5 * (newvalue - b_returns[mb]).pow(2).mean()
            loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            metrics["policy_loss"] += float(pg_loss.detach())
            metrics["value_loss"] += float(v_loss.detach())
            metrics["entropy"] += float(entropy.mean().detach())
            metrics["approx_kl"] += float(approx_kl.detach())
            metrics["clipfrac"] += float(clipfrac.detach())
            metrics["grad_norm"] += float(grad_norm.detach())
            steps += 1
        epochs_completed = epoch + 1
        if stop:
            break
    for k in metrics:
        metrics[k] /= max(1, steps)
    metrics.update(
        ppo_epochs=epochs_completed,
        ppo_minibatches=steps,
        ppo_early_stop=float(stop),
    )
    return metrics


# ---------------------------------------------------------------------------
# Vectorized eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(adapter, agent, args, device, episodes: int) -> dict:
    """Match the upstream ManiSkill PPO eval:

    - reset the vec env WITHOUT a fixed seed so tasks are freshly randomized
      (mixing this with a big vec fanout gives an unbiased success estimate);
    - roll ``adapter.max_episode_steps`` steps windows until we have at least
      ``episodes`` completed episodes across all slots, tracking ``success_once``;
    - success is per-episode ``max`` of the per-step ``info["success"]`` flag.

    The previous implementation seeded reset to a single fixed task setup and
    then bailed as soon as each slot finished ONE episode — so it evaluated a
    trained policy against exactly ``num_eval_envs`` runs of ONE particular
    task, which made success stay 0 even when training was working.
    """
    obs, _ = adapter.reset()
    ep_returns: list[float] = []
    ep_successes: list[float] = []
    per_slot_return = torch.zeros(adapter.num_envs, device=device)
    per_slot_success = torch.zeros(adapter.num_envs, device=device)
    steps = 0
    episode_windows = max(1, (episodes + adapter.num_envs - 1) // adapter.num_envs)
    max_steps = max(50, adapter.max_episode_steps) * int(episode_windows)
    while len(ep_returns) < episodes and steps < max_steps:
        action = agent.act(obs.to(device), deterministic=True)
        action = action.clamp(adapter.action_low(), adapter.action_high())
        obs, reward, term, trunc, info = adapter.step(action)
        reward = reward.to(device).float()
        per_slot_return = per_slot_return + reward
        per_slot_success = torch.maximum(
            per_slot_success, adapter.success_from_info(info).float().to(device)
        )
        done = (term.to(device) | trunc.to(device))
        if done.any():
            done_ids = torch.where(done)[0].tolist()
            for i in done_ids:
                ep_returns.append(float(per_slot_return[i].item()))
                ep_successes.append(float(per_slot_success[i].item()))
                per_slot_return[i] = 0.0
                per_slot_success[i] = 0.0
        steps += 1
    # A vector step can finish more slots than requested; report exactly the
    # configured episode count so methods with the same protocol stay comparable.
    ep_returns = ep_returns[:episodes]
    ep_successes = ep_successes[:episodes]
    return dict(
        eval_return=float(np.mean(ep_returns)) if ep_returns else 0.0,
        eval_success=float(np.mean(ep_successes)) if ep_successes else 0.0,
        eval_success_final=float(np.mean(ep_successes)) if ep_successes else 0.0,
        eval_steps=steps * adapter.num_envs,
        eval_episodes=len(ep_returns),
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def _pkg_version(mod: str) -> str | None:
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def write_provenance(out_dir: Path, args: Args) -> None:
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mani_skill": _pkg_version("mani_skill"),
        "metaworld": _pkg_version("metaworld"),
        "gymnasium_robotics": _pkg_version("gymnasium_robotics"),
        "mujoco": _pkg_version("mujoco"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "seed": args.seed,
        "device": args.device,
        "sim_backend": args.sim_backend,
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "hostname": platform.node(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cli": sys.argv,
    }
    (out_dir / "provenance.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_adapter(args: Args, num_envs: int, role: str = "train"):
    kwargs = dict(env_id=args.env_id, obs_mode=args.obs_mode, reward_mode=args.reward_mode)
    if args.benchmark == "maniskill":
        kwargs.update(
            control_mode=args.control_mode,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            device=args.device,
        )
        # Eval env re-randomises every episode (upstream ManiSkill recipe);
        # training env keeps the sampled task fixed after reset. This is the
        # difference that made train.py's eval report success=0 even when
        # training was solving PushCube — the eval env kept re-running the
        # same fixed configuration.
        kwargs["reconfiguration_freq"] = 1 if role == "eval" else None
        # ManiSkill asserts reconfiguration_freq>0 requires ignore_terminations=True
        # (partial-reset envs can't be silently reconfigured). Match upstream:
        # training runs with partial_reset=True (ignore_terminations=False),
        # eval runs with partial_reset=False (ignore_terminations=True).
        kwargs["ignore_terminations"] = (role == "eval")
    else:
        kwargs.update(device=args.device if args.device == "cpu" else "cpu")
    adapter = make_adapter(args.benchmark, **kwargs)
    adapter.make(num_envs=num_envs, seed=args.seed)
    return adapter


def main():
    args = parse_args()
    # Seed EVERYTHING before touching CUDA — the outer determinism block below
    # must match the recipe verified in mini_train_direct2.py (upstream env +
    # my Agent + HERP scaffolding → success=1.0 on PushCube-v1 by 491k steps).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Throughput mode: let cuDNN autotune and use TF32 matmul on Ampere+.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir) / f"{args.env_id}_{args.method}_seed{args.seed}_{time.time_ns()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(args), indent=2))
    logger = CsvLogger(out_dir / "metrics.csv")
    region_logger = CsvLogger(out_dir / "regions.csv")

    # Env-creation order matters — creating an env consumes torch RNG (SAPIEN
    # allocates GPU tensors), so we build ALL envs first and then Agent, so
    # Agent's initial weights land at a well-defined RNG offset (matches the
    # upstream ManiSkill PPO recipe verified via mini_train_direct.py).
    adapter = _build_adapter(args, args.num_envs)
    write_provenance(out_dir, args)
    obs_dim = adapter.obs_dim
    action_dim = adapter.action_dim
    ref_adapter = None
    if args.method in ("herp", "herp_p"):
        ref_adapter = _build_adapter(args, args.num_envs_ref, role="ref")
    eval_adapter = _build_adapter(args, args.num_eval_envs, role="eval")
    agent = Agent(obs_dim, action_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    archive = RegionArchive(args.max_snapshots_per_region, args.seed)
    regionizer = OnlineRegionizer(
        obs_dim,
        archive,
        RegionizerConfig(region_radius=args.region_radius, max_regions=args.max_regions),
    )
    intrinsic = None
    if args.method == "rnd":
        intrinsic = RND(obs_dim, action_dim).to(device)
    elif args.method == "disagreement":
        intrinsic = Disagreement(obs_dim, action_dim).to(device)
    intrinsic_optimizer = None
    if intrinsic is not None:
        intrinsic_optimizer = torch.optim.Adam(
            [p for p in intrinsic.parameters() if p.requires_grad], lr=args.learning_rate
        )
    use_archive = args.method not in ("ppo", "rnd", "disagreement")
    needs_reference = args.method in ("herp", "herp_p")

    obs, _ = adapter.reset(seed=args.seed)
    obs = obs.to(device).float()
    next_done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    global_steps = 0
    cumulative = dict(normal_steps=0, probe_steps=0, allocated_steps=0, reference_steps=0)
    next_eval = args.eval_interval
    generator = torch.Generator().manual_seed(args.seed)
    adv_scale_state = RunningNormalizer(1)
    reference_gradient = None
    fisher_diag = None
    candidates: list = []
    q = torch.zeros(0)
    recent_regions: list[int] = []
    update = 0
    start_time = time.monotonic()

    checkpoint = None
    if args.resume_from:
        resume_path = Path(args.resume_from).expanduser().resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        saved_args = checkpoint.get("args", {})
        for name in ("benchmark", "env_id", "method", "hidden", "control_mode", "obs_mode", "reward_mode"):
            saved = saved_args.get(name, getattr(args, name))
            if saved != getattr(args, name):
                raise ValueError(f"Resume mismatch for {name}: checkpoint={saved!r}, current={getattr(args, name)!r}")
        if "optimizer" not in checkpoint:
            raise ValueError("Checkpoint predates resumable-state support; optimizer state is missing")
        agent.load_state_dict(checkpoint["agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if intrinsic is not None:
            if checkpoint.get("intrinsic") is None or checkpoint.get("intrinsic_optimizer") is None:
                raise ValueError("Checkpoint is missing intrinsic model/optimizer state")
            intrinsic.load_state_dict(checkpoint["intrinsic"])
            intrinsic_optimizer.load_state_dict(checkpoint["intrinsic_optimizer"])
        global_steps = int(checkpoint["global_steps"])
        cumulative = dict(checkpoint.get("cumulative", cumulative))
        update = int(checkpoint.get("update", 0))
        next_eval = int(checkpoint.get(
            "next_eval", ((global_steps // args.eval_interval) + 1) * args.eval_interval
        ))
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            try:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
            except Exception:
                pass
        if checkpoint.get("generator_state") is not None:
            generator.set_state(checkpoint["generator_state"].cpu())
        (out_dir / "resume.json").write_text(json.dumps(dict(
            resumed_from=str(resume_path), start_global_steps=global_steps,
            target_total_timesteps=args.total_timesteps,
        ), indent=2))

    wandb_run = None
    wandb_module = None
    if args.wandb_mode != "disabled":
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise RuntimeError("W&B logging requested but wandb is not installed") from exc
        resume_id = checkpoint.get("wandb_run_id") if checkpoint is not None else None
        run_name = (
            args.wandb_run_name
            or f"{args.benchmark}-{args.env_id}-{args.method}-s{args.seed}"
        )
        tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
        wandb_run = wandb_module.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group or None,
            name=run_name,
            tags=tags or None,
            mode=args.wandb_mode,
            dir=str(out_dir),
            config=asdict(args),
            id=resume_id,
            resume="allow" if resume_id else None,
        )
        (out_dir / "wandb.json").write_text(
            json.dumps(
                {
                    "id": wandb_run.id,
                    "name": wandb_run.name,
                    "project": args.wandb_project,
                    "entity": args.wandb_entity or None,
                    "group": args.wandb_group or None,
                    "mode": args.wandb_mode,
                    "url": getattr(wandb_run, "url", None),
                },
                indent=2,
            )
        )
        wandb_run.define_metric("global_env_steps")
        wandb_run.define_metric("*", step_metric="global_env_steps")

    session_start_steps = global_steps
    max_updates = max(1, args.total_timesteps // (args.num_envs * args.num_steps))

    while global_steps < args.total_timesteps:
        if args.anneal_lr:
            frac = max(0.0, 1.0 - global_steps / max(1, args.total_timesteps))
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ---------------- rollout with per-slot modes ----------------
        num_probe_slots = int(args.probe_frac * args.num_envs) if use_archive and candidates else 0
        num_alloc_slots = int(args.allocated_frac * args.num_envs) if use_archive and candidates else 0
        num_normal_slots = args.num_envs - num_probe_slots - num_alloc_slots

        obs_buf = torch.zeros(args.num_steps, args.num_envs, obs_dim, device=device)
        act_buf = torch.zeros(args.num_steps, args.num_envs, action_dim, device=device)
        logp_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
        rew_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
        val_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
        done_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
        # Per-step V(final_obs) for slots that ended at that step. Zero
        # elsewhere; see compute_gae for the truncation-bootstrap contract.
        final_values_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
        next_obs_buf = torch.zeros(args.num_steps, args.num_envs, obs_dim, device=device)
        extrinsic_reward_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
        intrinsic_bonus_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
        success_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
        episode_end_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
        mode_buf = torch.zeros(args.num_steps, args.num_envs, dtype=torch.long, device=device)

        # Probe assignment (fixed for this rollout window)
        probe_regions: list = []
        probe_snaps: list = []
        probe_slot_ids = torch.arange(num_probe_slots, device=device, dtype=torch.long)
        if num_probe_slots > 0 and candidates:
            probe_regions, probe_snaps, _ = sample_probe_assignment(
                candidates, q, num_probe_slots, args.probes_per_region, generator
            )
            snaps_state = [s.env_state for s in probe_snaps]
            new_obs = adapter.restore_state(probe_slot_ids.cpu(), snaps_state)
            obs[: num_probe_slots] = new_obs.to(device)

        # Allocated restore
        alloc_slot_ids = torch.arange(
            num_probe_slots, num_probe_slots + num_alloc_slots, device=device, dtype=torch.long
        )
        if num_alloc_slots > 0 and candidates:
            regs, snaps, _ = sample_probe_assignment(
                candidates, q, num_alloc_slots, 1, generator
            )
            snaps_state = [s.env_state for s in snaps]
            new_obs = adapter.restore_state(alloc_slot_ids.cpu(), snaps_state)
            obs[num_probe_slots : num_probe_slots + num_alloc_slots] = new_obs.to(device)

        # Rollout
        counts = dict(normal_steps=0, probe_steps=0, allocated_steps=0, reference_steps=0)
        step_modes = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        step_modes[probe_slot_ids] = MODE_PROBE
        step_modes[alloc_slot_ids] = MODE_ALLOCATED

        # Rollout — matches upstream ManiSkill ppo.py structure verbatim
        # (verified end-to-end via mini_train_direct.py → success=1.0 on
        # PushCube-v1 in 800k steps). Keep the HERP-specific accounting
        # (counts, mode_buf) but do not rearrange this block.
        action_low = adapter.action_low()
        action_high = adapter.action_high()
        for step in range(args.num_steps):
            obs_buf[step] = obs
            done_buf[step] = next_done.float()
            mode_buf[step] = step_modes
            with torch.no_grad():
                action, logprob, _ent, value = agent.get_action_and_value(obs)
                val_buf[step] = value.flatten()
            act_buf[step] = action
            logp_buf[step] = logprob
            clipped = action.detach().clamp(action_low, action_high)
            next_obs, reward, term, trunc, info = adapter.step(clipped)
            next_obs = next_obs.to(device)
            next_done = torch.logical_or(term.to(device), trunc.to(device)).float()
            transition_next_obs = next_obs.clone()
            # Truncation-bootstrap using upstream's `_final_info` mask + `final_observation` obs.
            if isinstance(info, dict) and "final_info" in info:
                done_mask = info.get("_final_info")
                if done_mask is not None and done_mask.any():
                    with torch.no_grad():
                        fo = info["final_observation"]
                        if not torch.is_tensor(fo):
                            fo = torch.as_tensor(fo, device=device)
                        fo = fo.to(device)
                        idx = torch.arange(args.num_envs, device=device)[done_mask]
                        transition_next_obs[done_mask] = fo[done_mask]
                        final_values_buf[step, idx] = agent.get_value(fo[done_mask]).view(-1)
            next_obs_buf[step] = transition_next_obs
            extrinsic_reward = reward.to(device).view(-1) * args.reward_scale
            extrinsic_reward_buf[step] = extrinsic_reward
            success_buf[step] = adapter.success_from_info(info).float().to(device)
            episode_end_buf[step] = next_done
            if intrinsic is not None:
                with torch.no_grad():
                    bonus = intrinsic.bonus(obs_buf[step], act_buf[step], transition_next_obs)
                intrinsic_bonus_buf[step] = bonus
                rew_buf[step] = extrinsic_reward + args.intrinsic_coef * bonus
            else:
                rew_buf[step] = extrinsic_reward
            counts["normal_steps"] += num_normal_slots
            counts["probe_steps"] += num_probe_slots
            counts["allocated_steps"] += num_alloc_slots
            step_modes = torch.full_like(step_modes, MODE_NORMAL)
            obs = next_obs

            # Archive new observations from NORMAL slots
            if use_archive and (step % args.archive_interval == 0):
                # Assign regions for a subsample of NORMAL slots
                sample_n = min(args.num_envs, 64)
                sample_ids = torch.randperm(args.num_envs, generator=generator)[:sample_n]
                for i in sample_ids.tolist():
                    rid = regionizer.assign(obs[i].detach().cpu(), global_steps + counts["normal_steps"])
                    recent_regions.append(rid)
                # Save snapshots for a small tail — the archive itself caps per-region snapshots.
                if len(archive) > 0:
                    ids_to_save = sample_ids[: min(16, len(sample_ids))].to("cpu")
                    snaps = adapter.save_state(ids_to_save)
                    region_ids = [regionizer.assign(obs[i].detach().cpu(),
                                                   global_steps + counts["normal_steps"],
                                                   update_stats=False)
                                  for i in ids_to_save.tolist()]
                    archive.add(ids_to_save, snaps, region_ids, obs.detach().cpu(),
                                global_steps + counts["normal_steps"])

        # ---------------- reference rollout (also counted) ----------------
        if needs_reference and (update % args.reference_interval == 0):
            adv_scale = max(float(adv_scale_state.std.item()), 1e-6)
            ref_batch = collect_reference(
                ref_adapter, agent, args.reference_horizon, device,
                adv_scale=adv_scale, seed=1_000_000 + args.seed * 10_000 + update,
            )
            counts["reference_steps"] += ref_batch.steps
            reference_gradient = policy_gradient_signature(
                agent,
                ref_batch.obs.to(device),
                ref_batch.actions.to(device),
                ref_batch.advantages.to(device),
                adv_scale=adv_scale,
            )
            if args.p_estimator == "fisher":
                fisher_diag = empirical_fisher_diagonal(
                    agent, ref_batch.obs.to(device), ref_batch.actions.to(device)
                )

        # ---------------- σ and p_v estimation from probe slots ----------
        if use_archive and probe_regions:
            # Compressed future was the obs after `num_steps` — use the value tail.
            # Here we treat each PROBE slot's post-restore trajectory (obs_buf[0..num_steps])
            # as its future rollout. Group by region for σ.
            slot_features = obs_buf.mean(dim=0)[:num_probe_slots].detach().cpu()  # [P, D]
            region_of_slot = [r.region_id for r in probe_regions]
            probe_of_slot = list(range(num_probe_slots))
            sigmas = sigma_from_features(
                slot_features, region_of_slot, probe_of_slot,
                lambda_dyn=args.lambda_dyn, estimator=args.sigma_estimator,
            )
            for rid, s in sigmas.items():
                region = archive.regions[rid]
                first = region.sigma_ema == 0.0
                region.sigma_raw = float(s)
                region.sigma_ema = (
                    region.sigma_raw
                    if first
                    else args.sigma_ema_tau * region.sigma_ema
                    + (1 - args.sigma_ema_tau) * region.sigma_raw
                )

            if (
                needs_reference
                and reference_gradient is not None
                and update % args.scoring_interval == 0
            ):
                # Compute a region gradient signature from the ALLOCATED transitions
                # of each region (that's exactly the "on-policy, region-local"
                # sample the p_v estimator wants).
                for rid in set(region_of_slot):
                    mask = (mode_buf.view(-1) == MODE_ALLOCATED) | (
                        mode_buf.view(-1) == MODE_PROBE
                    )
                    reg_obs = obs_buf.reshape(-1, obs_dim)[mask]
                    reg_act = act_buf.reshape(-1, action_dim)[mask]
                    reg_adv = (rew_buf.reshape(-1) - val_buf.reshape(-1))[mask]
                    if reg_obs.numel() == 0:
                        continue
                    adv_scale = max(float(adv_scale_state.std.item()), 1e-6)
                    grad = policy_gradient_signature(
                        agent, reg_obs, reg_act, reg_adv, adv_scale=adv_scale
                    )
                    region = archive.regions[rid]
                    region.gradient_norm = float(grad.norm())
                    if args.p_estimator == "cosine":
                        region.p_raw = cosine_relevance(grad, reference_gradient)
                    elif args.p_estimator == "dot":
                        region.p_raw = float(torch.dot(grad, reference_gradient))
                    elif args.p_estimator == "fisher" and fisher_diag is not None:
                        region.p_raw = float(
                            torch.dot(reference_gradient, grad / (fisher_diag + args.fisher_damping))
                        )
                    else:
                        region.p_raw = cosine_relevance(grad, reference_gradient)
                    positive = max(0.0, region.p_raw)
                    region.p_ema = args.p_ema_tau * region.p_ema + (1 - args.p_ema_tau) * positive

        # ---------------- intrinsic predictor update ----------------
        intrinsic_loss_value = 0.0
        intrinsic_mean_value = float(intrinsic_bonus_buf.mean()) if intrinsic is not None else 0.0
        if intrinsic is not None:
            flat_obs = obs_buf.reshape(-1, obs_dim)
            flat_actions = act_buf.reshape(-1, action_dim)
            flat_next_obs = next_obs_buf.reshape(-1, obs_dim)
            for _ in range(args.intrinsic_epochs):
                intrinsic_optimizer.zero_grad()
                intrinsic_loss = intrinsic.loss(flat_obs, flat_actions, flat_next_obs)
                intrinsic_loss.backward()
                intrinsic_optimizer.step()
                intrinsic_loss_value = float(intrinsic_loss.detach())

        # ---------------- GAE + PPO update ----------------
        with torch.no_grad():
            next_value = agent.get_value(obs).reshape(1, -1).view(-1)
        advantages, returns = compute_gae(
            rew_buf, done_buf, val_buf, next_value,
            args.gamma, args.gae_lambda,
            next_done=next_done,
            final_values=final_values_buf,
        )
        if needs_reference:
            # RunningNormalizer.update loops in Python over every advantage —
            # only needed for HERP reference/relevance; skip in pure-PPO/RND
            # paths where its output is unused.
            adv_scale_state.update(advantages.reshape(-1, 1).detach().cpu())
        rollout = dict(
            obs=obs_buf.reshape(-1, obs_dim),
            actions=act_buf.reshape(-1, action_dim),
            logprobs=logp_buf.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            values=val_buf.reshape(-1),
        )
        metrics = ppo_update(agent, optimizer, rollout, args)
        metrics.update(
            intrinsic_loss=intrinsic_loss_value,
            intrinsic_mean=intrinsic_mean_value,
            intrinsic_std=float(intrinsic_bonus_buf.std(unbiased=False)),
            intrinsic_max=float(intrinsic_bonus_buf.max()),
            extrinsic_reward_mean=float(extrinsic_reward_buf.mean()),
            extrinsic_reward_std=float(extrinsic_reward_buf.std(unbiased=False)),
            total_reward_mean=float(rew_buf.mean()),
            total_reward_std=float(rew_buf.std(unbiased=False)),
            rollout_success_rate=float(success_buf.mean()),
            rollout_episode_end_rate=float(episode_end_buf.mean()),
            advantage_mean=float(advantages.mean()),
            advantage_std=float(advantages.std(unbiased=False)),
            return_mean=float(returns.mean()),
            return_std=float(returns.std(unbiased=False)),
            value_mean=float(val_buf.mean()),
            value_std=float(val_buf.std(unbiased=False)),
            action_mean=float(act_buf.mean()),
            action_std=float(act_buf.std(unbiased=False)),
            logprob_mean=float(logp_buf.mean()),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            policy_std=float(agent.logstd.exp().mean()),
            explained_variance=float(
                1.0 - torch.var(returns - val_buf, unbiased=False)
                / torch.var(returns, unbiased=False).clamp_min(1e-8)
            ),
        )

        # ---------------- allocator + logging ----------------
        candidates = archive.candidates(recent_regions, global_steps, args.max_candidates) if use_archive else []
        recent_regions = recent_regions[-256:]
        cfg = AllocationConfig(
            uniform_mix=args.uniform_mix,
            staleness_mix=args.staleness_mix,
            current_step=global_steps,
            beta=0.0 if args.method in ("herp_p", "plr", "go_explore") else 1.0,
            alpha=0.0 if args.method in ("herp_sigma", "go_explore") else 1.0,
        )
        q = priority_distribution(candidates, cfg)
        for region, priority in zip(candidates, q):
            region.priority = float(priority)

        for key in counts:
            cumulative[key] += counts[key]
        global_steps = sum(cumulative.values())
        update += 1
        assert global_steps <= args.total_timesteps + args.num_envs * args.num_steps, (
            f"budget breach: {global_steps} > {args.total_timesteps}"
        )

        # ---------------- eval ----------------
        region_values = list(archive)
        sigma_values = np.asarray([r.sigma_ema for r in region_values], dtype=np.float64)
        p_values = np.asarray([r.p_ema for r in region_values], dtype=np.float64)
        alignment_values = np.asarray([r.p_raw for r in region_values], dtype=np.float64)
        visit_values = np.asarray([r.count for r in region_values], dtype=np.float64)
        staleness_values = np.asarray(
            [max(0, global_steps - r.last_seen_step) for r in region_values], dtype=np.float64
        )
        elapsed = time.monotonic() - start_time
        row = dict(
            global_env_steps=global_steps,
            update=update,
            wall_time=elapsed,
            steps_per_second=(global_steps - session_start_steps) / max(elapsed, 1e-9),
            eval_return="",
            eval_success="",
            eval_success_final="",
            eval_steps="",
            eval_episodes="",
            num_regions=len(archive),
            num_archived_states=sum(len(r.snapshots) for r in archive),
            mean_sigma=float(sigma_values.mean()) if sigma_values.size else 0.0,
            std_sigma=float(sigma_values.std()) if sigma_values.size else 0.0,
            min_sigma=float(sigma_values.min()) if sigma_values.size else 0.0,
            max_sigma=float(sigma_values.max()) if sigma_values.size else 0.0,
            mean_p=float(p_values.mean()) if p_values.size else 0.0,
            std_p=float(p_values.std()) if p_values.size else 0.0,
            min_p=float(p_values.min()) if p_values.size else 0.0,
            max_p=float(p_values.max()) if p_values.size else 0.0,
            mean_alignment=float(alignment_values.mean()) if alignment_values.size else 0.0,
            alignment_positive_frac=(
                float((alignment_values > 0).mean()) if alignment_values.size else 0.0
            ),
            mean_region_visits=float(visit_values.mean()) if visit_values.size else 0.0,
            max_region_visits=float(visit_values.max()) if visit_values.size else 0.0,
            mean_region_staleness=float(staleness_values.mean()) if staleness_values.size else 0.0,
            num_candidates=len(candidates),
            priority_entropy=float(-(q * q.clamp_min(1e-12).log()).sum()) if q.numel() else 0.0,
            priority_effective_regions=(
                float(torch.exp(-(q * q.clamp_min(1e-12).log()).sum())) if q.numel() else 0.0
            ),
            priority_min=float(q.min()) if q.numel() else 0.0,
            priority_max=float(q.max()) if q.numel() else 0.0,
            adv_scale=float(adv_scale_state.std.item()),
            gradient_signature_norm_ref=float(reference_gradient.norm()) if reference_gradient is not None else 0.0,
            gradient_signature_norm_region=float(
                np.mean([r.gradient_norm for r in candidates]) if candidates else 0.0
            ),
            **counts,
            **{f"cumulative_{k}": v for k, v in cumulative.items()},
            normal_fraction=cumulative["normal_steps"] / max(1, global_steps),
            probe_fraction=cumulative["probe_steps"] / max(1, global_steps),
            allocated_fraction=cumulative["allocated_steps"] / max(1, global_steps),
            reference_fraction=cumulative["reference_steps"] / max(1, global_steps),
            **metrics,
        )
        did_eval = global_steps >= next_eval
        if did_eval:
            evaluation = evaluate(eval_adapter, agent, args, device, args.eval_episodes)
            row.update(evaluation)
            if args.save_model:
                torch.save(
                    dict(
                        format_version=2,
                        agent=agent.state_dict(),
                        optimizer=optimizer.state_dict(),
                        intrinsic=None if intrinsic is None else intrinsic.state_dict(),
                        intrinsic_optimizer=(
                            None if intrinsic_optimizer is None else intrinsic_optimizer.state_dict()
                        ),
                        args=asdict(args),
                        global_steps=global_steps,
                        cumulative=cumulative,
                        update=update,
                        next_eval=next_eval + args.eval_interval,
                        python_rng_state=random.getstate(),
                        numpy_rng_state=np.random.get_state(),
                        torch_rng_state=torch.get_rng_state(),
                        cuda_rng_state_all=(
                            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                        ),
                        generator_state=generator.get_state(),
                        wandb_run_id=None if wandb_run is None else wandb_run.id,
                    ),
                    out_dir / f"checkpoint_{global_steps}.pt",
                )
            next_eval += args.eval_interval

        for region in candidates:
            region_logger.log(
                dict(
                    step=global_steps,
                    env_id=args.env_id,
                    region_id=region.region_id,
                    sigma=region.sigma_ema,
                    p=region.p_ema,
                    alignment=region.p_raw,
                    priority=region.priority,
                    visit_count=region.count,
                    last_seen=region.last_seen_step,
                    last_probed=region.last_probed_step,
                )
            )
        logger.log(row)
        if wandb_run is not None and (did_eval or update % args.wandb_log_every == 0):
            wandb_payload = {"global_env_steps": global_steps}
            budget_keys = {
                "normal_steps", "probe_steps", "allocated_steps", "reference_steps",
                "normal_fraction", "probe_fraction", "allocated_fraction", "reference_fraction",
            }
            ppo_keys = {
                "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac",
                "grad_norm", "ppo_epochs", "ppo_minibatches", "ppo_early_stop",
                "policy_std", "explained_variance", "learning_rate",
            }
            rollout_keys = {
                "extrinsic_reward_mean", "extrinsic_reward_std",
                "total_reward_mean", "total_reward_std",
                "rollout_success_rate", "rollout_episode_end_rate",
                "advantage_mean", "advantage_std", "return_mean", "return_std",
                "value_mean", "value_std", "action_mean", "action_std", "logprob_mean",
            }
            herp_prefixes = (
                "num_region", "num_archived", "num_candidates", "mean_sigma", "std_sigma",
                "min_sigma", "max_sigma", "mean_p", "std_p", "min_p", "max_p",
                "mean_alignment", "alignment_", "mean_region", "max_region",
                "priority_", "adv_scale", "gradient_signature_",
            )
            for key, value in row.items():
                if key == "global_env_steps" or value == "":
                    continue
                if not isinstance(value, (int, float, np.number)):
                    continue
                if key.startswith("eval_"):
                    metric_name = f"eval/{key.removeprefix('eval_')}"
                elif key.startswith("cumulative_"):
                    metric_name = f"budget/cumulative_{key.removeprefix('cumulative_')}"
                elif key in budget_keys:
                    metric_name = f"budget/{key}"
                elif key.startswith("intrinsic_"):
                    metric_name = f"intrinsic/{key.removeprefix('intrinsic_')}"
                elif key.startswith(herp_prefixes):
                    metric_name = f"herp/{key}"
                elif key in ppo_keys:
                    metric_name = f"ppo/{key}"
                elif key in rollout_keys:
                    metric_name = f"rollout/{key}"
                else:
                    metric_name = f"system/{key}"
                wandb_payload[metric_name] = float(value)
            if did_eval and region_values:
                wandb_payload.update(
                    {
                        "herp/sigma_hist": wandb_module.Histogram(sigma_values),
                        "herp/p_hist": wandb_module.Histogram(p_values),
                        "herp/alignment_hist": wandb_module.Histogram(alignment_values),
                        "herp/visits_hist": wandb_module.Histogram(visit_values),
                    }
                )
                if q.numel():
                    wandb_payload["herp/priority_hist"] = wandb_module.Histogram(
                        q.detach().cpu().numpy()
                    )
            wandb_run.log(wandb_payload)
        print(json.dumps(row), flush=True)

    final_wall_time = time.monotonic() - start_time
    (out_dir / "complete.json").write_text(
        json.dumps(
            dict(global_env_steps=global_steps, counts=cumulative, wall_time=final_wall_time),
            indent=2,
        )
    )
    if wandb_run is not None:
        wandb_run.summary["final/global_env_steps"] = global_steps
        wandb_run.summary["final/wall_time"] = final_wall_time
        wandb_run.summary["final/steps_per_second"] = (
            (global_steps - session_start_steps) / max(final_wall_time, 1e-9)
        )
        wandb_run.finish()
    for a in (adapter, ref_adapter, eval_adapter):
        if a is not None:
            a.close()


if __name__ == "__main__":
    main()
