"""HERP training entrypoint: PPO + benchmark adapter + performance-aware allocator."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.distributions import Normal

from herp.allocator import AllocationConfig, priority_distribution
from herp.archive import RegionArchive
from herp.envs import make_adapter
from herp.eval import evaluate_policy
from herp.gradient_signature import (
    empirical_fisher_diagonal,
    policy_gradient_signature,
    signature_parameters,
)
from herp.logging import CsvLogger
from herp.probe import default_obs_tensor, probe_region, reset_to_snapshot
from herp.regions import OnlineRegionizer, RegionizerConfig, RunningNormalizer
from herp.relevance import cosine_relevance
from herp.rollout_buffer import ALLOCATED, NORMAL, compute_gae
from herp.baselines import RND, Disagreement


@dataclass
class Args:
    benchmark: str = "maniskill"
    env_id: str = "PushCube-v1"
    method: str = "herp"
    seed: int = 0
    total_timesteps: int = 300_000
    rollout_horizon: int = 2048
    allocated_frac: float = 0.25
    allocated_rollout_horizon: int = 32
    scoring_interval: int = 5
    relevance_interval: int = 5
    reference_horizon: int = 128
    signature_horizon: int = 32
    p_estimator: str = "cosine"
    sigma_estimator: str = "pairwise"
    fisher_damping: float = 1e-3
    hybrid_eta: float = 0.5
    eps_hybrid: float = 1e-6
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    region_radius: float = 0.5
    max_regions: int = 128
    max_snapshots_per_region: int = 8
    archive_interval: int = 8
    max_candidates: int = 4
    probe_horizon: int = 8
    num_probes: int = 4
    num_env_repeats: int = 1
    probe_scale: float = 1.0
    gamma_branch: float = 0.95
    lambda_dyn: float = 0.0
    sigma_ema_tau: float = 0.9  # weight on the OLD estimate
    p_ema_tau: float = 0.9      # weight on the OLD estimate
    uniform_mix: float = 0.1
    staleness_mix: float = 0.05
    intrinsic_coef: float = 0.01
    intrinsic_epochs: int = 4
    eval_interval: int = 25_000
    eval_episodes: int = 50
    control_mode: str = "pd_ee_delta_pose"
    obs_mode: str = "state"
    reward_mode: str = "dense"
    sim_backend: str = "physx_cpu"
    render_backend: str = "cpu"
    output_dir: str = "outputs/herp"
    device: str = "cpu"
    torch_threads: int = 1


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, action_dim)
        self.logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        hidden = self.actor(obs.float())
        mean = self.actor_head(hidden)
        std = self.logstd.exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs.float()).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.get_distribution(obs)
        return dist.mean if deterministic else dist.sample()

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        dist = self.get_distribution(obs)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, self.value(obs)


def ppo_update(agent: Agent, optimizer, batch: dict, args: Args) -> dict[str, float]:
    device = torch.device(args.device)
    obs = batch["obs"].to(device)
    actions = batch["actions"].to(device)
    old_logprobs = batch["logprobs"].to(device)
    advantages = batch["advantages"].to(device)
    returns = batch["returns"].to(device)
    old_values = batch["values"].to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    minibatch = max(1, obs.shape[0] // args.num_minibatches)
    history = []
    for _ in range(args.update_epochs):
        inds = torch.randperm(obs.shape[0], device=device)
        for start in range(0, obs.shape[0], minibatch):
            mb = inds[start : start + minibatch]
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs[mb], actions[mb])
            logratio = newlogprob - old_logprobs[mb]
            ratio = logratio.exp()
            pg_loss1 = -advantages[mb] * ratio
            pg_loss2 = -advantages[mb] * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = 0.5 * (newvalue - returns[mb]).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
            history.append((len(mb), {
                "policy_loss": float(pg_loss.detach().cpu()),
                "value_loss": float(v_loss.detach().cpu()),
                "entropy": float(entropy_loss.detach().cpu()),
                "approx_kl": float(approx_kl.detach().cpu()),
                "clipfrac": float(clipfrac.detach().cpu()),
                "old_value_mean": float(old_values.mean().cpu()),
            }))
    total = sum(n for n, _ in history)
    return {key: sum(n * row[key] for n, row in history) / total for key in history[0][1]}


VALID_METHODS = ("ppo", "herp", "herp_sigma", "herp_p", "rnd", "disagreement", "go_explore", "plr")
VALID_P_ESTIMATORS = ("cosine", "dot", "fisher", "occupancy", "hybrid")
VALID_SIGMA_ESTIMATORS = ("pairwise", "branch", "return")


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
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=defaults[name])
    parsed = vars(parser.parse_args(argv))
    parsed.pop("config")
    args = Args(**parsed)
    if args.method not in VALID_METHODS:
        parser.error(f"Unknown method {args.method!r}; expected one of {VALID_METHODS}")
    if args.p_estimator not in VALID_P_ESTIMATORS:
        parser.error(f"Unknown p estimator {args.p_estimator!r}; expected one of {VALID_P_ESTIMATORS}")
    if args.sigma_estimator not in VALID_SIGMA_ESTIMATORS:
        parser.error(f"Unknown sigma estimator {args.sigma_estimator!r}")
    if not 0 <= args.allocated_frac < 1:
        parser.error("allocated-frac must be in [0, 1)")
    positives = (
        "total_timesteps", "rollout_horizon", "allocated_rollout_horizon", "probe_horizon",
        "num_probes", "num_env_repeats", "scoring_interval", "relevance_interval",
        "reference_horizon", "signature_horizon", "archive_interval", "eval_interval",
        "eval_episodes", "num_minibatches", "update_epochs", "max_candidates", "max_regions",
        "max_snapshots_per_region", "hidden", "torch_threads",
    )
    for name in positives:
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    for name in ("uniform_mix", "staleness_mix", "sigma_ema_tau", "p_ema_tau", "gamma",
                 "gae_lambda", "lambda_dyn", "hybrid_eta"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"{name} must be in [0, 1]")
    if args.fisher_damping <= 0 or args.probe_scale < 0:
        parser.error("fisher-damping must be positive and probe-scale nonnegative")
    return args


def build_adapter(args: Args, benchmark: str | None = None):
    """Instantiate + open an adapter for the requested benchmark family."""
    benchmark = benchmark or args.benchmark
    kwargs = dict(env_id=args.env_id, obs_mode=args.obs_mode, reward_mode=args.reward_mode)
    if benchmark == "maniskill":
        kwargs.update(
            control_mode=args.control_mode,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
        )
    adapter = make_adapter(benchmark, **kwargs)
    adapter.make()
    return adapter


def clamp_action(action, env):
    low = torch.as_tensor(env.action_space.low, dtype=action.dtype, device=action.device)
    high = torch.as_tensor(env.action_space.high, dtype=action.dtype, device=action.device)
    return action.clamp(low, high)


def collect_fragment(env, agent, args, horizon, source, regionizer=None,
                     archive_states=False, start_snapshot=None, episode_id=0,
                     global_step=0, reset_seed=None, adapter=None):
    """Roll out ``horizon`` steps of the current policy; bootstrap on truncation."""
    if horizon <= 0:
        raise ValueError("Cannot collect an empty fragment")
    to_tensor = adapter.obs_tensor if adapter is not None else default_obs_tensor
    if start_snapshot is None:
        obs, _ = env.reset(seed=reset_seed)
        return_so_far = 0.0
    else:
        obs, _ = reset_to_snapshot(env, start_snapshot, adapter=adapter)
        return_so_far = start_snapshot.return_so_far
    data = {k: [] for k in ("obs", "actions", "logprobs", "rewards", "dones", "values", "next_obs")}
    recent, episode_returns = [], []
    for t in range(horizon):
        obs_t = to_tensor(obs, args.device)
        if archive_states and regionizer is not None and adapter is not None and t % args.archive_interval == 0:
            snap_state = adapter.save_state()
            rid = regionizer.observe_snapshot(
                snap_state, obs_t.cpu(), global_step + t, episode_id, return_so_far, adapter.elapsed_steps()
            )
            recent.append(rid)
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs_t[None])
            applied = clamp_action(action.squeeze(0), env)
            next_obs, reward, terminated, truncated, _ = env.step(applied.cpu().numpy())
            next_t = to_tensor(next_obs, args.device)
            reward_f = float(torch.as_tensor(reward).mean())
            done = bool(terminated) or bool(truncated)
            adjusted_reward = reward_f
            if bool(truncated) and not bool(terminated):
                adjusted_reward += args.gamma * float(agent.value(next_t[None]))
        for key, val in dict(obs=obs_t.cpu(), actions=action.squeeze(0).cpu(),
                             logprobs=logprob.squeeze(0).cpu(), rewards=adjusted_reward,
                             dones=float(done), values=value.squeeze(0).cpu(),
                             next_obs=next_t.cpu()).items():
            data[key].append(val)
        return_so_far += reward_f
        obs = next_obs
        if done:
            episode_returns.append(return_so_far)
            episode_id += 1
            if start_snapshot is not None:
                break
            obs, _ = env.reset()
            return_so_far = 0.0
    batch = {k: torch.stack(v).float() if isinstance(v[0], torch.Tensor) else torch.tensor(v).float()
             for k, v in data.items()}
    with torch.no_grad():
        next_value = agent.value(to_tensor(obs, args.device)[None]).squeeze(0).cpu()
    batch["advantages"], batch["returns"] = compute_gae(
        batch["rewards"], batch["dones"], batch["values"], next_value, args.gamma, args.gae_lambda
    )
    batch.update(
        steps=len(data["obs"]),
        source=torch.full((len(data["obs"]),), source, dtype=torch.long),
        recent_regions=recent,
        episode_id=episode_id,
        next_value=next_value,
        episode_returns=episode_returns,
    )
    return batch


def concat_batches(batches):
    batches = [b for b in batches if b["steps"] > 0]
    keys = ("obs", "actions", "logprobs", "advantages", "returns", "values", "source")
    return {**{k: torch.cat([b[k] for b in batches]) for k in keys},
            "steps": sum(b["steps"] for b in batches)}


def signature(agent, batch, device, adv_scale=None):
    """Score-function signature with a *shared* adv scale (IMPLEMENTATION.md §8)."""
    return policy_gradient_signature(
        agent,
        batch["obs"].to(device),
        batch["actions"].to(device),
        batch["advantages"].to(device),
        adv_scale=adv_scale,
    )


def occupancy_scores(regionizer, observations):
    centers = torch.stack([r.centroid for r in regionizer.archive])
    features = torch.stack([regionizer.phi(obs) for obs in observations])
    ids = torch.cdist(features, centers).argmin(1)
    return torch.bincount(ids, minlength=len(centers)).float() / len(ids)


def _hybrid_score(occ_val: float, grad_val: float, eta: float, eps: float) -> float:
    grad_pos = max(0.0, grad_val)
    return (occ_val + eps) ** eta * (grad_pos + eps) ** (1.0 - eta)


def update_region_scores(env, agent, args, archive, regionizer, recent_regions,
                         reference_gradient, step, max_probe_steps, reference_batch=None,
                         fisher=None, adapter=None, adv_scale=None):
    use_sigma = args.method not in ("herp_p", "go_explore", "plr")
    use_gradient = (
        args.method in ("herp", "herp_p")
        and args.p_estimator in ("cosine", "dot", "fisher", "hybrid")
    )
    cost = args.num_probes * args.num_env_repeats * args.probe_horizon if use_sigma else 0
    cost += args.signature_horizon if use_gradient or args.method == "plr" else 0
    n = min(args.max_candidates, max_probe_steps // max(cost, 1))
    candidates = archive.candidates(recent_regions, step, max_candidates=n)
    probe_steps, signature_steps, batches, gradients = 0, 0, [], {}
    if reference_batch is not None:
        occ = occupancy_scores(regionizer, reference_batch["obs"])
    else:
        occ = torch.zeros(len(archive))
    for region in candidates:
        snapshot = archive.sample_snapshot(region)
        first = region.last_probed_step == 0
        if use_sigma:
            stats = probe_region(
                env, snapshot, agent, regionizer,
                args.num_probes, args.num_env_repeats, args.probe_horizon,
                args.probe_scale, args.gamma_branch, args.lambda_dyn,
                device=args.device, estimator=args.sigma_estimator, gamma=args.gamma,
                adapter=adapter,
            )
            probe_steps += stats["steps"]
            region.sigma_raw = stats["sigma"]
            region.sigma_ema = (
                region.sigma_raw
                if first
                else args.sigma_ema_tau * region.sigma_ema + (1 - args.sigma_ema_tau) * region.sigma_raw
            )
        else:
            region.sigma_ema = 1.0

        grad_alignment = 0.0
        if use_gradient or args.method == "plr":
            batch = collect_fragment(
                env, agent, args, args.signature_horizon, ALLOCATED,
                start_snapshot=snapshot, adapter=adapter,
            )
            signature_steps += batch["steps"]
            batches.append(batch)  # real on-policy samples, reused for PPO training
            grad = signature(agent, batch, args.device, adv_scale=adv_scale)
            region.gradient_norm = float(grad.norm())
            gradients[region.region_id] = grad
            if args.method == "plr":
                region.p_raw = float(batch["advantages"].abs().mean())
                grad_alignment = region.p_raw
            elif args.p_estimator == "cosine":
                region.p_raw = cosine_relevance(grad, reference_gradient)
                grad_alignment = region.p_raw
            elif args.p_estimator == "dot":
                region.p_raw = float(torch.dot(grad, reference_gradient))
                grad_alignment = region.p_raw
            elif args.p_estimator == "fisher":
                region.p_raw = float(torch.dot(reference_gradient, grad / (fisher + args.fisher_damping)))
                grad_alignment = region.p_raw
            else:  # hybrid: combine occupancy and cosine alignment
                grad_alignment = cosine_relevance(grad, reference_gradient)
                region.p_raw = _hybrid_score(
                    float(occ[region.region_id]), grad_alignment, args.hybrid_eta, args.eps_hybrid
                )
        elif args.method == "go_explore":
            region.p_raw = 1.0 / np.sqrt(region.count + 1)
        elif args.method == "herp_sigma":
            region.p_raw = 1.0
        else:  # occupancy-only (or herp with p_estimator=occupancy)
            region.p_raw = float(occ[region.region_id])
        region.last_probed_step = step

    scale = 1.0
    if candidates and args.p_estimator in ("dot", "fisher"):
        scale = max(float(np.median([abs(r.p_raw) for r in candidates])), 1e-8)
    for region in candidates:
        positive = max(0.0, region.p_raw / scale)
        region.p_ema = args.p_ema_tau * region.p_ema + (1 - args.p_ema_tau) * positive
        if args.method == "herp_sigma":
            region.p_ema = 1.0
    return candidates, probe_steps, signature_steps, batches


def save_checkpoint(path, agent, optimizer, args, archive, regionizer, steps, update):
    payload = dict(
        agent=agent.state_dict(), optimizer=optimizer.state_dict(), config=asdict(args),
        archive=archive, regionizer=regionizer, global_env_steps=steps, update=update,
        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
    )
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def git_commit_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def benchmark_version(benchmark: str) -> str | None:
    try:
        if benchmark == "maniskill":
            import mani_skill
            return getattr(mani_skill, "__version__", None)
        if benchmark == "metaworld":
            import metaworld
            return getattr(metaworld, "__version__", None)
        if benchmark == "fetch":
            import gymnasium_robotics
            return getattr(gymnasium_robotics, "__version__", None)
    except Exception:
        return None
    return None


def write_provenance(out_dir: Path, args: Args, sources: list[Path]) -> None:
    payload = {
        "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "benchmark": args.benchmark,
        "benchmark_version": benchmark_version(args.benchmark),
        "git_commit": git_commit_sha(),
        "seed": args.seed,
        "device": args.device,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "total_timesteps": args.total_timesteps,
    }
    (out_dir / "provenance.json").write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start_time = time.monotonic()
    out_dir = Path(args.output_dir) / f"{args.env_id}_{args.method}_seed{args.seed}_{time.time_ns()}"
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(args), indent=2))
    logger = CsvLogger(out_dir / "metrics.csv")
    region_logger = CsvLogger(out_dir / "regions.csv")

    # Adapter build sets any benchmark-specific env vars (e.g. Vulkan ICD on WSL)
    # BEFORE any code path can import the simulator. Write provenance afterwards
    # so ``benchmark_version`` can safely ``import mani_skill``.
    adapter = build_adapter(args)
    ref_adapter = build_adapter(args)
    eval_adapter = build_adapter(args)
    sources = [Path(__file__), *Path("src/herp").rglob("*.py")]
    write_provenance(out_dir, args, sources)
    env, ref_env, eval_env = adapter.env, ref_adapter.env, eval_adapter.env
    obs, _ = env.reset(seed=args.seed)
    obs_dim = adapter.obs_tensor(obs).numel()
    action_dim = int(np.prod(env.action_space.shape))
    agent = Agent(obs_dim, action_dim, args.hidden).to(args.device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    archive = RegionArchive(args.max_snapshots_per_region, args.seed)
    regionizer = OnlineRegionizer(
        obs_dim, archive,
        RegionizerConfig(region_radius=args.region_radius, max_regions=args.max_regions),
    )
    intrinsic = (
        RND(obs_dim, action_dim) if args.method == "rnd"
        else Disagreement(obs_dim, action_dim) if args.method == "disagreement"
        else None
    )
    if intrinsic is not None:
        intrinsic.to(args.device)
        intrinsic_optimizer = torch.optim.Adam(
            [p for p in intrinsic.parameters() if p.requires_grad], lr=args.learning_rate
        )
        obs_normalizer, bonus_normalizer = RunningNormalizer(obs_dim), RunningNormalizer(1)

    use_archive = args.method not in ("ppo", "rnd", "disagreement")
    needs_reference = args.method in ("herp", "herp_p")
    global_steps, update, eval_total = 0, 0, 0
    cumulative = dict(normal_steps=0, probe_steps=0, allocated_steps=0, reference_steps=0)
    reference_batch, reference_gradient, fisher = None, None, None
    candidates = []
    adv_scale_state = RunningNormalizer(1)
    generator = torch.Generator().manual_seed(args.seed)
    next_eval = args.eval_interval
    initial = evaluate_policy(eval_env, agent, args.eval_episodes, args.device, adapter=eval_adapter)
    eval_total += initial["eval_steps"]
    (out_dir / "initial_eval.json").write_text(json.dumps(initial))
    evaluation = initial

    while global_steps < args.total_timesteps:
        round_budget = min(args.rollout_horizon, args.total_timesteps - global_steps, next_eval - global_steps)
        extra = int(round_budget * args.allocated_frac) if use_archive else 0
        normal_horizon = max(1, round_budget - extra)
        counts = dict(normal_steps=0, probe_steps=0, allocated_steps=0, reference_steps=0)
        normal = collect_fragment(
            env, agent, args, normal_horizon, NORMAL, regionizer, use_archive,
            global_step=global_steps, adapter=adapter,
        )
        counts["normal_steps"] += normal["steps"]
        batches = [normal]
        remaining = round_budget - normal["steps"]

        # Update the shared advantage scale with the raw (unnormalized) training advantages.
        adv_scale_state.update(normal["advantages"].reshape(-1, 1))
        adv_scale = max(float(adv_scale_state.std.item()), 1e-6)

        reference_refreshed = False
        if needs_reference and remaining > 0 and (
            reference_batch is None or update % args.relevance_interval == 0
        ):
            horizon = min(args.reference_horizon, remaining)
            reference_batch = collect_fragment(
                ref_env, agent, args, horizon, 3,
                reset_seed=1_000_000 + args.seed * 10_000 + update,
                adapter=ref_adapter,
            )
            counts["reference_steps"] += reference_batch["steps"]
            remaining -= reference_batch["steps"]
            reference_refreshed = True
        if reference_refreshed:
            reference_gradient = signature(agent, reference_batch, args.device, adv_scale=adv_scale)
            if args.p_estimator == "fisher":
                fisher = empirical_fisher_diagonal(
                    agent, reference_batch["obs"].to(args.device), reference_batch["actions"].to(args.device)
                )

        if use_archive and remaining > 0 and (not needs_reference or reference_gradient is not None) \
                and update % args.scoring_interval == 0:
            candidates, probe_steps, signature_steps, signature_batches = update_region_scores(
                env, agent, args, archive, regionizer, normal["recent_regions"],
                reference_gradient, global_steps + normal["steps"], remaining,
                reference_batch, fisher, adapter=adapter, adv_scale=adv_scale,
            )
            counts["probe_steps"] += probe_steps
            counts["allocated_steps"] += signature_steps
            remaining -= probe_steps + signature_steps
            batches.extend(signature_batches)

        allocation = {}
        cfg = AllocationConfig(
            uniform_mix=args.uniform_mix, staleness_mix=args.staleness_mix,
            current_step=global_steps + normal["steps"],
            beta=0.0 if args.method in ("herp_p", "plr", "go_explore") else 1.0,
        )
        q = priority_distribution(candidates, cfg)
        for region, priority in zip(candidates, q):
            region.priority = float(priority)
        while remaining > 0:
            if candidates:
                draw = int(torch.multinomial(q, 1, generator=generator))
                region = candidates[draw]
                fragment = collect_fragment(
                    env, agent, args,
                    min(remaining, args.allocated_rollout_horizon),
                    ALLOCATED, start_snapshot=archive.sample_snapshot(region),
                    adapter=adapter,
                )
                allocation[region.region_id] = allocation.get(region.region_id, 0) + fragment["steps"]
                counts["allocated_steps"] += fragment["steps"]
            else:
                fragment = collect_fragment(
                    env, agent, args, remaining, NORMAL, regionizer, use_archive,
                    global_step=global_steps + round_budget - remaining, adapter=adapter,
                )
                counts["normal_steps"] += fragment["steps"]
            batches.append(fragment)
            remaining -= fragment["steps"]

        intrinsic_loss, intrinsic_mean = 0.0, 0.0
        if intrinsic is not None:
            obs_normalizer.update(normal["obs"])
            x = obs_normalizer.normalize(normal["obs"]).to(args.device)
            y = obs_normalizer.normalize(normal["next_obs"]).to(args.device)
            a = clamp_action(normal["actions"].to(args.device), env)
            with torch.no_grad():
                bonus = intrinsic.bonus(x, a, y).cpu()
                bonus_normalizer.update(bonus[:, None])
                bonus = (bonus / bonus_normalizer.std.item()).clamp(0, 10)
                intrinsic_mean = float(bonus.mean())
                normal["advantages"], normal["returns"] = compute_gae(
                    normal["rewards"] + args.intrinsic_coef * bonus, normal["dones"],
                    normal["values"], normal["next_value"], args.gamma, args.gae_lambda,
                )
            for _ in range(args.intrinsic_epochs):
                loss = intrinsic.loss(x, a, y)
                intrinsic_optimizer.zero_grad()
                loss.backward()
                intrinsic_optimizer.step()
                intrinsic_loss += float(loss.detach()) / args.intrinsic_epochs

        global_steps += sum(counts.values())
        assert sum(counts.values()) == round_budget
        for key in counts:
            cumulative[key] += counts[key]
        assert sum(cumulative.values()) == global_steps <= args.total_timesteps

        metrics = ppo_update(agent, optimizer, concat_batches(batches), args)
        update += 1

        sigmas = [r.sigma_ema for r in archive] or [0.0]
        ps = [r.p_ema for r in archive] or [0.0]
        row = dict(
            global_env_steps=global_steps, update=update,
            wall_time=time.monotonic() - start_time,
            train_return=float(np.mean(normal["episode_returns"])) if normal["episode_returns"] else "",
            eval_return="", eval_success="", eval_success_final="", eval_steps="", eval_episodes="",
            cumulative_eval_steps=eval_total,
            num_regions=len(archive),
            num_archived_states=sum(len(r.snapshots) for r in archive),
            mean_sigma=float(np.mean(sigmas)), median_sigma=float(np.median(sigmas)),
            max_sigma=float(np.max(sigmas)),
            mean_p=float(np.mean(ps)), median_p=float(np.median(ps)),
            fraction_positive_alignment=float(np.mean([r.p_raw > 0 for r in candidates])) if candidates else 0.0,
            priority_entropy=float(-(q * q.clamp_min(1e-12).log()).sum()),
            allocation_histogram=json.dumps(allocation),
            adv_scale=float(adv_scale),
            gradient_signature_norm_ref=float(reference_gradient.norm()) if reference_gradient is not None else 0.0,
            gradient_signature_norm_region=float(np.mean([r.gradient_norm for r in candidates])) if candidates else 0.0,
            intrinsic_loss=intrinsic_loss, intrinsic_mean=intrinsic_mean,
            **counts, **{f"cumulative_{k}": v for k, v in cumulative.items()}, **metrics,
        )
        if global_steps >= next_eval or global_steps == args.total_timesteps:
            evaluation = evaluate_policy(eval_env, agent, args.eval_episodes, args.device,
                                         adapter=eval_adapter)
            eval_total += evaluation["eval_steps"]
            row.update(evaluation, cumulative_eval_steps=eval_total,
                       wall_time=time.monotonic() - start_time)
            save_checkpoint(out_dir / f"checkpoint_{global_steps}.pt",
                            agent, optimizer, args, archive, regionizer, global_steps, update)
            next_eval += args.eval_interval

        for region in candidates:
            region_logger.log(dict(
                step=global_steps, env_id=args.env_id, region_id=region.region_id,
                sigma=region.sigma_ema, p=region.p_ema, alignment=region.p_raw,
                priority=region.priority, visit_count=region.count,
                last_seen=region.last_seen_step, last_probed=region.last_probed_step,
            ))
        logger.log(row)
        print(json.dumps(row), flush=True)

    (out_dir / "complete.json").write_text(json.dumps(dict(
        global_env_steps=global_steps, counts=cumulative,
        wall_time=time.monotonic() - start_time, final_eval=evaluation,
    ), indent=2))
    for a in (adapter, ref_adapter, eval_adapter):
        a.close()


if __name__ == "__main__":
    main()
