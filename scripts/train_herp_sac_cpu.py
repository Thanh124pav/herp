"""HERP-SAC on CPU: supports DMC, MetaWorld, and ManiSkill (physx_cpu).

Follows the same acquisition loop as train_herp_sac.py but generalizes
the adapter creation to all three benchmarks.

Methods: sac | sac_herp | sac_herp_p | sac_herp_sigma
"""
import os
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.json")
import json
import math
import random
import sys
import time
from copy import copy
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1] / "src")]

from herp.config import HERPV3Config
from herp.archive import Snapshot
from herp.envs import make_adapter
from herp.learners.sac import SACLearner
from herp.allocation_controller import AllocationController
from herp.video_events import CandidateStore, build_event, event_is_valid
from herp.provenance import save_provenance
from scripts.train_v3 import evaluate


class EnvShim:
    """Wraps a HERP adapter to look like a gymnasium VectorEnv for SACLearner."""
    def __init__(self, adapter):
        self.adapter = adapter
        obs_shape = (adapter.obs_dim,)
        act_shape = (adapter.action_dim,)
        low = adapter.action_low().cpu().numpy().flatten()
        high = adapter.action_high().cpu().numpy().flatten()
        import gymnasium
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(
            low=low, high=high, shape=act_shape, dtype=np.float32)
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.num_envs = 1


@dataclass
class SACArgs:
    num_envs: int = 1
    buffer_size: int = 1_000_000
    buffer_device: str = "cpu"
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    autotune: bool = True
    alpha: float = 0.2


def snapshot_without_origins(snapshot):
    """Shallow-copy metadata while preventing recursive provenance payloads."""
    clean = copy(snapshot)
    clean.origin_fragments = []
    return clean


def origin_fragment(base, start_snapshot):
    """Attach an exact replay start to the real fragment that created a region."""
    origin = copy(base)
    origin.category = "ORIGIN_TRAJECTORY"
    origin.restart_region_id = base.source_region_id
    origin.restart_snapshot_index = None
    origin.restart_snapshot_timestep = int(start_snapshot.timestep)
    origin.restart_snapshot_elapsed_steps = int(start_snapshot.elapsed_steps)
    origin.restart_snapshot = snapshot_without_origins(start_snapshot)
    return origin


def event_origin_fragments(fragments):
    """Collect unique trajectory-zero ancestry for all selected restart states."""
    origins, seen = [], set()
    for fragment in fragments:
        snapshot = fragment.restart_snapshot
        for origin in getattr(snapshot, "origin_fragments", []) if snapshot is not None else []:
            key = (int(origin.fragment_id), int(origin.restart_snapshot_timestep))
            if key not in seen:
                seen.add(key)
                origins.append(origin)
    return origins


def event_origin_trajectories_for_video(
        fragments, adapter, learner, normalizer, *, full_steps,
        max_total_steps, neighbor_fragments):
    """Prefer complete reset-origin trajectories for an event package.

    The training collector still uses fixed-length fragments. For video only,
    replay every exact root fragment and continue the current policy to
    terminal/time-limit. If doing that would exceed the configured payload,
    retain a bounded provenance window instead.
    """
    origins = event_origin_fragments(fragments)
    roots = [
        fragment for fragment in origins
        if int(fragment.restart_snapshot_elapsed_steps or 0) == 0
    ]
    estimated_steps = len(roots) * int(full_steps)
    if roots and (max_total_steps <= 0 or estimated_steps <= max_total_steps):
        extended = [
            extend_fragment_for_video(
                fragment, adapter, learner, normalizer, full_steps)
            for fragment in roots
        ]
        return extended, "full_reset_to_terminal"

    keep = max(8, min(12, int(neighbor_fragments)))
    return origins[-keep:], "neighbor_segments"


def extend_fragment_for_video(base, adapter, learner, normalizer, total_steps):
    """Replay the exact training prefix, then add a video-only continuation.

    The continuation is never inserted into replay and does not consume the
    training interaction budget. Keeping this separate preserves MIRA's
    32-step allocation/sigma semantics while producing legible trajectories.
    """
    if base.restart_snapshot is None:
        return base
    obs = adapter.restore_state(torch.tensor([0]), [base.restart_snapshot.env_state])
    states, next_states, actions = [], [], []
    rewards, terminated, truncated = [], [], []
    prefix = list(base.actions)
    total_steps = max(len(prefix), int(total_steps))

    for index in range(total_steps):
        with torch.no_grad():
            action = prefix[index][None].to(adapter.device) if index < len(prefix) else learner.act(obs)
        nxt, reward, term, trunc, info = adapter.step(action)
        done = bool(term[0] | trunc[0])
        actual = info.get("final_observation", nxt) if done and "final_observation" in info else nxt
        states.append(obs.detach().cpu())
        next_states.append(actual.detach().cpu())
        actions.append(action.detach().cpu())
        rewards.append(reward.detach().cpu().reshape(-1))
        terminated.append(term.detach().cpu().reshape(-1))
        truncated.append(trunc.detach().cpu().reshape(-1))
        obs = nxt
        if done:
            break

    next_tensor = torch.cat(next_states)
    return SimpleNamespace(
        source_region_id=base.source_region_id,
        policy_version=base.policy_version,
        fragment_id=base.fragment_id,
        slot_id=getattr(base, "slot_id", 0),
        category=base.category,
        states=torch.cat(states),
        state_features=normalizer.normalize(next_tensor),
        actions=torch.cat(actions),
        next_states=next_tensor,
        rewards=torch.cat(rewards),
        terminated=torch.cat(terminated),
        truncated=torch.cat(truncated),
        chain_region_id=getattr(base, "chain_region_id", None),
        successes=getattr(base, "successes", None),
        restart_region_id=base.restart_region_id,
        restart_snapshot_index=base.restart_snapshot_index,
        restart_snapshot_timestep=base.restart_snapshot_timestep,
        restart_snapshot_elapsed_steps=base.restart_snapshot_elapsed_steps,
        restart_snapshot=base.restart_snapshot,
    )


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["dmc", "metaworld", "maniskill"], required=True)
    p.add_argument("--env-id", required=True)
    p.add_argument("--method", choices=["sac", "sac_herp", "sac_herp_p", "sac_herp_sigma"], default="sac_herp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--root-floor", type=float, default=0.15)
    p.add_argument("--max-regions", type=int, default=256)
    p.add_argument("--score-temperature", type=float, default=1.0)
    p.add_argument("--relevance-threshold", type=float, default=0.03)
    p.add_argument("--min-relevance-measurements", type=int, default=3)
    p.add_argument("--max-restart-fraction", type=float, default=0.35)
    p.add_argument("--relevance-ema-tau", type=float, default=0.8)
    p.add_argument("--temporal-stratification", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--temporal-bins", type=int, default=3)
    p.add_argument("--temporal-exploration-mix", type=float, default=0.15)
    p.add_argument("--temporal-min-measurements", type=int, default=3)
    p.add_argument("--reference-states-per-bin", type=int, default=128)
    p.add_argument("--reference-buffer-per-bin", type=int, default=512)
    p.add_argument("--phase", default="performance")
    # SAC
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=5000)
    p.add_argument("--utd", type=int, default=1)
    # Eval
    p.add_argument("--eval-interval", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-interval", type=int, default=100_000)
    # Wandb
    p.add_argument("--wandb-mode", default="online")
    p.add_argument("--wandb-project", default="herp-framework")
    p.add_argument("--wandb-group", default="rq1-sac")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--wandb-tags", default="")
    # HERP
    p.add_argument("--future-horizon", type=int, default=32)
    p.add_argument("--capture-video-events", action="store_true")
    p.add_argument("--video-max-candidates", type=int, default=20)
    p.add_argument("--video-upload-top-k", type=int, default=5)
    p.add_argument("--video-min-allocated-fragments", type=int, default=2)
    p.add_argument("--video-min-distinct-outcomes", type=int, default=2)
    p.add_argument("--video-outcome-radius", type=float, default=1.0)
    p.add_argument("--video-rollout-steps", type=int, default=128,
                   help="Recorded steps per restart rollout; does not count toward training budget")
    p.add_argument("--video-origin-full-steps", type=int, default=1000,
                   help="Maximum steps for each reset-origin diagnostic trajectory")
    p.add_argument("--video-origin-max-total-steps", type=int, default=12000,
                   help="Fall back to neighboring fragments above this per-event origin payload; <=0 disables")
    p.add_argument("--video-origin-neighbor-fragments", type=int, default=10,
                   help="Fallback provenance window, clamped to 8..12 fragments")
    p.add_argument("--video-max-rollouts", type=int, default=0,
                   help="Maximum allocated rollouts retained per event; 0 keeps all")
    p.add_argument("--video-event-dir", default="")
    p.add_argument("--sigma-mode", default="shrinkage")
    p.add_argument("--sigma-kappa", type=float, default=8.0)
    # Env-specific
    p.add_argument("--control-mode", default="pd_ee_delta_pose")
    p.add_argument("--reward-mode", default="dense")
    p.add_argument("--training-freq", type=int, default=256)
    args = p.parse_args()

    if not 0 <= args.relevance_threshold < 1:
        p.error("--relevance-threshold must be in [0,1)")
    if not 0 <= args.max_restart_fraction < 1:
        p.error("--max-restart-fraction must be in [0,1)")
    if args.min_relevance_measurements < 1 or args.temporal_bins < 1:
        p.error("relevance measurements and temporal bins must be positive")
    if not 0 <= args.temporal_exploration_mix <= 1:
        p.error("--temporal-exploration-mix must be in [0,1]")
    if args.temporal_min_measurements < 1:
        p.error("--temporal-min-measurements must be positive")
    if args.reference_states_per_bin < 1 or args.reference_buffer_per_bin < args.reference_states_per_bin:
        p.error("reference buffer must hold at least reference-states-per-bin")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kw = dict(benchmark=args.benchmark, env_id=args.env_id, device=device)
    if args.benchmark == "metaworld":
        env_kw["reward_mode"] = args.reward_mode
    if args.benchmark == "maniskill":
        env_kw.update(control_mode=args.control_mode, sim_backend="physx_cpu",
                      render_backend="cpu")

    env = make_adapter(**env_kw).make(1, args.seed)
    env.reset(seed=args.seed)
    ev_env = make_adapter(**env_kw).make(1, args.seed + 10000)
    video_env = (make_adapter(**env_kw).make(1, args.seed + 20000)
                 if args.capture_video_events else None)

    shim = EnvShim(env)
    sac_args = SACArgs(gamma=args.gamma, tau=args.tau, batch_size=args.batch_size,
                       learning_starts=args.learning_starts)
    learner = SACLearner(shim, sac_args, device)

    cfg = HERPV3Config(
        future_horizon=args.future_horizon,
        min_common_steps=max(2, args.future_horizon // 4),
        root_floor=args.root_floor,
        max_regions=args.max_regions,
        score_temperature=args.score_temperature,
        sigma_predictor_kappa=args.sigma_kappa,
        relevance_threshold=args.relevance_threshold,
        min_relevance_measurements=args.min_relevance_measurements,
        max_restart_fraction=args.max_restart_fraction,
        relevance_ema_tau=args.relevance_ema_tau,
        temporal_stratification=args.temporal_stratification,
        temporal_bins=args.temporal_bins,
        temporal_exploration_mix=args.temporal_exploration_mix,
        temporal_min_measurements=args.temporal_min_measurements,
    )
    controller = AllocationController(env, cfg, args.seed, args.sigma_mode)
    controller.observer.capture_provenance = bool(args.capture_video_events)
    video_dir = Path(args.video_event_dir) if args.video_event_dir else out / "video_events"
    video_store = CandidateStore(video_dir, args.video_max_candidates) if args.capture_video_events else None

    save_provenance(out, Path(__file__).resolve().parents[1])
    (out / "config.json").write_text(json.dumps({
        "args": vars(args), "herp": asdict(cfg),
        "acquisition": "persistent_ordinary_stream_adaptive_temporal_snapshot_sampling",
        "replay": "uniform_region_labelled",
    }, indent=2))

    # Wandb
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=args.wandb_project, group=args.wandb_group,
            name=args.wandb_run_name or f"{args.method}-{args.env_id}-s{args.seed}",
            tags=[t for t in args.wandb_tags.split(",") if t],
            config=vars(args), mode=args.wandb_mode,
        )

    metrics_f = open(out / "metrics.jsonl", "a", buffering=1)
    counts = dict(ROOT_ACQUISITION=0, REGION_ACQUISITION=0, REFERENCE=0)
    steps = version = eval_steps = 0
    fragment_id = 0
    next_eval = 0
    obs = None
    learning = False
    done = True
    ordinary_snapshot = None
    ordinary_episode_id = 0
    reference_bins = [deque(maxlen=args.reference_buffer_per_bin)
                      for _ in range(args.temporal_bins)]
    started = time.time()

    def phase_for_elapsed(elapsed):
        usable = max(1, int(env.max_episode_steps) - args.future_horizon)
        elapsed = max(0, min(int(elapsed), usable))
        return min(args.temporal_bins - 1,
                   (elapsed * args.temporal_bins) // (usable + 1))

    def balanced_reference():
        """Current-policy reference states, balanced over available phases."""
        present = [bucket for bucket in reference_bins if bucket]
        if not present:
            return torch.empty(0, env.obs_dim)
        take = min(args.reference_states_per_bin, min(len(bucket) for bucket in present))
        return torch.stack([state for bucket in present for state in list(bucket)[-take:]])

    def eval_and_log():
        nonlocal eval_steps
        with torch.random.fork_rng(devices=[]):
            row = evaluate(ev_env, learner, args.eval_episodes, args.seed + 10000)
        eval_steps += row["eval_steps"]
        row.update(type="evaluation", step=steps, policy_version=version,
                   wall_seconds=time.time() - started)
        metrics_f.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if not k.startswith("episode_")}), flush=True)
        if run:
            run.log({"env_steps": steps, **{f"eval/{k}": v for k, v in row.items() if isinstance(v, (float, int))}})

    try:
        while steps < args.total_timesteps:
            if steps >= next_eval:
                eval_and_log()
                next_eval = steps + args.eval_interval

            start_steps = steps
            fragments = []
            pre = controller.begin_round()
            chosen_probs = defaultdict(list)
            remaining = min(args.training_freq, args.total_timesteps - steps)

            while remaining:
                use_herp = args.method != "sac" and learning
                reference_job = use_herp and not fragments

                if use_herp:
                    if reference_job:
                        rid, snapshot = 0, None
                    else:
                        rid, snapshot, probs = controller.choose(args.method.replace("sac_", ""))
                        chosen_probs[rid].append(float(probs[rid]))
                    if rid == 0:
                        # Region 0 is a persistent ordinary SAC stream.  Local
                        # restart jobs may temporarily use the same simulator,
                        # but the root state is restored and continues rather
                        # than being reset every future_horizon steps.
                        snapshot = ordinary_snapshot
                        root_started = snapshot is None
                        if root_started:
                            obs = env.reset()[0]
                            if args.capture_video_events:
                                saved = env.save_state(torch.tensor([0]))[0]
                                trajectory_start = Snapshot(
                                    saved, obs[0].detach().cpu().clone(), steps,
                                    episode_id=ordinary_episode_id,
                                    elapsed_steps=int(getattr(saved, "elapsed_steps", 0)))
                            else:
                                trajectory_start = None
                        else:
                            obs = env.restore_state(torch.tensor([0]), [snapshot.env_state])
                            trajectory_start = snapshot
                    else:
                        obs = env.restore_state(torch.tensor([0]), [snapshot.env_state])
                        trajectory_start = snapshot
                        root_started = False
                    controller.observer.new_fragment(rid, root_started)
                    controller.observer.snapshot_events.clear()
                else:
                    rid = 0
                    snapshot = None
                    root_started = obs is None or done
                    if root_started:
                        obs = env.reset()[0]

                states = []
                next_states = []
                actions_list = []
                rewards_list = []
                terminated_list = []
                truncated_list = []

                for _ in range(min(args.future_horizon, remaining)):
                    if use_herp and rid == 0:
                        elapsed = int(env.elapsed_steps()[0])
                        reference_bins[phase_for_elapsed(elapsed)].append(
                            obs[0].detach().cpu().clone())
                    with torch.no_grad():
                        if learning:
                            action = learner.act(obs)
                        else:
                            action = 2 * torch.rand((1, env.action_dim), device=device) - 1

                        if use_herp:
                            mean, logstd = learner.actor(obs)
                            controller.observer.observe(obs[0], mean[0], logstd[0], steps)

                    nxt, reward, term, trunc, info = env.step(action)
                    done = bool(term[0] | trunc[0])
                    actual = info.get("final_observation", nxt) if done and "final_observation" in info else nxt

                    learner.observe(dict(
                        obs=obs, next_obs=actual, action=action,
                        reward=reward.view(1, -1) if reward.dim() > 0 else reward.unsqueeze(0).unsqueeze(0),
                        done=term.float().view(1, -1) if term.dim() > 0 else term.float().unsqueeze(0).unsqueeze(0),
                        region=rid,
                    ))

                    states.append(obs.detach().cpu())
                    next_states.append(actual.detach().cpu())
                    actions_list.append(action.detach().cpu())
                    rewards_list.append(reward.detach().cpu().reshape(-1))
                    terminated_list.append(term.detach().cpu().reshape(-1))
                    truncated_list.append(trunc.detach().cpu().reshape(-1))

                    if not controller.active:
                        controller.calibrate(obs.cpu())

                    category = "REFERENCE" if reference_job else "REGION_ACQUISITION" if rid else "ROOT_ACQUISITION"
                    counts[category] += 1
                    obs = nxt
                    steps += 1
                    remaining -= 1

                    if done:
                        if use_herp:
                            controller.observer.end_episode()
                        break

                if use_herp and rid == 0:
                    if done:
                        ordinary_snapshot = None
                        ordinary_episode_id += 1
                    else:
                        saved = env.save_state(torch.tensor([0]))[0]
                        ordinary_snapshot = Snapshot(
                            saved, obs[0].detach().cpu().clone(), steps,
                            episode_id=ordinary_episode_id,
                            elapsed_steps=int(env.elapsed_steps()[0]))

                fragment = controller.fragment(
                    rid, torch.cat(states), torch.cat(next_states),
                    torch.cat(actions_list), version, root_started,
                    rewards=torch.cat(rewards_list),terminated=torch.cat(terminated_list),
                    truncated=torch.cat(truncated_list),restart_snapshot=snapshot,
                    category=category,fragment_id=fragment_id)
                fragments.append(fragment)
                if args.capture_video_events and use_herp and trajectory_start is not None:
                    ancestry = list(getattr(trajectory_start, "origin_fragments", []))
                    lineage = ancestry + [origin_fragment(fragment, trajectory_start)]
                    for _event_step, _event_region, created_snapshot in controller.observer.snapshot_events:
                        created_snapshot.origin_fragments = lineage
                    controller.observer.snapshot_events.clear()
                fragment_id += 1

            # HERP round diagnostics
            diagnostics = {}
            if args.method != "sac" and learning:
                diagnostics = controller.finish_round(
                    fragments,
                    balanced_reference(),
                    learner, version, pre)
                diagnostics.update(controller.temporal_diagnostics())
                measured_p = [float(r.p_ema) for r in controller.archive
                              if r.region_id > 0 and r.relevance_count > 0]
                diagnostics.update(
                    reference_bin_sizes=[len(bucket) for bucket in reference_bins],
                    ordinary_elapsed=(0 if ordinary_snapshot is None
                                      else int(ordinary_snapshot.elapsed_steps)),
                    relevance_eligible=sum(
                        int(r.region_id > 0 and
                            r.relevance_count >= cfg.min_relevance_measurements and
                            r.p_ema > cfg.relevance_threshold)
                        for r in controller.archive),
                    p_ema_min=(min(measured_p) if measured_p else 0.),
                    p_ema_mean=(float(np.mean(measured_p)) if measured_p else 0.),
                    p_ema_max=(max(measured_p) if measured_p else 0.),
                )

            if args.capture_video_events and learning:
                by_region = defaultdict(list)
                for fragment in fragments:
                    if fragment.category == "REGION_ACQUISITION" and fragment.restart_snapshot is not None:
                        by_region[fragment.source_region_id].append(fragment)
                root_fraction = sum(f.source_region_id == 0 for f in fragments) / max(1, len(fragments))
                for rid, event_fragments in by_region.items():
                    if len(event_fragments) < args.video_min_allocated_fragments:
                        continue
                    # Preserve each real 32-step allocation as an exact prefix,
                    # then continue only for visualization. Forking RNG keeps
                    # video capture from changing subsequent SAC exploration.
                    devices = ([learner.device.index or 0]
                               if learner.device.type == "cuda" else [])
                    with torch.random.fork_rng(devices=devices):
                        video_fragments = [
                            extend_fragment_for_video(
                                fragment, video_env, learner, controller.normalizer,
                                args.video_rollout_steps)
                            for fragment in (event_fragments if args.video_max_rollouts <= 0
                                             else event_fragments[:args.video_max_rollouts])
                        ]
                        origin_trajectories, origin_capture_mode = (
                            event_origin_trajectories_for_video(
                                event_fragments, video_env, learner,
                                controller.normalizer,
                                full_steps=args.video_origin_full_steps,
                                max_total_steps=args.video_origin_max_total_steps,
                                neighbor_fragments=args.video_origin_neighbor_fragments))
                    region = controller.archive.regions[rid]
                    metadata = dict(
                        event_id=f"step{steps:09d}_r{rid}_v{version}",
                        training_step=steps,policy_version=version,region_id=rid,
                        p_raw=float(region.p_raw),p_ema=float(region.p_ema),
                        sigma_raw=float(region.sigma_raw),q_direct=float(region.q_direct),
                        q_pred=float(region.q_pred),q_combined=float(region.q_combined),
                        allocation_prob=float(np.mean(chosen_probs[rid])) if chosen_probs[rid] else 0.0,
                        allocated_fragments=len(event_fragments),root_fraction=root_fraction,
                        allocation_entropy=0.0,
                        source=dict(benchmark=args.benchmark,env_id=args.env_id,device=device,
                                    sim_backend="cpu",render_backend="cpu",seed=args.seed,
                                    method=args.method,output_dir=str(out),
                                    training_run_id=None if run is None else run.id,
                                    training_fragment_steps=args.future_horizon,
                                    video_rollout_steps=args.video_rollout_steps,
                                    video_rollouts_recorded=len(video_fragments),
                                    origin_capture_mode=origin_capture_mode,
                                    origin_trajectories_recorded=len(origin_trajectories),
                                    origin_full_steps=args.video_origin_full_steps,
                                    replay_action_source="exact training prefix + video-only policy continuation"))
                    event = build_event(metadata=metadata,fragments=video_fragments,
                                        pre_fragments=origin_trajectories,
                                        radius=args.video_outcome_radius)
                    if event_is_valid(event,args.video_min_allocated_fragments,
                                      args.video_min_distinct_outcomes,0.0):
                        video_store.consider(event)

            # SAC updates
            losses = {}
            if steps >= args.learning_starts:
                activating = not learning
                if activating and args.method != "sac" and not done:
                    # Preserve the state reached by ordinary warmup.  The first
                    # MIRA reference fragment continues from here, allowing
                    # partitioning to begin at the actual episode phase rather
                    # than discarding warmup and resetting to step zero.
                    saved = env.save_state(torch.tensor([0]))[0]
                    ordinary_snapshot = Snapshot(
                        saved, obs[0].detach().cpu().clone(), steps,
                        episode_id=ordinary_episode_id,
                        elapsed_steps=int(env.elapsed_steps()[0]))
                learning = True
                controller.active = True
                for _ in range(int((steps - start_steps) * args.utd)):
                    losses = learner.update()

            version += 1

            row = dict(type="training", step=steps, budget=counts.copy(),
                       losses=losses, **diagnostics)
            metrics_f.write(json.dumps(row) + "\n")
            if run:
                run.log({"env_steps": steps,
                         **{f"loss/{k}": v for k, v in losses.items()},
                         **{f"herp/{k}": v for k, v in diagnostics.items()},
                         **{f"budget/{k}": v for k, v in counts.items()}})

            # Checkpoint
            if steps >= next_eval:
                learner.save(out / f"checkpoint_{steps}.learner.pt")

        # Final eval
        eval_and_log()
        learner.save(out / "checkpoint_final.learner.pt")

        summary = dict(
            status="completed", method=args.method, env_id=args.env_id,
            seed=args.seed, training_steps=steps,
            total_timesteps=args.total_timesteps,
            budget=counts, eval_steps=eval_steps,
            final_evaluation=json.loads(
                open(out / "metrics.jsonl").readlines()[-1]),
            wall_seconds=time.time() - started,
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

        if args.capture_video_events:
            from scripts.render_mira_video_events import render_ranked_events
            render_ranked_events(video_dir,top_k=args.video_upload_top_k,fps=30,
                                 pre_seconds=300,post_seconds=300,max_seconds=600,
                                 resolution=720,replay_tolerance=1e-3,
                                 wandb_run=run,wandb_module=wandb if run else None,
                                 wandb_prefix="mira_sac_events")

    finally:
        metrics_f.close()
        env.close()
        ev_env.close()
        if video_env is not None:
            video_env.close()
        if run:
            run.finish()

    print(f"Done: {args.method} {args.env_id} s{args.seed} in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
