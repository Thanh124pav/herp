"""HERP-TD-MPC2: HERP allocation with TD-MPC2 world model + MPPI planner.

Uses ManiSkillAdapter for env management and HERP allocation.
Imports TD-MPC2 agent/buffer from third_party/ManiSkill.
TD-MPC2's policy prior (Gaussian) provides mean/logstd for HERP observer.

Methods: tdmpc2 | tdmpc2_herp
"""
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace

os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from herp.allocation_controller import AllocationController
from herp.archive import Snapshot
from herp.config import HERPV3Config
from herp.envs import make_adapter
from herp.provenance import save_provenance
from herp.video_events import CandidateStore, build_event, event_is_valid
from herp.vector_acquisition import VectorPartitionObserver


TDMPC2_SRC = ROOT / "third_party" / "ManiSkill" / "examples" / "baselines" / "tdmpc2"
sys.path.insert(0, str(TDMPC2_SRC))

from tdmpc2 import TDMPC2 as TDMPC2Agent
from common.buffer import Buffer
from common import math as tdmpc2_math


def make_tdmpc2_cfg(obs_dim, action_dim, episode_length, args):
    """Create a config namespace with all attributes TD-MPC2 needs."""
    cfg = SimpleNamespace(
        # env
        obs="state",
        obs_shape={"state": (obs_dim,)},
        action_dim=action_dim,
        episode_length=episode_length,
        episode_lengths=[episode_length],
        multitask=False,
        task_dim=0,
        tasks=[args.env_id],
        env_id=args.env_id,
        include_state=False,
        # training
        steps=args.total_timesteps,
        batch_size=256,
        reward_coef=0.1,
        value_coef=0.1,
        consistency_coef=20,
        rho=0.5,
        lr=args.learning_rate,
        enc_lr_scale=0.3,
        grad_clip_norm=20,
        tau=0.01,
        discount_denom=5,
        discount_min=0.95,
        discount_max=0.995,
        buffer_size=min(1_000_000, args.total_timesteps),
        steps_per_update=1,
        # planning
        mpc=True,
        iterations=6,
        num_samples=512,
        num_elites=64,
        num_pi_trajs=24,
        horizon=3,
        min_std=0.05,
        max_std=2,
        temperature=0.5,
        # actor
        log_std_min=-10,
        log_std_max=2,
        entropy_coef=1e-4,
        # critic
        num_bins=101,
        vmin=-10,
        vmax=+10,
        # architecture (model_size=1)
        enc_dim=256,
        mlp_dim=384,
        latent_dim=128,
        num_enc_layers=2,
        num_q=2,
        dropout=0.01,
        simnorm_dim=8,
        rgb_state_enc_dim=64,
        rgb_state_num_enc_layers=1,
        rgb_state_latent_dim=64,
        num_channels=32,
        true_latent_dim=128,
        # seed
        seed_steps=args.num_envs * episode_length * 2,
        seed=args.seed,
        # env parallelism
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        env_type="gpu",
        # eval
        eval_freq=args.eval_interval,
        eval_episodes_per_env=max(1, args.eval_episodes // args.num_eval_envs),
        # logging
        wandb=args.wandb_mode != "disabled",
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_entity="hust_edu_vn",
        wandb_silent=False,
        save_video_local=False,
        save_agent=True,
        save_csv=False,
        render_mode="rgb_array",
        render_size=64,
        control_mode=args.control_mode,
    )
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    return cfg


class TDMPC2Learner:
    """Wraps TD-MPC2 agent to provide the interface HERP needs for finish_round."""
    def __init__(self, agent, obs_dim, device):
        self.agent = agent
        self.obs_dim = obs_dim
        self.device = device

    def gradient_signature(self, batch):
        obs = batch["obs"].to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        z = self.agent.model.encode(obs, None)
        mu, pi, log_pi, _ = self.agent.model.pi(z, None)
        qs = self.agent.model.Q(z, pi, None, return_type="avg")
        loss = (self.agent.cfg.entropy_coef * log_pi - qs).mean()
        grads = torch.autograd.grad(loss, self.agent.model._pi.parameters(),
                                     allow_unused=True)
        return torch.cat([g.detach().flatten() for g in grads if g is not None])

    def get_policy_stats(self, obs):
        """Get mean and log_std from TD-MPC2's policy prior for HERP observer."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        z = self.agent.model.encode(obs.to(self.device), None)
        raw = self.agent.model._pi(z)
        mu, log_std = raw.chunk(2, dim=-1)
        log_std = tdmpc2_math.log_std(log_std,
                                       self.agent.model.log_std_min,
                                       self.agent.model.log_std_dif)
        return mu.detach().cpu(), log_std.detach().cpu()


@torch.no_grad()
def evaluate(env, agent, episodes, seed, device, num_eval_envs):
    obs, _ = env.reset(seed=seed)
    n = env.num_envs
    ep_returns, ep_success = [], []
    ret = torch.zeros(n, device=device)
    suc = torch.zeros(n, device=device)
    steps = 0
    max_steps = 200 * max(1, (episodes + n - 1) // n) * 2
    while len(ep_returns) < episodes and steps < max_steps:
        action = agent.act(obs, t0=(steps == 0), eval_mode=True)
        obs, r, term, trunc, info = env.step(action)
        ret += r.to(device).float()
        cur = env.success_from_info(info).float().to(device)
        done = term.to(device) | trunc.to(device)
        fi = info.get("final_info") if isinstance(info, dict) else None
        end = env.success_from_info(fi).float().to(device) if fi is not None else cur
        suc = torch.maximum(suc, torch.where(done, end, cur))
        if bool(done.any()):
            for i in torch.where(done)[0].tolist():
                ep_returns.append(float(ret[i]))
                ep_success.append(float(suc[i]))
                ret[i] = 0.0
                suc[i] = 0.0
        steps += 1
    ep_returns = ep_returns[:episodes]
    ep_success = ep_success[:episodes]
    return dict(
        eval_return=float(np.mean(ep_returns)) if ep_returns else 0.0,
        success_once=float(np.mean(ep_success)) if ep_success else 0.0,
        eval_steps=steps * n,
        eval_episodes=len(ep_returns),
    )


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="walker-run")
    p.add_argument("--method", choices=["tdmpc2", "tdmpc2_herp"], default="tdmpc2_herp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--step-offset", type=int, default=0,
                   help="Global environment-step offset for checkpoint fine-tuning")
    p.add_argument("--init-checkpoint", default="",
                   help="TD-MPC2 model checkpoint used to initialize a continuation run")
    p.add_argument("--init-pi-scale", type=float, default=0.,
                   help="Restore RunningScale value omitted by upstream checkpoints")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--eval-at-start", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-eval-envs", type=int, default=1)
    p.add_argument("--root-floor", type=float, default=0.8)
    p.add_argument("--relevance-threshold", type=float, default=0.03)
    p.add_argument("--min-relevance-measurements", type=int, default=3)
    p.add_argument("--max-restart-fraction", type=float, default=0.2)
    p.add_argument("--relevance-ema-tau", type=float, default=0.8)
    p.add_argument("--temporal-stratification", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--temporal-bins", type=int, default=3)
    p.add_argument("--temporal-exploration-mix", type=float, default=0.15)
    p.add_argument("--temporal-min-measurements", type=int, default=3)
    p.add_argument("--temporal-gate-temperature", type=float, default=0.03)
    p.add_argument("--reference-states-per-bin", type=int, default=128)
    p.add_argument("--reference-buffer-per-bin", type=int, default=512)
    p.add_argument("--control-mode", default="none")
    p.add_argument("--learning-starts", type=int, default=0,
                   help="0 uses max(1000, num_envs * true episode length)")
    p.add_argument("--mira-starts", type=int, default=5_000)
    p.add_argument("--eval-interval", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-interval", type=int, default=125_000)
    p.add_argument("--future-horizon", type=int, default=32)
    p.add_argument("--sigma-mode", default="shrinkage")
    p.add_argument("--wandb-mode", default="online")
    p.add_argument("--wandb-project", default="herp-v3")
    p.add_argument("--wandb-group", default="icra2027-tdmpc2-mira-video-500k")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--wandb-tags", default="")
    p.add_argument("--capture-video-events", action="store_true")
    p.add_argument("--video-max-candidates", type=int, default=20)
    p.add_argument("--video-upload-top-k", type=int, default=5)
    p.add_argument("--video-min-allocated-fragments", type=int, default=2)
    p.add_argument("--video-min-distinct-outcomes", type=int, default=2)
    p.add_argument("--video-outcome-radius", type=float, default=1.0)
    args = p.parse_args()
    if not 0 <= args.relevance_threshold < 1:
        p.error("--relevance-threshold must be in [0,1)")
    if not 0 <= args.max_restart_fraction < 1:
        p.error("--max-restart-fraction must be in [0,1)")
    if not 0 <= args.temporal_exploration_mix <= 1:
        p.error("--temporal-exploration-mix must be in [0,1]")
    if args.min_relevance_measurements < 1 or args.temporal_min_measurements < 1:
        p.error("relevance measurement counts must be positive")
    if args.temporal_bins < 1:
        p.error("--temporal-bins must be positive")
    if args.temporal_gate_temperature <= 0:
        p.error("--temporal-gate-temperature must be positive")
    if args.step_offset < 0 or args.step_offset >= args.total_timesteps:
        p.error("--step-offset must be in [0, total-timesteps)")
    if args.init_pi_scale < 0 or args.learning_rate <= 0:
        p.error("--init-pi-scale must be nonnegative and --learning-rate positive")
    if (args.reference_states_per_bin < 1 or
            args.reference_buffer_per_bin < args.reference_states_per_bin):
        p.error("reference buffer must hold at least reference-states-per-bin")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_adapter(benchmark="dmc", env_id=args.env_id, device=device).make(args.num_envs, args.seed)
    eval_env = make_adapter(benchmark="dmc", env_id=args.env_id, device=device).make(args.num_eval_envs, args.seed + 10000)
    cfg = make_tdmpc2_cfg(env.obs_dim, env.action_dim, env.max_episode_steps, args)
    learning_starts = (args.learning_starts if args.learning_starts > 0
                       else max(1000, args.num_envs * env.max_episode_steps))
    mira_starts = max(args.mira_starts, learning_starts)
    cfg.seed_steps = learning_starts
    agent = TDMPC2Agent(cfg)
    initialized_agent = bool(args.init_checkpoint)
    if initialized_agent:
        agent.load(args.init_checkpoint)
        if args.init_pi_scale > 0:
            agent.scale._value.fill_(args.init_pi_scale)
    buffer = Buffer(cfg)

    herp_cfg = HERPV3Config(
        future_horizon=args.future_horizon,
        min_common_steps=max(2, args.future_horizon // 4),
        root_floor=args.root_floor,
        relevance_threshold=args.relevance_threshold,
        min_relevance_measurements=args.min_relevance_measurements,
        max_restart_fraction=args.max_restart_fraction,
        relevance_ema_tau=args.relevance_ema_tau,
        temporal_stratification=args.temporal_stratification,
        temporal_bins=args.temporal_bins,
        temporal_exploration_mix=args.temporal_exploration_mix,
        temporal_min_measurements=args.temporal_min_measurements,
        temporal_gate_temperature=args.temporal_gate_temperature,
    )
    controller = AllocationController(env, herp_cfg, args.seed, args.sigma_mode)
    controller.observer = VectorPartitionObserver(env, controller.archive, controller.normalizer, herp_cfg)
    learner = TDMPC2Learner(agent, env.obs_dim, device)

    video_dir = out / "video_events"
    video_store = CandidateStore(video_dir, args.video_max_candidates) if args.capture_video_events else None
    save_provenance(out, ROOT)
    (out / "config.json").write_text(json.dumps({
        "args": vars(args), "herp": asdict(herp_cfg),
        "resolved": {"episode_length": env.max_episode_steps,
                     "discount": float(agent.discount),
                     "learning_starts": learning_starts,
                     "mira_starts": mira_starts,
                     "step_offset": args.step_offset,
                     "init_checkpoint": args.init_checkpoint or None},
        "acquisition": "persistent_vector_root_stream_soft_temporal_gate",
    }, indent=2))

    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=args.wandb_project, group=args.wandb_group,
            name=args.wandb_run_name or f"{args.method}-{args.env_id}-s{args.seed}",
            tags=[x for x in args.wandb_tags.split(",") if x],
            config=vars(args), mode=args.wandb_mode,
        )

    from tensordict.tensordict import TensorDict

    def to_td(obs_t, action=None, reward=None):
        obs_t = obs_t.detach().unsqueeze(1).cpu()
        if action is None:
            action = torch.full((args.num_envs, cfg.action_dim), float("nan"))
        else:
            action = action.detach().cpu()
        if reward is None:
            reward = torch.full((args.num_envs,), float("nan"))
        else:
            reward = reward.detach().cpu()
        return TensorDict(
            dict(obs=obs_t, action=action.unsqueeze(1), reward=reward.unsqueeze(1)),
            batch_size=(args.num_envs, 1),
        )

    ordinary_snapshots = [None] * args.num_envs
    reference_bins = [deque(maxlen=args.reference_buffer_per_bin)
                      for _ in range(args.temporal_bins)]

    def phase_for_elapsed(elapsed):
        usable = max(1, int(env.max_episode_steps) - args.future_horizon)
        elapsed = max(0, min(int(elapsed), usable))
        return min(args.temporal_bins - 1,
                   (elapsed * args.temporal_bins) // (usable + 1))

    def balanced_reference():
        present = [bucket for bucket in reference_bins if bucket]
        if not present:
            return torch.empty(0, env.obs_dim)
        take = min(args.reference_states_per_bin,
                   min(len(bucket) for bucket in present))
        return torch.stack([state for bucket in present
                            for state in list(bucket)[-take:]])

    def restore_or_reset(sources, snapshots):
        obs_curr = torch.zeros(args.num_envs, env.obs_dim, device=device)
        job_restarted = sources > 0
        root_started = torch.zeros(args.num_envs, dtype=torch.bool)
        roots = torch.where(sources == 0)[0]
        fresh_roots = torch.tensor(
            [i for i in roots.tolist() if ordinary_snapshots[i] is None],
            dtype=torch.long)
        saved_roots = torch.tensor(
            [i for i in roots.tolist() if ordinary_snapshots[i] is not None],
            dtype=torch.long)
        if len(fresh_roots):
            reset_obs, _ = env.reset_indices(fresh_roots)
            obs_curr[fresh_roots.to(device)] = reset_obs[fresh_roots.to(device)]
            job_restarted[fresh_roots] = True
            root_started[fresh_roots] = True
        if len(saved_roots):
            picked = [ordinary_snapshots[i] for i in saved_roots.tolist()]
            restored = env.restore_state(saved_roots, [s.env_state for s in picked])
            obs_curr[saved_roots.to(device)] = restored
            starts = torch.tensor(
                [s.elapsed_steps == 0 for s in picked], dtype=torch.bool)
            job_restarted[saved_roots] = starts
            root_started[saved_roots] = starts
        local = torch.where(sources > 0)[0]
        if len(local):
            picked = [snapshots[int(i)] for i in local]
            restored = env.restore_state(local, [s.env_state for s in picked])
            obs_curr[local.to(device)] = restored
        return obs_curr, root_started, job_restarted

    @torch.no_grad()
    def eval_policy():
        obs_eval, _ = eval_env.reset(seed=args.seed + 10000)
        returns = torch.zeros(args.num_eval_envs, device=device)
        finished = []
        t = 0
        max_steps = eval_env.max_episode_steps * max(1, args.eval_episodes)
        while len(finished) < args.eval_episodes and t < max_steps:
            action = agent.act(obs_eval, t0=(t % eval_env.max_episode_steps == 0), eval_mode=True)
            obs_eval, reward, term, trunc, _ = eval_env.step(action)
            returns += reward
            done = term | trunc
            for i in torch.where(done)[0].tolist():
                finished.append(float(returns[i]))
                returns[i] = 0
            t += 1
        return dict(eval_return=float(np.mean(finished[:args.eval_episodes])),
                    eval_steps=t * args.num_eval_envs,
                    eval_episodes=min(len(finished), args.eval_episodes))

    metrics_f = open(out / "metrics.jsonl", "a", buffering=1)
    counts = dict(ROOT_ACQUISITION=0, REGION_ACQUISITION=0, REFERENCE=0)
    steps = args.step_offset
    version = eval_steps = fragment_id = 0
    next_eval = steps if args.eval_at_start else steps + args.eval_interval
    next_checkpoint = steps + args.checkpoint_interval
    started = time.time()
    learning = False
    use_herp = args.method == "tdmpc2_herp"

    def eval_and_log():
        nonlocal eval_steps
        with torch.random.fork_rng(devices=[0]):
            row = eval_policy()
        eval_steps += row["eval_steps"]
        row.update(type="evaluation", step=steps, policy_version=version,
                   wall_seconds=time.time() - started)
        metrics_f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if run:
            run.log({"env_steps": steps, **{f"eval/{k}": v for k, v in row.items()
                                             if isinstance(v, (int, float))}})

    try:
        while steps < args.total_timesteps:
            local_steps = steps - args.step_offset
            if steps >= next_eval:
                eval_and_log()
                next_eval = steps + args.eval_interval

            pre = controller.begin_round()
            mira_active = use_herp and learning and local_steps >= mira_starts
            chosen_probs = defaultdict(list)
            if mira_active:
                sources, snapshots, probs = controller.choose_batch("herp", n=args.num_envs)
                sources[0] = 0
                snapshots[0] = None
                for rid in sources[1:].tolist():
                    chosen_probs[rid].append(float(probs[rid]))
            else:
                sources = torch.zeros(args.num_envs, dtype=torch.long)
                snapshots = [None] * args.num_envs

            obs, root_started, job_restarted = restore_or_reset(sources, snapshots)
            if use_herp:
                restarted_ids = torch.where(job_restarted)[0]
                if len(restarted_ids):
                    controller.observer.new_jobs(restarted_ids, sources[restarted_ids])
            if not controller.active:
                controller.calibrate(obs.detach().cpu())

            states_rows, next_rows, action_rows = [], [], []
            reward_rows, term_rows, trunc_rows = [], [], []
            tds = [to_td(obs)]
            max_fragment_steps = min(args.future_horizon,
                                     (args.total_timesteps - steps + args.num_envs - 1) // args.num_envs)
            for t in range(max_fragment_steps):
                states_rows.append(obs.detach().cpu())
                if mira_active:
                    elapsed = int(env.elapsed_steps()[0])
                    reference_bins[phase_for_elapsed(elapsed)].append(
                        obs[0].detach().cpu().clone())
                with torch.no_grad():
                    if initialized_agent or local_steps >= learning_starts:
                        action = agent.act(obs, t0=(t == 0), eval_mode=False)
                    else:
                        low, high = env.action_low().cpu(), env.action_high().cpu()
                        action = low + torch.rand(args.num_envs, env.action_dim) * (high - low)
                    if mira_active:
                        mu, log_std = learner.get_policy_stats(obs)
                        controller.observer.observe(obs.detach().cpu(), mu, log_std, steps)

                nxt, reward, term, trunc, info = env.step(action)
                done = term | trunc
                actual = nxt.detach().clone()
                if bool(done.any()) and "final_observation" in info:
                    finals = torch.as_tensor(info["final_observation"], device=device).float()
                    actual[done] = finals[done]
                if not controller.active:
                    controller.calibrate(actual.detach().cpu())

                next_rows.append(actual.detach().cpu())
                action_rows.append(action.detach().cpu())
                reward_rows.append(reward.detach().cpu())
                term_rows.append(term.detach().cpu())
                trunc_rows.append(trunc.detach().cpu())
                tds.append(to_td(actual, action, reward))

                if mira_active:
                    counts["REFERENCE"] += 1
                    counts["ROOT_ACQUISITION"] += int((sources[1:] == 0).sum())
                    counts["REGION_ACQUISITION"] += int((sources[1:] > 0).sum())
                else:
                    counts["ROOT_ACQUISITION"] += args.num_envs
                steps += args.num_envs
                obs = nxt
                if bool(done.any()) or steps >= args.total_timesteps:
                    if use_herp:
                        controller.observer.close_slots(torch.where(done)[0], False)
                    break

            root_ids = torch.where(sources == 0)[0]
            if len(root_ids):
                saved = env.save_state(root_ids)
                elapsed = env.elapsed_steps().cpu()
                for j,i in enumerate(root_ids.tolist()):
                    ordinary_snapshots[i] = Snapshot(
                        saved[j], obs[i].detach().cpu().clone(), steps,
                        elapsed_steps=int(elapsed[i]))
            buffer.add(torch.cat(tds, dim=1))
            states = torch.stack(states_rows, dim=1)
            next_states = torch.stack(next_rows, dim=1)
            actions = torch.stack(action_rows, dim=1)
            rewards = torch.stack(reward_rows, dim=1)
            terminated = torch.stack(term_rows, dim=1)
            truncated = torch.stack(trunc_rows, dim=1)
            fragments = []
            for i in range(args.num_envs):
                category = ("REFERENCE" if mira_active and i == 0 else
                            "REGION_ACQUISITION" if int(sources[i]) > 0 else "ROOT_ACQUISITION")
                fragments.append(controller.fragment(
                    int(sources[i]), states[i], next_states[i], actions[i], version,
                    root_started=bool(root_started[i]), rewards=rewards[i],
                    terminated=terminated[i], truncated=truncated[i],
                    restart_snapshot=snapshots[i], category=category,
                    fragment_id=fragment_id))
                fragment_id += 1

            diagnostics = {}
            if mira_active:
                diagnostics = controller.finish_round(
                    fragments, balanced_reference(), learner, version, pre)
                diagnostics.update(controller.temporal_diagnostics())
                measured_p = [float(r.p_ema) for r in controller.archive
                              if r.region_id > 0 and r.relevance_count > 0]
                diagnostics.update(
                    reference_bin_sizes=[len(bucket) for bucket in reference_bins],
                    relevance_eligible=sum(
                        int(r.region_id > 0 and
                            r.relevance_count >= herp_cfg.min_relevance_measurements and
                            r.p_ema > herp_cfg.relevance_threshold)
                        for r in controller.archive),
                    p_ema_min=(min(measured_p) if measured_p else 0.),
                    p_ema_mean=(float(np.mean(measured_p)) if measured_p else 0.),
                    p_ema_max=(max(measured_p) if measured_p else 0.),
                )

                if args.capture_video_events:
                    by_region = defaultdict(list)
                    for fragment in fragments[1:]:
                        if fragment.category == "REGION_ACQUISITION" and fragment.restart_snapshot is not None:
                            by_region[fragment.source_region_id].append(fragment)
                    root_fraction = float((sources == 0).sum()) / args.num_envs
                    for rid, event_fragments in by_region.items():
                        if len(event_fragments) < args.video_min_allocated_fragments:
                            continue
                        region = controller.archive.regions[rid]
                        metadata = dict(
                            event_id=f"step{steps:09d}_r{rid}_v{version}",
                            training_step=steps, policy_version=version, region_id=rid,
                            p_raw=float(region.p_raw), p_ema=float(region.p_ema),
                            sigma_raw=float(region.sigma_raw), q_direct=float(region.q_direct),
                            q_pred=float(region.q_pred), q_combined=float(region.q_combined),
                            allocation_prob=float(np.mean(chosen_probs[rid])),
                            allocated_fragments=len(event_fragments),
                            root_fraction=root_fraction, allocation_entropy=0.0,
                            source=dict(benchmark="dmc", env_id=args.env_id, device="cpu",
                                        sim_backend="cpu", render_backend="cpu", seed=args.seed,
                                        method=args.method, output_dir=str(out),
                                        training_run_id=None if run is None else run.id,
                                        replay_action_source="stored_exact_actions"))
                        event = build_event(metadata=metadata, fragments=event_fragments,
                                            pre_fragments=[], radius=args.video_outcome_radius)
                        if event_is_valid(event, args.video_min_allocated_fragments,
                                          args.video_min_distinct_outcomes, 0.0):
                            video_store.consider(event)

            train_metrics = {}
            local_steps = steps - args.step_offset
            if local_steps >= learning_starts:
                first_learning_round = not learning
                learning = True
                controller.active = local_steps >= mira_starts
                # A warm-started model already contains hundreds of thousands
                # of updates. Do not replay the seed-buffer catch-up burst when
                # only its optimizer/replay state had to be rebuilt.
                updates = (args.num_envs if initialized_agent and first_learning_round
                           else learning_starts if first_learning_round
                           else args.num_envs)
                for _ in range(updates):
                    train_metrics = agent.update(buffer)

            version += 1
            row = dict(type="training", step=steps, budget=counts.copy(),
                       losses=train_metrics, **diagnostics,
                       wall_seconds=time.time() - started)
            metrics_f.write(json.dumps(row) + "\n")
            if run:
                run.log({"env_steps": steps,
                         **{f"loss/{k}": v for k, v in train_metrics.items()},
                         **{f"herp/{k}": v for k, v in diagnostics.items()},
                         **{f"budget/{k}": v for k, v in counts.items()},
                         "herp/num_regions": len(controller.archive)})

            if steps >= next_checkpoint:
                agent.save(out / f"checkpoint_{steps}.pt")
                next_checkpoint += args.checkpoint_interval

        eval_and_log()
        agent.save(out / "checkpoint_final.pt")
        summary = dict(status="completed", method=args.method, env_id=args.env_id,
                       seed=args.seed, training_steps=steps, budget=counts,
                       eval_steps=eval_steps, wall_seconds=time.time() - started)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

        if args.capture_video_events:
            from scripts.render_mira_video_events import render_ranked_events
            render_ranked_events(video_dir, top_k=args.video_upload_top_k, fps=30,
                                 pre_seconds=0, post_seconds=20, max_seconds=40,
                                 resolution=720, replay_tolerance=1e-3,
                                 wandb_run=run, wandb_module=wandb if run else None,
                                 wandb_prefix="mira_tdmpc2_events")
    finally:
        metrics_f.close()
        env.close()
        eval_env.close()
        if run:
            run.finish()

    print(f"Done: {args.method} {args.env_id} s{args.seed} in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
