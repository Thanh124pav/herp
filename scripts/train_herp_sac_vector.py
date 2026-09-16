"""Vectorized HERP-SAC on ManiSkill physx_cuda.

Same SAC update rule as the serial `train_herp_sac.py` (unchanged from the
ManiSkill upstream SAC), but N slots are stepped in parallel with HERP
choosing which region each slot starts from at the beginning of every
acquisition window and after every episode-end within the window.

Serial `train_herp_sac.py` is kept as a num_envs=1 backup for debugging
and for machines without GPU sim; new experiments should use this
vector runner.

Key contract, per HERP_UPDATED_POSITIONING_BASELINES_EXPERIMENTS_v2.md §0.5:

    HERP allocation
        -> restored-region interactions
        -> SAC replay buffer            (fairness: uniform sampling)
        -> native SAC updates           (unmodified)

Everything the acquisition consumes counts against the training-step
budget, including reference and warmup transitions. Evaluation
transitions are reported separately.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")

import numpy as np
import torch
import tyro

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1] / "src")]

from herp.allocation_controller import AllocationController
from herp.archive import Snapshot
from herp.config import HERPV3Config
from herp.envs.maniskill import ManiSkillAdapter
from herp.learners.sac import SACLearner
from herp.provenance import save_provenance
from herp.vector_acquisition import VectorPartitionObserver


@dataclass
class Args:
    # Task / seed
    env_id: str = "PickCube-v1"
    seed: int = 0
    method: str = "sac_herp"  # 'sac' | 'sac_herp' | 'sac_herp_p' | 'sac_herp_sigma'
    phase: str = "pilot"
    output_dir: str = "outputs/sac_herp_vec"
    resume_from: str = ""

    # Env parallelism
    num_envs: int = 16
    num_eval_envs: int = 8
    sim_backend: str = "physx_cuda"
    control_mode: str = "pd_ee_delta_pose"
    reward_mode: str = "normalized_dense"
    obs_mode: str = "state"
    partial_reset: bool = False
    cuda: bool = True

    # SAC hyperparameters (upstream ManiSkill SAC defaults)
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    gamma: float = 0.8
    tau: float = 0.01
    batch_size: int = 1024
    learning_starts: int = 4_000
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 1
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    training_freq: int = 64
    utd: float = 0.5
    bootstrap_at_done: str = "always"  # 'always' | 'never' | 'truncated'

    # HERP knobs
    future_horizon: int = 32
    sigma_mode: str = "shrinkage"  # 'direct' | 'predictor' | 'shrinkage'
    sigma_kappa: float = 8.
    max_regions: int = 64

    # Budget + eval
    total_timesteps: int = 1_000_000
    eval_freq: int = 50_000
    eval_episodes: int = 10
    checkpoint_interval: int = 100_000

    # Logging
    wandb_mode: str = "online"  # 'disabled' | 'offline' | 'online'
    wandb_project_name: str = "herp-framework"
    wandb_entity: Optional[str] = None
    wandb_group: str = "sac"
    wandb_run_name: str = ""


def _restore_or_reset(env, sources, snapshots, obs, device):
    """Ordinary-reset root slots; restore snapshots for the rest.

    sources: 1D long tensor of region IDs, length N (= num_envs).
    snapshots: list of Snapshot | None, len == len(sources).
    obs: current observation tensor (may be None on first call). Returned
    updated in-place-like via clone().
    """
    root = torch.where(sources == 0)[0]
    local = torch.where(sources > 0)[0]
    if obs is None:
        obs, _ = env.reset()
    if len(root):
        current, _ = env.reset_indices(root)
        obs[root.to(device)] = current[root.to(device)]
    if len(local):
        picked = [snapshots[int(i)] for i in local]
        restored = env.restore_state(local, [s.env_state for s in picked])
        err = float((restored.cpu() - torch.stack([s.obs for s in picked])).abs().max())
        if err > 1e-4:
            raise RuntimeError(f"vector snapshot restore mismatch {err}")
        obs[local.to(device)] = restored
    return obs


@torch.no_grad()
def _evaluate(env, actor, episodes, seed, device):
    """Deterministic eval mirroring scripts/train_v3.py's evaluate()."""
    obs, _ = env.reset(seed=seed)
    n = env.num_envs
    ep_returns, ep_success_once, ep_success_final = [], [], []
    ret = torch.zeros(n, device=device)
    per_slot_success = torch.zeros(n, device=device)
    steps = 0
    max_steps = max(50, getattr(env, "max_episode_steps", 200)) * max(1, (episodes + n - 1) // n) * 2
    while len(ep_returns) < episodes and steps < max_steps:
        action = actor.get_eval_action(obs)
        obs, r, term, trunc, info = env.step(action)
        ret = ret + r.to(device).float()
        cur = env.success_from_info(info).float().to(device)
        done = term.to(device) | trunc.to(device)
        fi = info.get("final_info") if isinstance(info, dict) else None
        end = env.success_from_info(fi).float().to(device) if fi is not None else cur
        cur = torch.where(done, end, cur)
        per_slot_success = torch.maximum(per_slot_success, cur)
        if bool(done.any()):
            for i in torch.where(done)[0].tolist():
                ep_returns.append(float(ret[i]))
                ep_success_once.append(max(float(per_slot_success[i]), float(end[i])))
                ep_success_final.append(float(end[i]))
                ret[i] = 0.
                per_slot_success[i] = 0.
        steps += 1
    ep_returns = ep_returns[:episodes]
    ep_success_once = ep_success_once[:episodes]
    ep_success_final = ep_success_final[:episodes]
    return dict(eval_return=float(np.mean(ep_returns)) if ep_returns else 0.,
                eval_success=float(np.mean(ep_success_once)) if ep_success_once else 0.,
                eval_success_final=float(np.mean(ep_success_final)) if ep_success_final else 0.,
                success_once=float(np.mean(ep_success_once)) if ep_success_once else 0.,
                success_at_end=float(np.mean(ep_success_final)) if ep_success_final else 0.,
                eval_steps=steps * n, eval_episodes=len(ep_returns))


def main():
    args = tyro.cli(Args)
    if args.method not in ("sac", "sac_herp", "sac_herp_p", "sac_herp_sigma"):
        raise ValueError(f"Unsupported method {args.method!r}")
    if args.num_envs < 1 or args.total_timesteps < 1 or args.training_freq < 1 or args.eval_freq < 1:
        raise ValueError("Positive num_envs / budget / freq required")
    if args.total_timesteps % args.num_envs:
        raise ValueError(f"total_timesteps ({args.total_timesteps}) must be divisible by num_envs ({args.num_envs})")
    if args.sigma_mode not in ("direct", "predictor", "shrinkage"):
        raise ValueError("sigma_mode must be direct|predictor|shrinkage")
    if args.wandb_mode not in ("disabled", "offline", "online"):
        raise ValueError("wandb_mode must be disabled|offline|online")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.jsonl").exists() and not args.resume_from:
        raise FileExistsError("Use a new output_dir or explicit --resume-from")
    save_provenance(out, Path(__file__).resolve().parents[1])
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    env = ManiSkillAdapter(args.env_id, control_mode=args.control_mode,
                           obs_mode=args.obs_mode, reward_mode=args.reward_mode,
                           sim_backend=args.sim_backend, render_backend="cpu",
                           device=device, ignore_terminations=not args.partial_reset).make(args.num_envs, args.seed)
    eval_env = ManiSkillAdapter(args.env_id, control_mode=args.control_mode,
                                obs_mode=args.obs_mode, reward_mode=args.reward_mode,
                                sim_backend=args.sim_backend, render_backend="cpu",
                                device=device, ignore_terminations=True).make(args.num_eval_envs, args.seed + 10000)

    # SACLearner + region-labelled replay (uniform sampling per v2.md §0.5).
    learner = SACLearner(env.env, args, device)
    cfg = HERPV3Config(future_horizon=args.future_horizon,
                       min_common_steps=max(2, args.future_horizon // 4),
                       max_regions=args.max_regions,
                       sigma_predictor_kappa=args.sigma_kappa)
    controller = AllocationController(env, cfg, args.seed, args.sigma_mode)
    controller.observer = VectorPartitionObserver(
        env, controller.archive, controller.normalizer, cfg)

    counts = dict(ROOT_ACQUISITION=0, REGION_ACQUISITION=0, REFERENCE=0)
    steps = version = eval_steps = 0
    next_eval = 0
    next_checkpoint = args.checkpoint_interval
    started = time.time()
    learning = False

    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()
        learner_path = Path(str(resume_path) + ".learner.pt")
        if not resume_path.is_file() or not learner_path.is_file():
            raise FileNotFoundError(
                f"Resume requires {resume_path} and {learner_path}")
        # These are checkpoints produced by save_checkpoint() below. Keep the
        # restricted unpickler enabled and allow only the repository-owned
        # controller graph plus the small set of standard containers it uses.
        from herp.archive import Region, RegionArchive
        from herp.chain_features import RunningFeatureNormalizer
        from herp.chain_partition import ChainRegionizer
        from herp.envs.maniskill import ManiSkillSnapshot
        from herp.region_graph import RegionGraph
        from herp.sigma_predictor import LinearVariancePredictor
        safe = [
            AllocationController, VectorPartitionObserver, HERPV3Config,
            Region, RegionArchive, Snapshot, RunningFeatureNormalizer,
            ChainRegionizer, RegionGraph, LinearVariancePredictor,
            ManiSkillSnapshot, defaultdict, deque, partial, torch.Generator,
            np.ndarray, np.dtype, np._core.multiarray._reconstruct,
            type(np.dtype(np.float32)), type(np.dtype(np.float64)),
            type(np.dtype(np.int64)), type(np.dtype(np.uint32)),
            type(np.dtype(object)),
        ]
        with torch.serialization.safe_globals(safe):
            state = torch.load(resume_path, map_location="cpu", weights_only=True)
        saved_args = state.get("args", {})
        for key in ("env_id", "method", "seed", "num_envs"):
            if saved_args.get(key) != getattr(args, key):
                raise ValueError(
                    f"Resume mismatch for {key}: checkpoint={saved_args.get(key)!r}, "
                    f"requested={getattr(args, key)!r}")
        learner.load(learner_path)
        controller = state["controller"]
        controller.adapter = env
        controller.observer.adapter = env
        steps = int(state["steps"])
        version = int(state["version"])
        eval_steps = int(state["eval_steps"])
        learning = bool(state["learning"])
        counts = dict(state["counts"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if args.total_timesteps <= steps:
            raise ValueError(
                f"total_timesteps ({args.total_timesteps}) must exceed resumed step {steps}")
        next_eval = ((steps // args.eval_freq) + 1) * args.eval_freq
        next_checkpoint = ((steps // args.checkpoint_interval) + 1) * args.checkpoint_interval
        print(json.dumps(dict(type="resume", checkpoint=str(resume_path),
                              step=steps, target=args.total_timesteps)), flush=True)

    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run_name = args.wandb_run_name or f"{args.method}-{args.env_id}-s{args.seed}"
        run = wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                         group=args.wandb_group, name=run_name, config=asdict(args),
                         dir=str(out), mode=args.wandb_mode)
        run.define_metric("env_steps")
        run.define_metric("*", step_metric="env_steps")

    (out / "config.json").write_text(json.dumps({**asdict(args), "herp": asdict(cfg),
        "acquisition": "vectorized_parallel_snapshot_restore",
        "replay": "uniform_region_labelled"}, indent=2))
    metrics_file = (out / "metrics.jsonl").open("a", buffering=1)
    regions_file = (out / "regions.jsonl").open("a", buffering=1)

    def save_checkpoint(name: str):
        learner.save(out / f"{name}.learner.pt")
        controller.adapter = None; controller.observer.adapter = None
        try:
            state = dict(args=asdict(args), controller=controller, steps=steps, version=version,
                         eval_steps=eval_steps, learning=learning, counts=counts,
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         wandb_id=run.id if run else None)
            path = out / name
            torch.save(state, path.with_suffix(".tmp"))
            path.with_suffix(".tmp").replace(path)
        finally:
            controller.adapter = env; controller.observer.adapter = env

    def eval_and_log():
        nonlocal eval_steps
        with torch.random.fork_rng(devices=[0] if device == "cuda" else []):
            row = _evaluate(eval_env, learner.actor, args.eval_episodes, args.seed + 10000, device)
        eval_steps += row["eval_steps"]
        row.update(type="evaluation", step=steps, wall_seconds=time.time() - started)
        metrics_file.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if not k.startswith("episode_")}), flush=True)
        if run:
            run.log({"env_steps": steps, **{f"eval/{k}": v for k, v in row.items()
                                             if isinstance(v, (int, float))}})

    obs = None
    # Every-slot region tracker; refreshed on window start + per-slot done.
    sources = torch.zeros(args.num_envs, dtype=torch.long)
    snapshots: list = [None] * args.num_envs

    try:
        while steps < args.total_timesteps:
            if steps >= next_eval:
                eval_and_log()
                next_eval = steps + args.eval_freq

            use_herp = args.method != "sac" and learning
            remaining = min(args.training_freq, args.total_timesteps - steps)
            # We step in windows of `training_freq` transitions per env slot,
            # or less if the residual budget doesn't cover a full window.
            window_steps = max(1, remaining // args.num_envs)

            # Sample per-slot sources for the round.
            rids, snaps, probs = controller.choose_batch(
                args.method.replace("sac_", ""), n=args.num_envs,
                warmup=not use_herp, ordinary=not use_herp,
            )
            sources = rids
            snapshots = snaps
            obs = _restore_or_reset(env, sources, snapshots, obs, torch.device(device))

            # HERP partition observer starts a fresh chain per slot on the
            # restored/reset transition.
            controller.observer.new_jobs(torch.arange(args.num_envs), sources)

            start_steps = steps
            for _ in range(window_steps):
                # Calibrate the normalizer on early raw obs (matches serial).
                if not controller.active:
                    controller.calibrate(obs.detach().cpu())

                with torch.no_grad():
                    if not learning:
                        action = 2 * torch.rand(args.num_envs, env.action_dim, device=device) - 1
                    else:
                        action = learner.act(obs)
                    if use_herp:
                        mean, logstd = learner.actor(obs)
                        controller.observer.observe(obs, mean, logstd, steps)

                nxt, reward, term, trunc, info = env.step(action)
                done = term | trunc
                actual_next = nxt.clone()
                if bool(done.any()) and isinstance(info, dict) and "final_observation" in info:
                    actual_next[done] = info["final_observation"][done]
                if args.bootstrap_at_done == "never":
                    stop = done
                elif args.bootstrap_at_done == "always":
                    stop = torch.zeros_like(term)
                else:
                    stop = term
                # Push all N slots at once, with per-slot region labels.
                learner.replay.add(obs=obs, next_obs=actual_next, action=action,
                                   reward=reward, done=stop, region=sources)

                counts["REFERENCE"] += 0  # vector runner: no dedicated reference slots
                counts["ROOT_ACQUISITION"] += int((sources == 0).sum())
                counts["REGION_ACQUISITION"] += int((sources > 0).sum())

                obs = nxt
                steps += args.num_envs

                # On any slot's done, HERP picks a fresh source and restores.
                ended = torch.where(done.cpu())[0]
                if len(ended):
                    controller.observer.close_slots(ended, False)
                    if use_herp:
                        new_rids, new_snaps, _ = controller.choose_batch(
                            args.method.replace("sac_", ""), n=len(ended),
                        )
                        sources[ended] = new_rids
                        for j, i in enumerate(ended.tolist()):
                            snapshots[i] = new_snaps[j]
                        obs = _restore_or_reset(env, sources[ended], [snapshots[int(i)] for i in ended],
                                                obs, torch.device(device))
                        controller.observer.new_jobs(ended, sources[ended])
                    else:
                        # Ordinary reset already happened via the vec env; mark
                        # the observer so it starts a fresh chain from root.
                        sources[ended] = 0
                        controller.observer.new_jobs(ended, sources[ended])

            # SAC updates for the transitions collected this window.
            losses = {}
            if steps >= args.learning_starts:
                learning = True
                controller.active = True
                grad_steps = max(1, int((steps - start_steps) * args.utd))
                for _ in range(grad_steps):
                    losses = learner.update()
            version += 1
            assert sum(counts.values()) == steps

            row = dict(type="training", step=steps, budget=counts.copy(), losses=losses,
                       num_regions=len(controller.archive),
                       predictor_labels=len(controller.predictor.y),
                       wall_seconds=time.time() - started)
            metrics_file.write(json.dumps(row) + "\n")
            if run:
                run.log({"env_steps": steps,
                         **{f"loss/{k}": v for k, v in losses.items()},
                         **{f"budget/{k}": v for k, v in counts.items()},
                         "herp/num_regions": len(controller.archive),
                         "herp/predictor_labels": len(controller.predictor.y)})
            if version % 32 == 0:
                occupancy = learner.replay.occupancy()
                for r in controller.archive:
                    regions_file.write(json.dumps(dict(
                        step=steps, region=r.region_id,
                        new_transitions=learner.replay.acquired.get(r.region_id, 0),
                        buffer_occupancy=occupancy.get(r.region_id, 0),
                        replay_samples=learner.replay.sampled.get(r.region_id, 0),
                    )) + "\n")

            if steps >= next_checkpoint:
                save_checkpoint(f"checkpoint_{steps}.pt")
                next_checkpoint += args.checkpoint_interval

        eval_and_log()
        save_checkpoint("checkpoint_final.pt")
        (out / "summary.json").write_text(json.dumps(dict(
            status="completed", method=args.method, env_id=args.env_id, seed=args.seed,
            env_steps=steps, eval_env_steps=eval_steps, wall_seconds=time.time() - started,
        )))
    finally:
        metrics_file.close(); regions_file.close()
        env.close(); eval_env.close()
        if run:
            run.summary["final/env_steps"] = steps
            run.summary["final/wall_seconds"] = time.time() - started
            run.finish()


if __name__ == "__main__":
    main()
