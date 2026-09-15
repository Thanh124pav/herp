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
from collections import defaultdict
from pathlib import Path
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
from herp.config import HERPV3Config
from herp.envs.maniskill import ManiSkillAdapter
from herp.provenance import save_provenance
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
        lr=3e-4,
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
    p.add_argument("--env-id", default="PickCube-v1")
    p.add_argument("--method", choices=["tdmpc2", "tdmpc2_herp"], default="tdmpc2_herp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-eval-envs", type=int, default=8)
    p.add_argument("--control-mode", default="pd_ee_delta_pose")
    p.add_argument("--reward-mode", default="normalized_dense")
    p.add_argument("--root-floor", type=float, default=0.15)
    p.add_argument("--eval-interval", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--checkpoint-interval", type=int, default=500_000)
    p.add_argument("--future-horizon", type=int, default=32)
    p.add_argument("--sigma-mode", default="shrinkage")
    p.add_argument("--wandb-mode", default="online")
    p.add_argument("--wandb-project", default="herp-framework")
    p.add_argument("--wandb-group", default="rq1-mbrl")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--wandb-tags", default="")
    p.add_argument("--gamma", type=float, default=0.8)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = ManiSkillAdapter(
        args.env_id, control_mode=args.control_mode,
        obs_mode="state", reward_mode=args.reward_mode,
        sim_backend="physx_cuda", render_backend="gpu",
        device=device, ignore_terminations=True,
    ).make(args.num_envs, args.seed)

    eval_env = ManiSkillAdapter(
        args.env_id, control_mode=args.control_mode,
        obs_mode="state", reward_mode=args.reward_mode,
        sim_backend="physx_cuda", render_backend="gpu",
        device=device, ignore_terminations=True,
    ).make(args.num_eval_envs, args.seed + 10000)

    obs_dim = env.obs_dim
    action_dim = env.action_dim
    episode_length = getattr(env.env.unwrapped, "max_episode_steps", 50) or 50

    cfg = make_tdmpc2_cfg(obs_dim, action_dim, episode_length, args)
    agent = TDMPC2Agent(cfg)
    buffer = Buffer(cfg)

    herp_cfg = HERPV3Config(
        future_horizon=args.future_horizon,
        min_common_steps=max(2, args.future_horizon // 4),
        root_floor=args.root_floor,
    )
    controller = AllocationController(env, herp_cfg, args.seed, args.sigma_mode)
    controller.observer = VectorPartitionObserver(
        env, controller.archive, controller.normalizer, herp_cfg)

    learner_wrapper = TDMPC2Learner(agent, obs_dim, torch.device(device))

    save_provenance(out, ROOT)

    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run_name = args.wandb_run_name or f"{args.method}-{args.env_id}-s{args.seed}"
        run = wandb.init(
            project=args.wandb_project, group=args.wandb_group,
            name=run_name,
            tags=[t for t in args.wandb_tags.split(",") if t],
            config=vars(args), mode=args.wandb_mode,
        )

    metrics_f = open(out / "metrics.jsonl", "a", buffering=1)
    counts = dict(ROOT_ACQUISITION=0, REGION_ACQUISITION=0, REFERENCE=0)
    steps = version = eval_steps = 0
    next_eval = 0
    started = time.time()
    learning = False
    use_herp = args.method == "tdmpc2_herp"

    from tensordict.tensordict import TensorDict

    def to_td(obs_t, num_envs, action=None, reward=None):
        obs_t = obs_t.detach().unsqueeze(1).cpu()
        if action is None:
            action = torch.full((num_envs, cfg.action_dim), float("nan"))
        else:
            action = action.detach().cpu()
        if reward is None:
            reward = torch.full((num_envs,), float("nan"))
        else:
            reward = reward.detach().cpu()
        return TensorDict(
            dict(obs=obs_t, action=action.unsqueeze(1), reward=reward.unsqueeze(1)),
            batch_size=(num_envs, 1),
        )

    def _restore_or_reset(sources, snapshots, obs_curr):
        root = torch.where(sources == 0)[0]
        local = torch.where(sources > 0)[0]
        if obs_curr is None:
            obs_curr, _ = env.reset()
        if len(root):
            current, _ = env.reset_indices(root)
            obs_curr[root.to(device)] = current[root.to(device)]
        if len(local):
            picked = [snapshots[int(i)] for i in local]
            restored = env.restore_state(local, [s.env_state for s in picked])
            obs_curr[local.to(device)] = restored
        return obs_curr

    def eval_and_log():
        nonlocal eval_steps
        with torch.random.fork_rng(devices=[0]):
            row = evaluate(eval_env, agent, args.eval_episodes,
                          args.seed + 10000, torch.device(device), args.num_eval_envs)
        eval_steps += row["eval_steps"]
        row.update(type="evaluation", step=steps, wall_seconds=time.time() - started)
        metrics_f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if run:
            run.log({"env_steps": steps, **{f"eval/{k}": v for k, v in row.items()
                                             if isinstance(v, (int, float))}})

    sources = torch.zeros(args.num_envs, dtype=torch.long)
    snapshots = [None] * args.num_envs
    obs = None
    tds_list = []
    ep_done = True

    try:
        while steps < args.total_timesteps:
            if steps >= next_eval:
                eval_and_log()
                next_eval = steps + args.eval_interval

            if ep_done:
                if tds_list and len(tds_list) > 1:
                    tds = torch.cat(tds_list, dim=1)
                    train_metrics = {}
                    buffer.add(tds)

                    if steps >= cfg.seed_steps:
                        learning = True
                        controller.active = True
                        if not hasattr(main, "_seed_pretrained"):
                            main._seed_pretrained = True
                            num_updates = int(cfg.seed_steps / cfg.steps_per_update)
                            print(f"Pretraining agent on seed data ({num_updates} updates)...", flush=True)
                        else:
                            num_updates = max(1, int(args.num_envs / cfg.steps_per_update))
                        for _ in range(num_updates):
                            train_metrics = agent.update(buffer)

                        version += 1
                        row = dict(type="training", step=steps, budget=counts.copy(),
                                   losses=train_metrics,
                                   num_regions=len(controller.archive),
                                   wall_seconds=time.time() - started)
                        metrics_f.write(json.dumps(row) + "\n")
                        if run:
                            run.log({"env_steps": steps,
                                     **{f"loss/{k}": v for k, v in train_metrics.items()},
                                     **{f"budget/{k}": v for k, v in counts.items()},
                                     "herp/num_regions": len(controller.archive)})

                if use_herp and learning:
                    rids, snaps, _ = controller.choose_batch(
                        "herp", n=args.num_envs, warmup=not learning, ordinary=not learning)
                    sources = rids
                    snapshots = snaps
                    obs = _restore_or_reset(sources, snapshots, obs)
                    controller.observer.new_jobs(torch.arange(args.num_envs), sources)
                else:
                    obs, _ = env.reset()
                    sources[:] = 0
                    if use_herp:
                        controller.observer.new_jobs(torch.arange(args.num_envs), sources)

                if not controller.active:
                    controller.calibrate(obs.detach().cpu())

                tds_list = [to_td(obs, args.num_envs)]
                ep_done = False

                counts["ROOT_ACQUISITION"] += int((sources == 0).sum())
                counts["REGION_ACQUISITION"] += int((sources > 0).sum())

            with torch.no_grad():
                if steps >= cfg.seed_steps:
                    action = agent.act(obs, t0=(len(tds_list) == 1), eval_mode=False)
                else:
                    action = torch.from_numpy(env.env.action_space.sample()).to(device)

                if use_herp and learning:
                    mu, log_std = learner_wrapper.get_policy_stats(obs)
                    controller.observer.observe(obs.detach().cpu(), mu, log_std, steps)

            nxt, reward, term, trunc, info = env.step(action)
            done = term | trunc

            if bool(done[0]):
                if isinstance(info, dict) and "final_observation" in info:
                    final_obs = info["final_observation"]
                else:
                    final_obs = nxt
                tds_list.append(to_td(final_obs, args.num_envs, action, reward))

                if use_herp and learning:
                    controller.observer.close_slots(torch.arange(args.num_envs), False)
                ep_done = True
            else:
                tds_list.append(to_td(nxt, args.num_envs, action, reward))

            obs = nxt
            steps += args.num_envs

        eval_and_log()

        agent.save(out / "checkpoint_final.pt")
        summary = dict(
            status="completed", method=args.method, env_id=args.env_id,
            seed=args.seed, training_steps=steps,
            total_timesteps=args.total_timesteps,
            budget=counts, eval_steps=eval_steps,
            wall_seconds=time.time() - started,
        )
        last_eval = [json.loads(l) for l in (out / "metrics.jsonl").read_text().strip().split("\n")
                     if '"type": "evaluation"' in l]
        if last_eval:
            summary["final_evaluation"] = last_eval[-1]
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

    finally:
        metrics_f.close()
        env.close()
        eval_env.close()
        if run:
            run.finish()

    print(f"Done: {args.method} {args.env_id} s{args.seed} in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
