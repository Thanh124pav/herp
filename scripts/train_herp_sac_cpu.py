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
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1] / "src")]

from herp.config import HERPV3Config
from herp.envs import make_adapter
from herp.learners.sac import SACLearner
from herp.allocation_controller import AllocationController
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
    p.add_argument("--sigma-mode", default="shrinkage")
    p.add_argument("--sigma-kappa", type=float, default=8.0)
    # Env-specific
    p.add_argument("--control-mode", default="pd_ee_delta_pose")
    p.add_argument("--reward-mode", default="dense")
    p.add_argument("--training-freq", type=int, default=256)
    args = p.parse_args()

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

    shim = EnvShim(env)
    sac_args = SACArgs(gamma=args.gamma, tau=args.tau, batch_size=args.batch_size,
                       learning_starts=args.learning_starts)
    learner = SACLearner(shim, sac_args, device)

    cfg = HERPV3Config(
        future_horizon=args.future_horizon,
        min_common_steps=max(2, args.future_horizon // 4),
        root_floor=args.root_floor,
        sigma_predictor_kappa=args.sigma_kappa,
    )
    controller = AllocationController(env, cfg, args.seed, args.sigma_mode)

    save_provenance(out, Path(__file__).resolve().parents[1])

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
    next_eval = 0
    obs = None
    learning = False
    done = True
    started = time.time()

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
            reference = []
            pre = controller.begin_round()
            remaining = min(args.training_freq, args.total_timesteps - steps)

            while remaining:
                use_herp = args.method != "sac" and learning
                reference_job = use_herp and not fragments

                if use_herp:
                    if reference_job:
                        rid, snapshot = 0, None
                    else:
                        rid, snapshot, _ = controller.choose(args.method.replace("sac_", ""))
                    if rid == 0 or snapshot is None:
                        obs = env.reset()[0]
                    else:
                        obs = env.restore_state(torch.tensor([0]), [snapshot.env_state])
                    root_started = (rid == 0)
                    controller.observer.new_fragment(rid, root_started)
                else:
                    rid = 0
                    root_started = obs is None or done
                    if root_started:
                        obs = env.reset()[0]

                states = []
                next_states = []
                actions_list = []

                for _ in range(min(args.future_horizon, remaining)):
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

                if reference_job:
                    reference.extend(states)

                fragments.append(controller.fragment(
                    rid, torch.cat(states), torch.cat(next_states),
                    torch.cat(actions_list), version, root_started))

            # HERP round diagnostics
            diagnostics = {}
            if args.method != "sac" and learning:
                diagnostics = controller.finish_round(
                    fragments,
                    torch.cat(reference) if reference else torch.empty(0, env.obs_dim),
                    learner, version, pre)

            # SAC updates
            losses = {}
            if steps >= args.learning_starts:
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

    finally:
        metrics_f.close()
        env.close()
        ev_env.close()
        if run:
            run.finish()

    print(f"Done: {args.method} {args.env_id} s{args.seed} in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
