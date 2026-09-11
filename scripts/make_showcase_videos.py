#!/usr/bin/env python3
"""Load the best-eval-success checkpoint of selected (task, method) cells, roll
out the deterministic policy with rendering, and upload the videos to W&B.

For presentation: showcases HERP solving the harder ManiSkill tasks. ManiSkill
only (physx_cuda render verified on this box); Meta-World uses MuJoCo rendering
which is a separate path.
"""
import csv, glob, os, sys
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

sys.path.insert(0, "/workspace/herp")
from train import Agent  # noqa: E402

MATRIX = "/workspace/herp/outputs/full_matrix_1seed/maniskill"
OUT = "/workspace/herp/outputs/showcase_videos"
DEVICE = "cuda"
EPISODES = 3  # per cell

# Full comparison: every method on every learnable task (StackCube omitted —
# all methods 0 there). Includes failing methods on purpose, to contrast where
# HERP succeeds and baselines fail.
_TASKS = ["PushCube-v1", "PickCube-v1", "LiftPegUpright-v1",
          "PlaceSphere-v1", "PokeCube-v1"]
_METHODS = ["ppo", "rnd", "disagreement", "herp_sigma", "herp_p", "herp"]
TARGETS = [(t, m) for t in _TASKS for m in _METHODS]


def best_checkpoint(cell_dir):
    """Return (checkpoint_path, step, success) for the max-eval-success row."""
    csvs = glob.glob(f"{cell_dir}/*/metrics.csv")
    if not csvs:
        return None
    best = None
    for row in csv.DictReader(open(csvs[0])):
        v = row.get("eval_success", "")
        if v in ("", None):
            continue
        try:
            s = float(v); step = int(row["global_env_steps"])
        except ValueError:
            continue
        if best is None or s > best[2]:
            best = (step, None, s)
    if best is None:
        return None
    step, _, succ = best
    cks = glob.glob(f"{cell_dir}/*/checkpoint_*.pt")
    if not cks:
        return None
    # exact match on step, else numerically nearest checkpoint
    exact = [c for c in cks if c.endswith(f"checkpoint_{step}.pt")]
    path = exact[0] if exact else min(
        cks, key=lambda c: abs(int(c.rsplit("_", 1)[1][:-3]) - step))
    return path, step, succ


def rollout(task, method, run):
    cell = f"{MATRIX}/{task}_{method}_s0"
    got = best_checkpoint(cell)
    if got is None:
        print(f"  [{task}/{method}] no checkpoint/metrics — skip"); return
    ckpt_path, step, succ = got
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    a = ck["args"]
    vid_dir = f"{OUT}/{task}_{method}"
    os.makedirs(vid_dir, exist_ok=True)

    env = gym.make(task, num_envs=1, obs_mode="state", render_mode="rgb_array",
                   sim_backend="physx_cuda", control_mode=a.get("control_mode", "pd_joint_delta_pos"),
                   reward_mode=a.get("reward_mode", "normalized_dense"), reconfiguration_freq=1)
    max_ep = env.unwrapped.max_episode_steps if hasattr(env.unwrapped, "max_episode_steps") else 50
    env = RecordEpisode(env, output_dir=vid_dir, save_trajectory=False,
                        max_steps_per_video=max_ep, video_fps=30)
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)

    obs_dim = int(np.prod(env.single_observation_space.shape))
    act_dim = int(np.prod(env.single_action_space.shape))
    low = torch.as_tensor(env.single_action_space.low, device=DEVICE)
    high = torch.as_tensor(env.single_action_space.high, device=DEVICE)
    agent = Agent(obs_dim, act_dim, a.get("hidden", 256)).to(DEVICE)
    agent.load_state_dict(ck["agent"]); agent.eval()

    successes = 0
    for _ in range(EPISODES):
        obs, _ = env.reset()
        for _ in range(max_ep):
            with torch.no_grad():
                act = agent.act(obs.to(DEVICE).float(), deterministic=True).clamp(low, high)
            obs, _rew, term, trunc, info = env.step(act)
            if "success" in info:
                successes += int(info["success"].any().item())
    env.close()

    mp4s = sorted(glob.glob(f"{vid_dir}/*.mp4"))
    print(f"  [{task}/{method}] ckpt step={step} eval={succ:.2f} -> {len(mp4s)} videos")
    for i, mp4 in enumerate(mp4s):
        run.log({f"video/{task}_{method}": wandb.Video(mp4, fps=30, format="mp4"),
                 f"meta/{task}_{method}_eval_success": succ,
                 f"meta/{task}_{method}_ckpt_step": step})


if __name__ == "__main__":
    import wandb
    os.makedirs(OUT, exist_ok=True)
    run = wandb.init(project="herp", group="showcase-videos",
                     name="showcase-all-methods", tags=["videos", "showcase", "all-methods"],
                     job_type="eval")
    for task, method in TARGETS:
        try:
            rollout(task, method, run)
        except Exception as e:
            print(f"  [{task}/{method}] FAILED: {type(e).__name__}: {str(e)[:200]}")
    run.finish()
    print("done")
