"""Milestone 0 -- single SAC on one task (PLAN.md section 16).

    python scripts/train_single.py --steps 20000

Verifies the RL backbone learns (Gate A) before any population/routing work.
"""

from __future__ import annotations

import argparse

import numpy as np

from experience_routing.envs.synthetic_reacher import SyntheticPickPlace
from experience_routing.rl.replay_buffer import ReplayBuffer
from experience_routing.rl.sac import SACAgent, SACConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--warmup", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = SyntheticPickPlace()
    agent = SACAgent(SACConfig(env.spec.obs_dim, env.spec.action_dim, seed=args.seed))
    buf = ReplayBuffer(200_000, env.spec.obs_dim, env.spec.action_dim, args.seed)

    def evaluate(n=20):
        succ, rets = 0, []
        for ep in range(n):
            o = env.reset(seed=10_000 + ep)
            done = trunc = False
            ret = 0.0
            info = {}
            while not (done or trunc):
                o, r, done, trunc, info = env.step(agent.act(o, deterministic=True))
                ret += r
            succ += int(info.get("success", False))
            rets.append(ret)
        return succ / n, float(np.mean(rets))

    rng = np.random.default_rng(args.seed)
    o = env.reset(seed=args.seed)
    total = 0
    ep = 0
    while total < args.steps:
        a = rng.uniform(-1, 1, env.spec.action_dim) if total < args.warmup else agent.act(o)
        no, r, done, trunc, _ = env.step(a)
        buf.add(o, a, r, no, done)
        o = no
        total += 1
        if len(buf) >= args.warmup:
            agent.update(buf.sample(128))
        if done or trunc:
            o = env.reset(seed=args.seed + ep)
            ep += 1
        if total % 4000 == 0:
            sr, mr = evaluate()
            print(f"steps={total} success={sr:.2f} return={mr:.1f}")

    sr, mr = evaluate()
    print(f"FINAL steps={total} success={sr:.2f} return={mr:.1f}")


if __name__ == "__main__":
    main()
