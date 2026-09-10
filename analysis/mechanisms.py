"""Mechanism validation (IMPLEMENTATION.md §4.12 σ, §4.13 controlled PPO-delta).

Loads a checkpoint from ``train.py`` and runs, on the same adapter:

* σ mechanism: cheap K=4 vs. oracle K=64 probe correlation over archived
  regions, Spearman ρ with bootstrap 95% CI.
* p_v mechanism: controlled PPO-delta.
    θ_A = PPOUpdate(θ, D_base)
    θ_B = PPOUpdate(θ, D_base ∪ D_v)
    Δ_v = J_ref(θ_B) − J_ref(θ_A)
  Both clones share optimizer state and PPO hyperparameters, so Δ_v isolates
  the marginal contribution of region v's data. The one-step SGD proxy that
  produced the sign flip in the v2 pilot has been removed entirely.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.stats import spearmanr

from herp.gradient_signature import empirical_fisher_diagonal, policy_gradient_signature
from herp.probe import probe_rollout
from herp.reference import collect_reference
from train import Agent, Args, _build_adapter, evaluate, ppo_update


def correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(rho=None, pvalue=None, n=len(x), reason="insufficient/constant obs")
    r = spearmanr(x, y)
    return dict(rho=float(r.statistic), pvalue=float(r.pvalue), n=len(x))


def bootstrap_ci(x, y, iters: int = 1000, seed: int = 0):
    if len(x) < 4:
        return None
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(iters):
        idx = rng.integers(0, len(x), size=len(x))
        xi = np.asarray(x)[idx]
        yi = np.asarray(y)[idx]
        if np.ptp(xi) == 0 or np.ptp(yi) == 0:
            continue
        rhos.append(spearmanr(xi, yi).statistic)
    if not rhos:
        return None
    return dict(lo=float(np.percentile(rhos, 2.5)), hi=float(np.percentile(rhos, 97.5)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--kind", choices=["sigma", "p", "both"], default="both")
    p.add_argument("--regions", type=int, default=30, help="number of regions to test (§4.13 minimum 30)")
    p.add_argument("--oracle-probes", type=int, default=64)
    p.add_argument("--small-probes", type=int, default=4)
    p.add_argument("--reference-episodes", type=int, default=50, help="§4.13 minimum 50")
    p.add_argument("--reference-horizon", type=int, default=32)
    p.add_argument("--region-horizon", type=int, default=32)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=710)
    p.add_argument("--output-dir", default="outputs/mechanisms")
    p.add_argument("--num-envs", type=int, default=32)
    opt = p.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    cp = torch.load(opt.checkpoint, map_location="cpu", weights_only=False)
    args = Args(**cp["args"]) if isinstance(cp.get("args"), dict) else Args(**cp.get("config", {}))
    args.num_envs = opt.num_envs
    args.num_envs_ref = min(args.num_envs_ref, opt.num_envs)
    args.num_eval_envs = min(8, opt.num_envs)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    adapter = _build_adapter(args, args.num_envs)
    ref_adapter = _build_adapter(args, args.num_envs_ref)
    eval_adapter = _build_adapter(args, args.num_eval_envs)
    obs_dim, action_dim = adapter.obs_dim, adapter.action_dim
    agent = Agent(obs_dim, action_dim, args.hidden).to(device)
    agent.load_state_dict(cp["agent"])
    out = Path(opt.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(
        json.dumps({**vars(opt), "training_step": cp.get("global_steps")}, indent=2)
    )
    summary = {}

    # -------------------- σ mechanism ---------------------------------
    if opt.kind in ("sigma", "both"):
        rows = []
        # Not enough state to reload archive from a plain checkpoint (we only save
        # `agent`); re-collect a small archive by rolling the current policy and
        # snapshotting slots at intervals.
        # For this mechanism check we treat each initial reset state as a "region".
        obs, _ = adapter.reset(seed=opt.seed)
        for r in range(opt.regions):
            slot_ids = torch.tensor([r % adapter.num_envs], dtype=torch.long)
            snaps = adapter.save_state(slot_ids)
            # small-K vs oracle-K probes from the same snapshot
            noise = torch.randn(opt.small_probes, action_dim, device=device)
            snaps_small = [snaps[0]] * opt.small_probes
            slots_small = torch.arange(opt.small_probes, device=device, dtype=torch.long)
            adapter_small = _build_adapter(args, opt.small_probes)
            z_small, _ = probe_rollout(
                adapter_small, agent, slots_small.cpu(), snaps_small,
                probe_horizon=args.probe_horizon, device=device,
                first_action_noise=noise, probe_scale=args.probe_scale,
            )
            adapter_small.close()
            adapter_oracle = _build_adapter(args, opt.oracle_probes)
            oracle_noise = torch.randn(opt.oracle_probes, action_dim, device=device)
            z_oracle, _ = probe_rollout(
                adapter_oracle, agent,
                torch.arange(opt.oracle_probes, dtype=torch.long),
                [snaps[0]] * opt.oracle_probes,
                probe_horizon=args.probe_horizon, device=device,
                first_action_noise=oracle_noise, probe_scale=args.probe_scale,
            )
            adapter_oracle.close()
            from herp.sigma import pairwise_sigma
            small_sig = float(pairwise_sigma(z_small.float()))
            oracle_sig = float(pairwise_sigma(z_oracle.float()))
            rows.append(dict(region_id=r, sigma=small_sig, oracle_sigma=oracle_sig))
            # step forward a bit before next region snapshot
            for _ in range(4):
                action = agent.act(obs.to(device))
                action = action.clamp(adapter.action_low(), adapter.action_high())
                obs, _, _, _, _ = adapter.step(action)
        c = correlation([r["sigma"] for r in rows], [r["oracle_sigma"] for r in rows])
        c["bootstrap_ci"] = bootstrap_ci(
            [r["sigma"] for r in rows], [r["oracle_sigma"] for r in rows], seed=opt.seed
        )
        summary["sigma"] = c
        (out / "sigma.json").write_text(json.dumps(rows, indent=2))

    # -------------------- p_v controlled PPO-delta ---------------------
    if opt.kind in ("p", "both"):
        # D_base — one reference rollout the two clones both train on.
        base = collect_reference(ref_adapter, agent, opt.reference_horizon, device, seed=opt.seed + 1)
        # Reference batch for evaluation of J_ref (deterministic policy, ordinary resets).
        def eval_J(policy):
            evald = evaluate(eval_adapter, policy, args, device, opt.eval_episodes)
            return float(evald["eval_return"])

        # Clone A: PPO update on D_base
        def _clone_update(policy, batch, seed):
            clone = copy.deepcopy(policy)
            opt2 = torch.optim.Adam(clone.parameters(), lr=args.learning_rate, eps=1e-5)
            rollout = dict(
                obs=batch["obs"].to(device),
                actions=batch["actions"].to(device),
                logprobs=torch.zeros(batch["obs"].shape[0], device=device),
                advantages=batch["advantages"].to(device),
                returns=batch["advantages"].to(device),
                values=torch.zeros(batch["obs"].shape[0], device=device),
            )
            # Recompute logprobs / values under the CLONE so PPO stays consistent.
            with torch.no_grad():
                _, lp, _, val = clone.get_action_and_value(rollout["obs"], rollout["actions"])
            rollout["logprobs"] = lp
            rollout["values"] = val.view(-1)
            rollout["returns"] = rollout["advantages"] + rollout["values"]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                ppo_update(clone, opt2, rollout, args)
            return clone

        base_dict = dict(
            obs=base.obs, actions=base.actions,
            advantages=base.advantages, rewards=base.rewards,
        )
        clone_A = _clone_update(agent, base_dict, seed=opt.seed + 7919)
        j_A = eval_J(clone_A)

        rows = []
        obs, _ = adapter.reset(seed=opt.seed + 2)
        for r in range(opt.regions):
            # Collect a region-specific batch by rolling out from a snapshot.
            slot = torch.tensor([r % adapter.num_envs], dtype=torch.long)
            snaps = adapter.save_state(slot)
            # Restore across many slots to produce a region batch
            region_slots = torch.arange(args.num_envs_ref, dtype=torch.long)
            region_snaps = [snaps[0]] * args.num_envs_ref
            _ = ref_adapter.restore_state(region_slots, region_snaps)
            region_batch = collect_reference(
                ref_adapter, agent, opt.region_horizon, device, seed=opt.seed + 1000 + r
            )
            combined = dict(
                obs=torch.cat([base.obs, region_batch.obs], dim=0),
                actions=torch.cat([base.actions, region_batch.actions], dim=0),
                advantages=torch.cat([base.advantages, region_batch.advantages], dim=0),
                rewards=torch.cat([base.rewards, region_batch.rewards], dim=0),
            )
            clone_B = _clone_update(agent, combined, seed=opt.seed + 7919)
            j_B = eval_J(clone_B)
            adv_scale = max(float(base.advantages.std().item()), 1e-6)
            g = policy_gradient_signature(
                agent, region_batch.obs.to(device), region_batch.actions.to(device),
                region_batch.advantages.to(device), adv_scale=adv_scale,
            )
            g_ref = policy_gradient_signature(
                agent, base.obs.to(device), base.actions.to(device),
                base.advantages.to(device), adv_scale=adv_scale,
            )
            cosine = float(torch.nn.functional.cosine_similarity(g, g_ref, dim=0))
            dot = float(torch.dot(g, g_ref))
            fisher_diag = empirical_fisher_diagonal(
                agent, base.obs.to(device), base.actions.to(device)
            )
            fisher_val = float(torch.dot(g_ref, g / (fisher_diag + args.fisher_damping)))
            rows.append(
                dict(
                    region_id=r,
                    delta=j_B - j_A,
                    cosine=cosine,
                    dot=dot,
                    fisher=fisher_val,
                    j_A=j_A,
                    j_B=j_B,
                )
            )
            for _ in range(4):
                action = agent.act(obs.to(device))
                action = action.clamp(adapter.action_low(), adapter.action_high())
                obs, _, _, _, _ = adapter.step(action)

        summary["p"] = {}
        for key in ("cosine", "dot", "fisher"):
            xs = [r[key] for r in rows]
            ys = [r["delta"] for r in rows]
            c = correlation(xs, ys)
            c["bootstrap_ci"] = bootstrap_ci(xs, ys, seed=opt.seed)
            summary["p"][key] = c
        (out / "p.json").write_text(json.dumps(rows, indent=2))

    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    for a in (adapter, ref_adapter, eval_adapter):
        a.close()


if __name__ == "__main__":
    main()
