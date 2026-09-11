#!/usr/bin/env python3
"""Re-evaluate saved checkpoints and report BOTH metrics:
  - success_once   (reached success at any step)   -> eval_success
  - success_at_end (success at the terminal step)   -> eval_success_final

Loads each cell's best-success_once checkpoint, rebuilds the eval env with the
training eval protocol, runs evaluate(), records both. Works for any benchmark
(cell dirs are found by globbing <root>/*/<task>_<method>_s0). No retraining.

Cells whose success_once (from metrics) is <= --skip-below are recorded as
0/0 without a GPU/CPU rollout (they never reach success, so at_end is 0 too).

Example (ManiSkill):
  python scripts/reeval_checkpoints.py --root outputs/full_matrix_1seed \
    --tasks PushCube-v1 PickCube-v1 LiftPegUpright-v1 PlaceSphere-v1 PokeCube-v1 StackCube-v1 \
    --eval-envs 32 --episodes 100 --out outputs/reeval_both_metrics.json
Example (Meta-World/Fetch, CPU sim — skip the all-zero cells):
  python scripts/reeval_checkpoints.py --root outputs/cpu_mw_fetch_1seed \
    --tasks button-press-v3 ... FetchPush-v4 ... --eval-envs 8 --episodes 30 \
    --skip-below 0.05 --out outputs/reeval_mw_fetch.json
"""
import argparse, csv, glob, json, sys
from dataclasses import fields
import torch

sys.path.insert(0, "/workspace/herp")
from train import Agent, evaluate, _build_adapter, Args  # noqa: E402

DEVICE = "cuda"
_ARGS_FIELDS = {f.name for f in fields(Args)}
_DEFAULT_METHODS = ["ppo", "rnd", "disagreement", "herp_sigma", "herp_p", "herp"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=_DEFAULT_METHODS)
    p.add_argument("--eval-envs", type=int, default=32)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--skip-below", type=float, default=-1.0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def cell_dir(root, task, method):
    hits = glob.glob(f"{root}/*/{task}_{method}_s0")
    return hits[0] if hits else None


def best_from_metrics(cd):
    """(best_step, best_once) from metrics.csv, else (None, None)."""
    csvs = glob.glob(f"{cd}/*/metrics.csv")
    if not csvs:
        return None, None
    best_step, best = None, -1.0
    for row in csv.DictReader(open(csvs[0])):
        v = row.get("eval_success", "")
        if v in ("", None):
            continue
        try:
            s = float(v); step = int(row["global_env_steps"])
        except ValueError:
            continue
        if s > best:
            best, best_step = s, step
    return best_step, (best if best >= 0 else None)


def find_ckpt(cd, step):
    cks = glob.glob(f"{cd}/*/checkpoint_*.pt")
    if not cks:
        return None
    exact = [c for c in cks if c.endswith(f"checkpoint_{step}.pt")]
    return exact[0] if exact else min(
        cks, key=lambda c: abs(int(c.rsplit("_", 1)[1][:-3]) - step))


def reeval(cd, eval_envs, episodes):
    step, _ = best_from_metrics(cd)
    if step is None:
        return None
    ckpt_path = find_ckpt(cd, step)
    if ckpt_path is None:
        return None
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    args = Args(**{k: v for k, v in ck["args"].items() if k in _ARGS_FIELDS})
    args.num_eval_envs = eval_envs
    adapter = _build_adapter(args, eval_envs, role="eval")
    try:
        agent = Agent(adapter.obs_dim, adapter.action_dim, args.hidden).to(DEVICE)
        agent.load_state_dict(ck["agent"]); agent.eval()
        res = evaluate(adapter, agent, args, DEVICE, episodes)
    finally:
        adapter.close()
    return dict(step=step, once=res["eval_success"],
                at_end=res["eval_success_final"], episodes=res["eval_episodes"])


def main():
    opt = parse()
    rows = []
    for t in opt.tasks:
        for m in opt.methods:
            cd = cell_dir(opt.root, t, m)
            if cd is None:
                print(f"  [{t}/{m}] no cell dir", flush=True); continue
            step, once = best_from_metrics(cd)
            if once is not None and once <= opt.skip_below:
                print(f"  [{t}/{m}] once={once:.2f} <= skip-below -> at_end=0 (no rollout)", flush=True)
                rows.append(dict(task=t, method=m, step=step, once=once, at_end=0.0,
                                 episodes=0, skipped=True))
                continue
            try:
                r = reeval(cd, opt.eval_envs, opt.episodes)
            except Exception as e:
                print(f"  [{t}/{m}] FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
                r = None
            if r is None:
                print(f"  [{t}/{m}] no checkpoint", flush=True); continue
            print(f"  [{t}/{m}] step={r['step']} once={r['once']:.2f} at_end={r['at_end']:.2f} (n={r['episodes']})", flush=True)
            rows.append(dict(task=t, method=m, **r))
    json.dump(rows, open(opt.out, "w"), indent=2)
    print(f"\nsaved {opt.out}")


if __name__ == "__main__":
    main()
