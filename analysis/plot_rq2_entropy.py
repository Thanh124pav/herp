"""RQ2 — Normalized allocation entropy over training.

Reads regions.jsonl from HERP runs and computes H_norm(t) per PLAN.md §7.1.
Produces a two-panel figure: (a) ManiSkill, (b) DMC.
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_allocation_entropy(run_dir):
    """Extract (step, H_norm) pairs from a HERP run's regions.jsonl."""
    regions_file = Path(run_dir) / "regions.jsonl"
    if not regions_file.exists():
        return [], []

    step_allocs = defaultdict(dict)
    with open(regions_file) as f:
        for line in f:
            row = json.loads(line)
            step = row.get("step", 0)
            rid = row.get("region_id", 0)
            frags = row.get("allocated_fragments", 0)
            step_allocs[step][rid] = frags

    steps, entropies = [], []
    for step in sorted(step_allocs):
        allocs = step_allocs[step]
        total = sum(allocs.values())
        if total <= 0:
            continue
        K = len(allocs)
        if K < 2:
            continue
        probs = np.array([v / total for v in allocs.values()])
        probs = probs[probs > 0]
        H = -np.sum(probs * np.log(probs))
        H_norm = H / np.log(K)
        steps.append(step)
        entropies.append(H_norm)

    return steps, entropies


def load_from_metrics(run_dir):
    """Fallback: extract allocation_entropy from metrics.jsonl training records."""
    metrics_file = Path(run_dir) / "metrics.jsonl"
    if not metrics_file.exists():
        return [], []

    steps, entropies = [], []
    with open(metrics_file) as f:
        for line in f:
            row = json.loads(line)
            if row.get("type") != "training":
                continue
            step = row.get("step", 0)
            H = row.get("allocation_entropy", None)
            K = row.get("num_regions", 1)
            if H is not None and K >= 2:
                H_norm = H / np.log(K)
                steps.append(step)
                entropies.append(H_norm)

    return steps, entropies


def smooth(values, window=5):
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dir", default="outputs/final_campaign")
    p.add_argument("--output", default="outputs/final_campaign/rq2/entropy_curves.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()

    campaign = Path(args.campaign_dir)
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    tasks = {
        "maniskill": ["PickCube-v1", "LiftPegUpright-v1"],
        "dmc": ["walker-run", "cartpole-swingup_sparse"],
    }

    results = {}
    for bench, env_ids in tasks.items():
        for env_id in env_ids:
            key = f"{bench}/{env_id}"
            seed_data = []
            for seed in args.seeds:
                # Look in rq3 first (HERP runs), then rq1
                for parent in ["rq3", "rq1"]:
                    run_dir = campaign / parent / bench / f"{env_id}_herp_s{seed}"
                    if not run_dir.exists():
                        run_dir = campaign / parent / f"{bench}_{env_id}_herp_s{seed}"
                    if run_dir.exists():
                        steps, ent = load_allocation_entropy(run_dir)
                        if not steps:
                            steps, ent = load_from_metrics(run_dir)
                        if steps:
                            seed_data.append({"seed": seed, "steps": steps,
                                              "entropy": [float(e) for e in ent]})
                            break

            if seed_data:
                results[key] = seed_data
                print(f"  {key}: {len(seed_data)} seeds, "
                      f"mean final H_norm = {np.mean([d['entropy'][-1] for d in seed_data]):.3f}")

    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_file}")
    print(f"Tasks with data: {list(results.keys())}")


if __name__ == "__main__":
    main()
