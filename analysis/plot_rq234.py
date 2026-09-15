"""Plot RQ2, RQ3, RQ4 figures and upload to W&B.

RQ2: Region allocation entropy over training (diversity of exploration).
RQ3: p×σ ablation — bar chart + learning curves.
RQ4: σ predictor vs oracle — scatter + per-checkpoint correlation.

Usage:
    python analysis/plot_rq234.py [--wandb-project herp-paper-figures] [--no-wandb]
"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_campaign"
FIG_DIR = ROOT / "analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS_RQ3 = ["herp", "herp_p", "herp_sigma", "uniform"]
METHOD_LABELS = {"herp": r"HERP ($p \times \sigma$)", "herp_p": r"HERP ($p$ only)",
                 "herp_sigma": r"HERP ($\sigma$ only)", "uniform": "Uniform"}
METHOD_COLORS = {"herp": "#2196F3", "herp_p": "#FF9800", "herp_sigma": "#4CAF50", "uniform": "#9E9E9E"}

TASK_LABELS = {"walker-run": "DMC Walker-Run", "cheetah-run": "DMC Cheetah-Run",
               "PickCube-v1": "ManiSkill PickCube"}


# ---------------------------------------------------------------------------
# RQ2: Entropy curves
# ---------------------------------------------------------------------------
def plot_rq2():
    entropy_file = OUT / "rq2" / "entropy_curves.json"
    if not entropy_file.exists():
        print("[RQ2] entropy_curves.json not found, skipping")
        return None

    data = json.loads(entropy_file.read_text())
    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 4), squeeze=False)

    for idx, (task_key, seeds) in enumerate(sorted(data.items())):
        ax = axes[0, idx]
        all_steps = []
        all_entropy = []
        for entry in seeds:
            s, e = np.array(entry["steps"]), np.array(entry["entropy"])
            # Drop last point for walker-run (outlier drop at end)
            if "walker" in task_key and len(s) > 1:
                s, e = s[:-1], e[:-1]
            ax.plot(s, e, alpha=0.25, color="#2196F3", linewidth=0.8)
            all_steps.append(s)
            all_entropy.append(e)

        # Mean curve (interpolated to common grid)
        if len(all_steps) > 1:
            min_len = min(len(s) for s in all_steps)
            common = all_steps[0][:min_len]
            aligned = np.array([e[:min_len] for e in all_entropy])
            mean_e = aligned.mean(0)
            std_e = aligned.std(0)
            ax.plot(common, mean_e, color="#1565C0", linewidth=2, label="Mean")
            ax.fill_between(common, mean_e - std_e, mean_e + std_e, alpha=0.15, color="#1565C0")

        task_name = task_key.split("/")[-1]
        ax.set_title(TASK_LABELS.get(task_name, task_name))
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Allocation Entropy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("RQ2: Exploration Diversity (Allocation Entropy)", fontsize=13, y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "rq2_entropy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ2] Saved {path}")
    return path


# ---------------------------------------------------------------------------
# RQ3: Ablation — learning curves + bar chart
# ---------------------------------------------------------------------------
def _load_rq3_curves():
    curves = defaultdict(lambda: defaultdict(list))
    for bench in ["dmc", "maniskill"]:
        base = OUT / "rq3" / bench
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            mf = d / "metrics.jsonl"
            if not mf.exists():
                continue
            parts = d.name.rsplit("_s", 1)
            method_key = parts[0]
            task = method_key.rsplit("_", 1)
            # Parse: e.g. walker-run_herp_p -> task=walker-run, method=herp_p
            # Complex because task names can have hyphens
            for m in METHODS_RQ3:
                suffix = f"_{m}"
                if method_key.endswith(suffix):
                    task_name = method_key[: -len(suffix)]
                    break
            else:
                continue

            evals = []
            with open(mf) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("type") == "evaluation":
                        evals.append((rec["step"], rec["eval_return"]))
            if evals:
                curves[(bench, task_name)][m].append(evals)
    return curves


def plot_rq3():
    curves = _load_rq3_curves()
    if not curves:
        print("[RQ3] No data found, skipping")
        return []

    paths = []

    # --- Learning curves ---
    tasks = sorted(curves.keys())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for idx, (bench, task) in enumerate(tasks):
        ax = axes[0, idx]
        for m in METHODS_RQ3:
            if m not in curves[(bench, task)]:
                continue
            runs = curves[(bench, task)][m]
            # Align to common steps
            min_len = min(len(run) for run in runs)
            common_steps = [runs[0][i][0] for i in range(min_len)]
            vals = np.array([[run[i][1] for i in range(min_len)] for run in runs])
            mean_v = vals.mean(0)
            std_v = vals.std(0)
            ax.plot(common_steps, mean_v, label=METHOD_LABELS[m], color=METHOD_COLORS[m], linewidth=1.5)
            ax.fill_between(common_steps, mean_v - std_v, mean_v + std_v,
                            alpha=0.12, color=METHOD_COLORS[m])

        ax.set_title(TASK_LABELS.get(task, f"{bench}/{task}"))
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Eval Return")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("RQ3: Ablation Learning Curves", fontsize=13, y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "rq3_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ3] Saved {path}")
    paths.append(path)

    # --- Bar chart (final performance) ---
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(tasks)), 4.5))
    x_pos = np.arange(len(tasks))
    width = 0.18
    for i, m in enumerate(METHODS_RQ3):
        means, stds = [], []
        for bench, task in tasks:
            runs = curves[(bench, task)].get(m, [])
            finals = [r[-1][1] for r in runs if r]
            means.append(np.mean(finals) if finals else 0)
            stds.append(np.std(finals) if finals else 0)
        offset = (i - len(METHODS_RQ3) / 2 + 0.5) * width
        ax.bar(x_pos + offset, means, width, yerr=stds, label=METHOD_LABELS[m],
               color=METHOD_COLORS[m], capsize=3, alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([TASK_LABELS.get(t, f"{b}/{t}") for b, t in tasks], fontsize=9)
    ax.set_ylabel("Final Eval Return")
    ax.set_title("RQ3: Ablation — Final Performance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = FIG_DIR / "rq3_bars.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ3] Saved {path}")
    paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# RQ4: σ predictor vs oracle
# ---------------------------------------------------------------------------
def _full_window_q(feat_3d):
    """Compute q = features.var(dim=0, ddof=1).sum(-1).mean() matching sigma.py."""
    if len(feat_3d) < 2:
        return float("nan")
    return feat_3d.astype(np.float64).var(axis=0, ddof=1).sum(axis=-1).mean()


def _combined_q(q_direct, q_pred, n, kappa=8.0):
    """Shrinkage: blend direct estimate with predictor, matching sigma_predictor.py."""
    q_pred = max(0.0, q_pred)
    if not np.isfinite(q_direct) or n == 0:
        return q_pred
    lam = n / (n + kappa)
    return lam * max(0.0, q_direct) + (1 - lam) * q_pred


def _compute_oracle_targets(pool_path):
    """Return (predictor, shrinkage, oracle, n_valid_per_region)."""
    d = np.load(pool_path)
    features = d["features"]   # (R, total_samples, H, D)
    valid = d["valid"]         # (R, total_samples, H)
    cp = d["checkpoint_predictions"]  # (R,)
    low_pool = int(d["low_pool"])

    oracle_feat = features[:, low_pool:, :, :]
    oracle_valid = valid[:, low_pool:, :]
    lp_feat = features[:, :low_pool, :, :]
    lp_valid = valid[:, :low_pool, :]

    oracle_q, lp_q, shrinkage_q, n_valid = [], [], [], []
    for r in range(features.shape[0]):
        # Oracle target (128 samples)
        ov = oracle_valid[r].all(axis=1)
        of = oracle_feat[r][ov]
        oq = _full_window_q(of) if len(of) >= 2 else 0.0
        oracle_q.append(oq)

        # Low-pool empirical (32 samples)
        lv = lp_valid[r].all(axis=1)
        lf = lp_feat[r][lv]
        lq = _full_window_q(lf) if len(lf) >= 2 else float("nan")
        lp_q.append(lq)
        n_valid.append(int(lv.sum()))

        # Shrinkage: combined_q(low_pool_empirical, predictor, n, kappa=8)
        sq = _combined_q(lq, float(cp[r]), int(lv.sum()), kappa=8.0)
        shrinkage_q.append(sq)

    return (cp, np.array(shrinkage_q), np.array(oracle_q),
            np.array(lp_q), np.array(n_valid))


def plot_rq4():
    rq4_dir = OUT / "rq4"
    if not rq4_dir.exists():
        print("[RQ4] No data found, skipping")
        return None

    # Collect data for 3 estimators: predictor, shrinkage, low-pool empirical
    all_data = {"pred": [], "shrinkage": [], "lp_emp": [], "oracle": [], "bench": []}
    corr_rows = []

    for bench in ["dmc", "maniskill"]:
        base = rq4_dir / bench
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            pool_f = d / "oracle_pool.npz"
            if not pool_f.exists():
                continue
            pred, shrinkage, oracle, lp_emp, n_valid = _compute_oracle_targets(pool_f)
            n = len(pred)
            all_data["pred"].extend(pred)
            all_data["shrinkage"].extend(shrinkage)
            all_data["lp_emp"].extend(lp_emp)
            all_data["oracle"].extend(oracle)
            all_data["bench"].extend([bench] * n)

            # Per-checkpoint correlations for all 3 estimators
            row = dict(checkpoint=d.name, bench=bench, n=n)
            for est_name, est_vals in [("pred", pred), ("shrinkage", shrinkage), ("lp_emp", lp_emp)]:
                valid = np.isfinite(est_vals) & np.isfinite(oracle)
                if valid.sum() >= 5:
                    r_sp, p_sp = stats.spearmanr(est_vals[valid], oracle[valid])
                    r_pe, p_pe = stats.pearsonr(est_vals[valid], oracle[valid])
                else:
                    r_sp = r_pe = p_sp = p_pe = float("nan")
                row[f"{est_name}_spearman"] = r_sp
                row[f"{est_name}_pearson"] = r_pe
                row[f"{est_name}_p_sp"] = p_sp
            corr_rows.append(row)

    if not all_data["pred"]:
        print("[RQ4] No oracle pools, skipping")
        return None

    for k in all_data:
        all_data[k] = np.array(all_data[k])

    # --- Main scatter: 3 estimators × 2 benchmarks ---
    estimators = [
        ("pred", "Predictor", "#9C27B0"),
        ("shrinkage", "Shrinkage", "#2196F3"),
        ("lp_emp", "Empirical (32)", "#FF9800"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    bench_labels = {"dmc": "DMC Walker-Run", "maniskill": "ManiSkill PickCube"}

    for row_i, bench in enumerate(["dmc", "maniskill"]):
        mask = all_data["bench"] == bench
        oracle = all_data["oracle"][mask]
        for col_i, (est_key, est_label, color) in enumerate(estimators):
            ax = axes[row_i, col_i]
            est = all_data[est_key][mask].astype(float)
            valid = np.isfinite(est) & np.isfinite(oracle)
            e, o = est[valid], oracle[valid]
            ax.scatter(e, o, alpha=0.3, s=12, color=color)

            if len(e) >= 5:
                slope, intercept, r_pe, _, _ = stats.linregress(e, o)
                x_line = np.linspace(e.min(), e.max(), 50)
                ax.plot(x_line, slope * x_line + intercept, "--", color="black", linewidth=1.5)
                r_sp, _ = stats.spearmanr(e, o)
                ax.set_title(f"{bench_labels[bench]}\nρ={r_sp:.3f}, r={r_pe:.3f}", fontsize=10)
            else:
                ax.set_title(f"{bench_labels[bench]}\ninsufficient data", fontsize=10)

            lo = min(e.min(), o.min()) if len(e) else 0
            hi = max(e.max(), o.max()) if len(e) else 1
            ax.plot([lo, hi], [lo, hi], ":", color="gray", alpha=0.5)
            ax.set_xlabel(f"{est_label} σ̂")
            ax.set_ylabel("Oracle σ")
            ax.grid(True, alpha=0.3)

    fig.suptitle("RQ4: Estimator vs Oracle (Predictor | Shrinkage | Empirical-32)", fontsize=13, y=1.01)
    fig.tight_layout()
    path = FIG_DIR / "rq4_scatter.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ4] Saved {path}")

    # --- Summary bar chart: mean Spearman across checkpoints per estimator ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    for pi, bench in enumerate(["dmc", "maniskill"]):
        ax = axes[0, pi]
        rows = [r for r in corr_rows if r["bench"] == bench]
        if not rows:
            continue
        xs = np.arange(len(rows))
        width = 0.25
        for ei, (est_key, est_label, color) in enumerate(estimators):
            vals = [r.get(f"{est_key}_spearman", 0) for r in rows]
            offset = (ei - 1) * width
            ax.bar(xs + offset, vals, width, label=est_label, color=color, alpha=0.8)

        ax.set_xticks(xs)
        labels = []
        for r in rows:
            name = r["checkpoint"]
            name = name.replace("walker-run_", "").replace("PickCube-v1_", "")
            name = name.replace("_checkpoint", "\nck").replace("_ck", "\nck")
            labels.append(name)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Spearman ρ")
        ax.set_title(bench_labels[bench])
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(-0.4, 0.7)

    fig.suptitle("RQ4: Per-Checkpoint Spearman ρ (3 estimators vs Oracle)", fontsize=13, y=1.02)
    fig.tight_layout()
    path2 = FIG_DIR / "rq4_correlations.png"
    fig.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[RQ4] Saved {path2}")

    return path, path2, corr_rows


# ---------------------------------------------------------------------------
# W&B upload
# ---------------------------------------------------------------------------
def upload_wandb(project, figures):
    import wandb
    run = wandb.init(project=project, name="rq234-figures", job_type="analysis",
                     tags=["rq2", "rq3", "rq4", "figures"])
    for path in figures:
        if path and isinstance(path, Path) and path.exists():
            run.log({path.stem: wandb.Image(str(path))})
            print(f"  [wandb] uploaded {path.name}")
    run.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wandb-project", default="herp-paper-figures")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    figures = []

    path = plot_rq2()
    if path:
        figures.append(path)

    paths = plot_rq3()
    figures.extend(paths)

    result = plot_rq4()
    if result:
        scatter, corr_fig, corr_rows = result
        figures.extend([scatter, corr_fig])
        print("\n[RQ4] Correlation summary (Spearman ρ with oracle):")
        print(f"  {'Checkpoint':40s}  {'Predictor':>10s}  {'Shrinkage':>10s}  {'Empirical':>10s}")
        print("  " + "-" * 75)
        for r in corr_rows:
            sp_p = r.get("pred_spearman", float("nan"))
            sp_s = r.get("shrinkage_spearman", float("nan"))
            sp_e = r.get("lp_emp_spearman", float("nan"))
            sig_s = "*" if r.get("shrinkage_p_sp", 1) < 0.05 else " "
            print(f"  {r['checkpoint']:40s}  {sp_p:+.3f}      {sp_s:+.3f}{sig_s}     {sp_e:+.3f}")
        # Summary
        for bench in ["dmc", "maniskill"]:
            rows = [r for r in corr_rows if r["bench"] == bench]
            if not rows:
                continue
            for est in ["pred", "shrinkage", "lp_emp"]:
                vals = [r[f"{est}_spearman"] for r in rows if np.isfinite(r[f"{est}_spearman"])]
                print(f"  {bench:10s} {est:12s} mean ρ = {np.mean(vals):+.3f} (n={len(vals)})")

    if figures and not args.no_wandb:
        upload_wandb(args.wandb_project, figures)
    elif not figures:
        print("No figures generated.")
    else:
        print(f"\n{len(figures)} figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
