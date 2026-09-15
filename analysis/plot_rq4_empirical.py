"""RQ4: σ oracle validation — empirical (32 samples) vs oracle (128 samples).

Produces a clean 1×2 scatter plot for the paper.
"""
import argparse, json, os
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


def full_window_q(feat):
    if len(feat) < 2:
        return float("nan")
    return feat.astype(np.float64).var(axis=0, ddof=1).sum(axis=-1).mean()


def load_rq4():
    rows = {"lp": [], "oracle": [], "bench": [], "ckpt": []}
    for bench in ["dmc", "maniskill"]:
        base = OUT / "rq4" / bench
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            pool_f = d / "oracle_pool.npz"
            if not pool_f.exists():
                continue
            data = np.load(pool_f)
            features = data["features"]
            valid = data["valid"]
            low_pool = int(data["low_pool"])

            for r in range(features.shape[0]):
                # Oracle (128 samples)
                ov = valid[r, low_pool:].all(axis=1)
                of = features[r, low_pool:][ov]
                oq = full_window_q(of) if len(of) >= 2 else np.nan

                # Empirical low-pool (32 samples)
                lv = valid[r, :low_pool].all(axis=1)
                lf = features[r, :low_pool][lv]
                lq = full_window_q(lf) if len(lf) >= 2 else np.nan

                rows["lp"].append(lq)
                rows["oracle"].append(oq)
                rows["bench"].append(bench)
                rows["ckpt"].append(d.name)

    return {k: np.array(v) for k, v in rows.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wandb-project", default="herp-paper-figures")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    data = load_rq4()
    bench_meta = {
        "dmc": ("DMC Walker-Run", "#E67E22"),
        "maniskill": ("ManiSkill PickCube", "#2980B9"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for i, bench in enumerate(["dmc", "maniskill"]):
        ax = axes[i]
        mask = (data["bench"] == bench) & np.isfinite(data["lp"]) & np.isfinite(data["oracle"])
        lp = data["lp"][mask]
        oracle = data["oracle"][mask]
        title, color = bench_meta[bench]

        ax.scatter(lp, oracle, alpha=0.35, s=18, color=color, edgecolors="none")

        # Regression
        slope, intercept, r_pe, p_pe, _ = stats.linregress(lp, oracle)
        r_sp, p_sp = stats.spearmanr(lp, oracle)
        x_line = np.linspace(lp.min(), lp.max(), 50)
        ax.plot(x_line, slope * x_line + intercept, "--", color="#2C3E50", linewidth=1.8,
                label=f"Pearson $r$={r_pe:.2f}")

        # Identity
        lo = min(lp.min(), oracle.min())
        hi = max(lp.max(), oracle.max())
        ax.plot([lo, hi], [lo, hi], ":", color="#95A5A6", linewidth=1)

        ax.set_xlabel(r"Empirical $\hat{\sigma}$ (32 samples)", fontsize=11)
        ax.set_ylabel(r"Oracle $\sigma$ (128 samples)", fontsize=11)
        ax.set_title(f"{title}\nSpearman $\\rho$={r_sp:.2f}, Pearson $r$={r_pe:.2f}", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.25)

    fig.suptitle(r"RQ4: Empirical $\hat{\sigma}$ vs Oracle $\sigma$", fontsize=14, y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "rq4_empirical.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name="rq4-empirical",
                         job_type="analysis", tags=["rq4", "empirical", "figure"])
        run.log({"rq4_empirical": wandb.Image(str(path))})
        run.finish()
        print("Uploaded to W&B")


if __name__ == "__main__":
    main()
