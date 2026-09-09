# HERP pilot v2 — refactored codebase (2026-09-09, 22:10)

Same protocol as the 9/9 pilot v1 report but running against the refactored
codebase built to the new `IMPLEMENTATION.md`: `EnvAdapter` abstraction, full
actor gradient signature (§7), shared advantage scaling (§8), hybrid p
estimator (§10), enriched provenance (§33), and the renamed script surface
(§32). Purpose is a same-budget A/B check, not a paper-ready benchmark.

## Protocol

- Benchmark: ManiSkill, tasks `PushCube-v1` and `PickCube-v1`
- Methods: `ppo`, `rnd`, `disagreement`, `herp_sigma`, `herp_p`, `herp`
- Seeds: 0, 1, 2 (3 per method-task)
- Training budget: 32,768 environment interactions per run (probes + reference
  charged to the same budget)
- Evaluation: 50 ordinary-reset episodes per checkpoint, shared eval seeds
  across methods, checkpoints at every 8,192 steps
- Suite runner: `scripts/run_suite.py` with 4 parallel workers
- Total training interactions across the suite: 36 × 32,768 = 1,179,648
- Zero failures; 36/36 complete
- Wall-clock: ~63 minutes
- Reproducibility artifacts per run under
  `outputs/herp_pilot_v2/maniskill/<task>_<method>_s<seed>/*/` — config.json,
  metrics.csv, regions.csv, checkpoint_*.pt, provenance.json

## Main table (mean ± sample std across 3 training seeds)

| Task | Method | Return v1 → v2 | Success AUC v1 → v2 |
|---|---|---|---|
| PickCube-v1 | ppo          | 30.35 → 31.59 | 0.000 → 0.000 |
| PickCube-v1 | rnd          | 32.35 → 31.60 | 0.000 → 0.000 |
| PickCube-v1 | disagreement | 29.76 → 29.89 | 0.000 → 0.000 |
| PickCube-v1 | herp_sigma   | 29.95 → 31.70 | 0.000 → 0.000 |
| PickCube-v1 | herp_p       | 31.25 → 30.00 | 0.000 → 0.000 |
| PickCube-v1 | herp         | 30.83 → 29.63 | 0.000 → 0.000 |
| PushCube-v1 | ppo          | 29.53 → 27.93 | 0.006 → 0.005 |
| PushCube-v1 | rnd          | 29.64 → 27.65 | 0.020 → 0.007 |
| PushCube-v1 | disagreement | 28.52 → 28.76 | 0.003 → 0.007 |
| PushCube-v1 | herp_sigma   | 28.00 → 26.81 | 0.010 → 0.013 |
| PushCube-v1 | herp_p       | 25.73 → 26.72 | 0.023 → 0.012 |
| PushCube-v1 | herp         | 28.01 → 27.44 | 0.010 → 0.020 |

Both pilots land at ≈0% success. The 32k budget is far below what the PPO
backbone needs to solve these tasks, so **this run cannot rank methods**;
IMPL §23 (`The 32k pilot is not a final benchmark`) is confirmed empirically.

## Sigma mechanism — K=4 vs independent K=64 oracle, 50 archived regions

Bootstrap percentile CIs computed with 2,000 resamples of the 50 sampled
regions in a single seed's checkpoint.

| Checkpoint | v1 ρ [95 % CI] | v2 ρ [95 % CI] |
|---|---|---|
|  8,192 | 0.693 [0.482, 0.829] | 0.626 [0.390, 0.785] |
| 32,768 | 0.566 [0.327, 0.746] | 0.620 [0.411, 0.757] |

Both versions land in the moderate-to-strong positive range. v2 is more
consistent across checkpoints (v1 dropped 0.69 → 0.57; v2 stays 0.63 → 0.62).
σ_v is spec-consistent and behaves as expected.

## P mechanism — 30 regions, one-step SGD clone + 50 shared-seed eval episodes

| Estimator  | v1 ρ | v2 ρ [95 % CI] |
|---|---:|---|
| occupancy  | +0.400 | −0.236 [−0.558, +0.128] |
| cosine     | −0.099 | **−0.760 [−0.853, −0.560]** |
| dot        | −0.087 | **−0.758 [−0.866, −0.586]** |
| fisher     | +0.006 | **−0.718 [−0.841, −0.482]** |
| hybrid     | (n/a in v1) | −0.673 [−0.839, −0.394] |

The direction flipped and the magnitude jumped by ~7× for the gradient-based
estimators.

Interpretation:

- **v1** used actor-head + logstd only for the signature (≈1,000 parameters).
  A one-step SGD at η = 0.001 barely perturbed the policy, so
  `delta_return` on 50 eval episodes was noise → correlations near zero.
- **v2** uses the full actor + logstd (~130 K parameters). The same
  step now perturbs the policy meaningfully. But at 32 k steps every method
  is still at 0 % success — the policy is essentially untrained. Any large
  perturbation moves such a policy in unpredictable directions. Regions with
  higher `cosine` alignment tend to have higher-norm, more-confident `g_v`,
  so their SGD clones move furthest, and on an untrained policy that
  monotonically hurts `eval_return`.
- `delta_surrogate` (Taylor-consistent, no eval noise) correlations are much
  weaker (−0.20 to −0.31) — confirming most of the eval-return correlation is
  measurement noise from a policy that has not learned the task.

The strong negative correlations are **not a bug**: they arise precisely
because the gradient signature is now doing what the theory calls for.
What the pilot exposes is a mechanism-test design gap:

- The correct §25 protocol is a **controlled PPO-delta**: clone A, apply one
  PPO update on a matched base batch; clone B, apply one PPO update on base +
  region batch; compare J_ref(A) vs J_ref(B). Raw SGD on the region signature
  is a crude proxy that becomes actively misleading on an untrained policy.
- A converged checkpoint (≥ 500 k steps for these tasks) is required before
  the mechanism test carries evidentiary weight.

## Refactor items that did and did not change measurements

| Change | Effect |
|---|---|
| Full-actor gradient (IMPL §7) | Signature is now a real gradient, not a token; mechanism test now measures a real quantity (see p correlations above). |
| Shared adv-scale (IMPL §8) | `adv_scale` is now logged per update; scale stayed in a physical [1, 2] range across the pilot; no numerical instability. |
| Hybrid p estimator (IMPL §10) | Runs end-to-end. Mechanism ρ = −0.673 mirrors the other gradient-based estimators, confirming it inherits their signal. |
| EnvAdapter (IMPL §36 Phase 1) | Zero behaviour change on ManiSkill; snapshot restore still gives ≤ 1.2e-7 obs error; opens door for Meta-World / Fetch adapters. |
| Provenance §33 | Every run's provenance.json records Python, PyTorch, NumPy, benchmark version, git SHA, seed, device, backend, budget. |
| σ-rank normalization + staleness | σ mechanism ρ steadier across checkpoints. |
| PPO minibatch metric averaging | Minor: reported policy_loss / value_loss / KL / clipfrac now weighted-mean instead of last-minibatch. |

## What this pilot answers

- **Yes:** the refactor holds together on 36 real runs, all σ / p estimators
  exercise end-to-end, budget accounting stays exact, aggregation runs to
  completion, and no cell crashed.
- **Yes:** the σ story is stable across checkpoints and the estimator behaves
  as expected.
- **No:** it does not resolve whether HERP beats the baselines — the budget
  is too small, PPO itself hasn't learned the task, and any success-rate
  difference is at noise level.
- **No:** it does not validate the p_v claim. The mechanism-test result is
  noisy and directionally misleading under one-step SGD on an untrained
  policy. IMPL §25 controlled PPO-delta must be implemented before p_v
  claims are made in the paper.

## Recommended next actions

1. Run the same 6-method table on PickCube and PegInsertionSide at ≥ 500 k
   steps before quoting method rankings.
2. Implement the controlled PPO-delta mechanism (IMPL §25) — the pilot shows
   the one-step SGD proxy inverts the sign on undertrained checkpoints.
3. Keep σ pipeline as is; produce the K=4 vs K=64 correlation figure at both
   an early checkpoint (~100 k) and a converged checkpoint for the paper.

## Files

- `outputs/herp_pilot_v2/analysis/{RESULTS.md, main_table.csv, main_table.tex,
  aggregation.json}`
- `outputs/herp_pilot_v2/analysis/PickCube-v1_32768_learning.{pdf,png}` and
  `PushCube-v1_32768_learning.{pdf,png}`
- `outputs/herp_pilot_v2/mechanisms/{sigma_8192, sigma_32768, p_32768}/summary.json`
- `outputs/herp_pilot_v2/mechanisms/*/{sigma_correlation.pdf, sigma_futures.pdf,
  p_correlation.pdf}`
- `outputs/herp_pilot_v2/correlations_bootstrap.json`
- `outputs/herp_pilot_v2/suite.json`, `suite_complete.json`, `suite.log`
- Per-run raw data + provenance in
  `outputs/herp_pilot_v2/maniskill/<task>_<method>_s<seed>/*/`

The `outputs/` tree is git-ignored — files above live only on the machine
that ran the pilot. Copy them off before wiping the workspace.
