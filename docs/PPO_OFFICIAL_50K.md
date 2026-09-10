# Official ManiSkill PPO — 50k-step probe (PickCube-v1, PegInsertionSide-v1)

**Date**: 2026-09-10  
**Purpose**: check whether the official ManiSkill upstream PPO (vs. the repo's
`train.py` PPO which was showing 0-success at 1M steps) reaches non-trivial
`eval_success` inside a 50k-step budget, on this WSL2 + GTX 1650 box.

## Setup

- **Script**: `scripts/ppo_official.py`, a lightly patched copy of
  `ManiSkill/examples/baselines/ppo/ppo.py` (sparse-cloned in
  `third_party/ManiSkill/`).
- **Patches** (all documented in git diff):
  - `sim_backend` selectable via `MS_SIM_BACKEND` env var; force
    `render_mode=None` and `render_backend=cpu` when sim is CPU.
  - Add `.to(device)` on `next_obs` / `eval_obs` / `reward` / `final_observation`
    because CPU sim returns CPU tensors while the policy lives on `cuda:0`.
  - Fix eval/save gating: `iteration % eval_freq == 1` → `(iteration-1) % eval_freq == 0`
    (the original was a silent no-op when `eval_freq=1`).
- **Backend**: `physx_cpu, num_envs=1`. GPU sim (`physx_cuda`) does not
  initialise on this GTX 1650 / WSL2 setup — `PhysxGpuSystem` raises a generic
  `RuntimeError: CUDA failed` even with the minimum memory config, so we could
  not use vectorised GPU sim here. Policy network still runs on `cuda:0`.
- **Hyperparams** (matched to upstream single-env sizing): `num_steps=1024`,
  `update_epochs=8`, `num_minibatches=8`, `learning_rate=3e-4`, `gamma=0.8`,
  `gae_lambda=0.9`, orthogonal init, `clip_vloss=False`, `target_kl=0.1`, no
  reward/obs normalisation (default), no LR anneal.
- **Tasks**: `PickCube-v1`, `PegInsertionSide-v1`.
- **Seeds**: 0, 1, 2 (3 per task).
- **Budget**: 50 000 env steps/run — 48 rollouts × 1024 steps.
- **Eval**: every rollout iteration (`eval_freq=1`), 500 env steps per eval
  (~10 episodes for PickCube 50-step, ~5 for Peg 100-step), deterministic
  policy.
- **Wall clock**: ~13 min/PickCube run, ~24 min/Peg run, 2 workers → **~63 min
  total**.
- **Output tree**: `outputs/ppo_official_50k/logs/{task}_s{seed}.log`,
  `eval_curves.csv`, `learning_curves.png`.

## Aggregate results (mean ± std across 3 seeds)

At the checkpoint nearest each target step:

| Task | Step | success_once | success_at_end | eval_return |
|---|---|---|---|---|
| PickCube-v1 | 10 240 | 0.000 ± 0.000 | 0.000 ± 0.000 | 7.20 ± 0.40 |
| PickCube-v1 | 20 480 | 0.000 ± 0.000 | 0.000 ± 0.000 | 7.84 ± 0.04 |
| PickCube-v1 | 30 720 | 0.000 ± 0.000 | 0.000 ± 0.000 | 8.20 ± 0.27 |
| PickCube-v1 | 40 960 | 0.000 ± 0.000 | 0.000 ± 0.000 | 9.12 ± 1.41 |
| PickCube-v1 | 48 128 | 0.000 ± 0.000 | 0.000 ± 0.000 | 10.99 ± 2.45 |
| PegInsertionSide-v1 | 10 240 | 0.000 ± 0.000 | 0.000 ± 0.000 | 5.46 ± 0.90 |
| PegInsertionSide-v1 | 20 480 | 0.000 ± 0.000 | 0.000 ± 0.000 | 6.66 ± 0.70 |
| PegInsertionSide-v1 | 30 720 | 0.000 ± 0.000 | 0.000 ± 0.000 | 7.25 ± 0.05 |
| PegInsertionSide-v1 | 40 960 | 0.000 ± 0.000 | 0.000 ± 0.000 | 6.90 ± 0.51 |
| PegInsertionSide-v1 | 48 128 | 0.000 ± 0.000 | 0.000 ± 0.000 | 7.45 ± 0.32 |

Across all 6 runs × 48 evals = 288 checkpoints, `success_at_end` was > 0 on
exactly **1 checkpoint** (PickCube s0 at 4 096 — `success_at_end=0.1`), and
`success_once` was > 0 on **4 checkpoints** total (all 0.1 = 1/10 eval
episodes). These are isolated flukes with no continuation.

**First step with `success_at_end > 0`, per seed:**

- PickCube-v1: seed 0 @ 4 096 (never again), seed 1 never, seed 2 never.
- PegInsertionSide-v1: all three seeds never.

## What this run answers, and what it does not

The question asked was: *when does PPO start solving?* At 50k CPU-single-env
steps, the answer is **not yet on either task**. But the run does show:

1. **Return is going up.** PickCube 3.2 → ~11 (~3.4×), Peg 2.3 → ~7.5 (~3.3×).
   The algorithm is learning something — dense reward shaping is pulling the
   policy toward the object / peg — but success gating is still failing.
2. **The 50k budget is compute-bound, not algorithm-bound.** The repo's own
   `train.py` PPO ran to 1M steps on PickCube (3 seeds) and also showed 0/144
   success. Upstream ManiSkill baselines report ~90% on PickCube-v1 after
   ~1–3M steps with **`num_envs=1024` on a GPU sim** — that is 3 orders of
   magnitude more parallel experience per wall-second than we can produce
   here. Peg upstream needs ~250M steps at 1024 envs.
3. **This box (WSL2 + GTX 1650 4 GB) cannot run `physx_cuda`.** The
   `PhysxGpuSystem` initialiser fails with `CUDA failed`. Until we move to a
   Volta/Ampere/Ada GPU (or a Colab / cloud instance with GPU sim working),
   we cannot answer the "when does PPO solve" question on this hardware
   inside a reasonable wall-clock budget.

## Recommended next actions

- **Do not extend budget on this machine.** 1M steps on `train.py` gave
  essentially the same signal as 50k on official PPO — success = 0. Both
  runs are throughput-bottlenecked on single-env CPU sim.
- **Move to a GPU-sim-capable machine** (Colab T4/L4, a Volta+ box, or a
  cloud A10/A100 instance). With `num_envs=1024, physx_cuda`, PickCube
  should show `success_at_end > 0.5` at ≈500k–1M steps, and PegInsertionSide
  probably at ≥50M steps. Same three seeds, `eval_freq=25` (default), gives
  a clean answer in <2 hours of wall time.
- **If we must stay on this box**, drop the per-checkpoint expectation and
  quote the ManiSkill upstream benchmark curves directly (their reported
  learning curves for PickCube-v1 and PegInsertionSide-v1 with 3 seeds are
  in the ManiSkill 3 paper / repo README).

## Files

- `outputs/ppo_official_50k/logs/*.log` — raw per-run stdout, one log per
  (task, seed) with `Epoch:`, `eval_*` lines, and SPS.
- `outputs/ppo_official_50k/eval_curves.csv` — flat CSV, all 288 evals.
- `outputs/ppo_official_50k/learning_curves.png` — 2×2 panel: return and
  success_once vs step for each task, mean±std ribbon across seeds plus
  thin per-seed traces.
- `scripts/ppo_official.py` — patched official PPO.
- `scripts/run_ppo_official_50k.sh` — 2-worker suite driver.
