# HERP v3 local campaign

- Start: 2026-09-12 20:45 UTC (2026-09-13 03:45 Bangkok).
- Deadline: 2026-09-13 04:45 UTC (11:45 Bangkok), eight hours.
- User priority: ManiSkill; one seed across algorithms before additional seeds.
- Required runtime: `source ~/miniconda3/etc/profile.d/conda.sh; conda activate deeplearning`.
- Hardware: GTX 1650 4 GB, CUDA driver 12.8; PyTorch 2.11.0+cu128; ManiSkill 3.0.1; SAPIEN 3.0.3.
- Preserve user edits to THEORY.md / IMPLEMENTATION.md / EXPERIMENTS.md.
- Scope: HERP code only; legacy SAC experience-routing code is unrelated.
- Implement v3 beside explicitly retained legacy paths; reuse existing Agent and PPO optimizer.
- Report actual results, failed gates, shortened budgets and missing external baselines. No single-seed training confidence intervals.

## Specification decisions requiring explicit reporting

THEORY §20 applies positive clipping after EMA of signed cosine; IMPLEMENTATION §26 clips before EMA. Follow THEORY (signed EMA then positive floor) as mathematical source of truth.
Root total variance over child-entry continuations is a proposed proxy: child-entry distributions need not equal root-time conditional distributions. Measure its discrepancy; do not claim exact equality without aligned conditional sampling.
Main fixed-M resetting can limit exploration depth during warm-up. Track discovered snapshot depth and activation, and report any failure to reach long-horizon task stages.

## Plan

1. Simulator throughput and snapshot replay checks.
2. Chain partition, fixed-window dispersion, predictor, root allocator, fragment-local collector and tests.
3. Frozen-policy sigma and controlled PPO-delta diagnostics.
4. Seed-0 same-backbone pilot with shared predeclared task budget; external repositories audited separately.
5. Paper figures (PDF/SVG), LaTeX tables and readable report, provenance and resumable commands.

## Early checks (21:04 UTC)

- CPU simulator: ~200 raw steps/s; initial snapshot observation error 0.
- GPU simulator initialization fails at `PhysxGpuSystem` with `CUDA failed`.
- 10 new v3 tests pass; existing regression suite passes with 3 simulator-related skips.
- End-to-end 2048-step HERP smoke completed; success 0/2 (smoke only).
- First mechanism development run uses frozen first-batch normalization; retained as development evidence only. Main runner now uses running normalization and explicitly rebases centroids, recent raw-state futures and child means together to avoid mixing coordinate systems.
- Optimized full-window U-statistic via sample covariance, mathematically equivalent to pairwise form. Boundary history now uses a tensor ring buffer.

## Snapshot and GPU findings (21:18 UTC)

- `LD_LIBRARY_PATH=/usr/lib/wsl/lib` resolves PhysX CUDA initialization on this WSL host. GPU raw throughput: 64 envs ~448 steps/s; 512 envs ~4,717 steps/s. GPU instantaneous restore error ~5.96e-8.
- CPU non-contact replay: observation error 0, next observation error 5.36e-7, reward error 0, clock preserved.
- The first development runs deliberately stopped when a restored contact-state observation differed by 1.0. Diagnosis: ManiSkill's contact-derived `is_grasped` observation uses previous-step contact memory, which scene state restoration does not reconstruct. Snapshot now includes observation info as observable history; the next counted action recomputes contacts normally. Controller state is also explicitly applied because ManiSkill 3.0.1's `set_state_dict` does not apply its saved controller dictionary.
- Contact dynamics replay is an approximate-simulator-restoration question; it needs a separate measured replay check, not just matching cached observation bits. Failed development runs are excluded from main results.

## Late validation correction

The longer oracle exposed a second restore issue: re-archiving immediately after restore recomputed contact info from PhysX's still-stale contact cache. The adapter now preserves the info associated with the returned observation across restore -> save -> restore, and refreshes it only on a real step/reset. The first 327,680-step HERP run is development evidence and must be superseded by a corrected run in the main comparison. A vector-entry feature correction also zeros policy/state changes when no temporal predecessor exists, avoiding cross-reset discontinuities in predictor inputs.

Each subsequent run now archives its actual Python sources and SHA-256 hashes. Checkpoints and oracle pools are written atomically. The initial interrupted oracle NPZ is corrupt and is excluded; no reconstructed numeric evidence is used.
