# HERP — Hindsight Experience Rollout Prioritisation

HERP is a **behavior-aware interaction allocation framework**: it chooses where
new environment interactions are collected; PPO/SAC remain the downstream
learners. The main specification is
[`HERP_UPDATED_POSITIONING_BASELINES_EXPERIMENTS_v2.md`](HERP_UPDATED_POSITIONING_BASELINES_EXPERIMENTS_v2.md).

The reproducible PPO core is `scripts/train_v3.py`. Native SAC is
`scripts/sac_official.py`; TD-MPC2 uses `scripts/tdmpc2_official.py` with its own
Python environment. See [`docs/FRAMEWORK_IMPLEMENTATION_STATUS.md`](docs/FRAMEWORK_IMPLEMENTATION_STATUS.md)
for validation status and outstanding gates. An entry point existing does not
mean that its learning-quality gate has passed.

`src/experience_routing/` is historical population replay-routing code, not
HERP-SAC. It is retained for old results and tests; new HERP code must not depend
on it. Historical notes below describe earlier revisions.

## Allocation rule

For each behavioral region, HERP estimates downstream policy-learning relevance
`p_v` and dispersion `sigma_v` of fixed-length future continuations. It allocates
new interactions according to `n_v ∝ p_v * sigma_v`. Reference and probing
interactions count toward the training budget. Evaluation interactions are
reported separately. Replay samples and imagined transitions are not raw
training interactions.

## Current state (2026-09-10)

Working (single-env CPU sim, `physx_cpu`, `num_envs=1`):

- `train.py` — PPO with HERP hooks (probing, region signatures, allocator).
- `src/herp/{envs/maniskill,allocator,archive,gradient_signature,regions,relevance,rollout_buffer,sigma,probe,eval,logging,baselines/{rnd,disagreement}}.py`
- `scripts/run_suite.py` — restartable (benchmark, task, method, seed) grid.
- `scripts/ppo_official.py` — patched copy of upstream ManiSkill PPO baseline.
- Tests: `tests/test_{herp_core,herp_regressions,checkpoint_qmp,vocabulary}.py`.

Not yet done (see `IMPLEMENTATION.md §4`):

- `src/herp/envs/metaworld.py` — stub (`NotImplementedError`).
- `src/herp/envs/fetch.py` — stub (`NotImplementedError`).
- Vectorized GPU sim: `train.py` assumes `num_envs=1`. The batched refactor
  targets `num_envs≥1024, sim_backend=physx_cuda` on a GPU-capable server.
- Controlled PPO-delta mechanism test (§4.13) — replaces the single-step
  SGD proxy that inverted sign on undertrained policies in pilot v2.

Recent evidence:

- `docs/HERP_PILOT_V2_RESULTS.md` — 32k-step pilot; σ mechanism ρ ≈ 0.62,
  every method still at ≈0 % success (budget too small; see the doc).
- `docs/PPO_OFFICIAL_50K.md` — 50k-step probe with the *upstream* PPO on
  CPU-single-env sim. Also 0 % success. Confirms the bottleneck is
  throughput (single-env CPU sim), not the algorithm. GPU sim on a proper
  box is the fix.

## Install

See [`IMPLEMENTATION.md §2`](IMPLEMENTATION.md) for the full install path
on a GPU server. Local quick start (CPU-only, for reading / unit tests):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
pip install -r requirements.txt
pip install -e .
```

If you do not have CUDA locally, drop the `--extra-index-url` line and the
`+cu128` suffix. Tests and small CPU pilots run on CPU-only torch.

## Run

Everything below assumes the `deeplearning` conda env (or the equivalent
venv from `requirements.txt`). Values reflect current single-env CPU
defaults; the GPU-sim refactor in `IMPLEMENTATION.md §4` will bump
`--num-envs` and add `--sim-backend physx_cuda`.

```bash
# One HERP run.
python train.py --benchmark maniskill --env-id PickCube-v1 --method herp \
    --seed 0 --total-timesteps 32768 --eval-interval 8192 --eval-episodes 50

# The suite (one cell per (task, method, seed)).
python scripts/run_suite.py --benchmarks maniskill \
    --tasks PickCube-v1 PushCube-v1 --methods ppo rnd disagreement herp \
    --seeds 0 1 2 --steps 5_000_000 --eval-interval 250_000 --eval-episodes 100

# Same suite with W&B. Use --wandb-mode offline when the server has no network.
python scripts/run_suite.py --benchmarks maniskill --tasks PushCube-v1 \
    --methods ppo rnd disagreement herp --seeds 0 --steps 5_000_000 \
    --wandb-mode online --wandb-project herp --wandb-group pushcube-5m \
    --wandb-tags maniskill,pushcube

# Baseline probe — upstream ManiSkill PPO, no HERP hooks.
MS_SIM_BACKEND=physx_cpu python scripts/ppo_official.py \
    --env_id=PickCube-v1 --seed=0 --num_envs=1 --num_eval_envs=1 \
    --num_steps=1024 --update_epochs=8 --num_minibatches=8 \
    --total_timesteps=50000 --eval_freq=1 \
    --no-capture_video --no-save_model --exp_name=probe
```

W&B is opt-in and disabled by default. Metrics are grouped under `eval/`, `ppo/`,
`rollout/`, `intrinsic/`, `budget/`, `herp/`, and `system/`. HERP region
distributions are logged as histograms at evaluation checkpoints. Checkpoints store the
W&B run ID, so `--resume-from checkpoint_*.pt` continues the same run. Each output
directory also contains `wandb.json` with the run identity.

Method names accepted by `train.py`:

    ppo, herp, herp_sigma, herp_p, rnd, disagreement, go_explore, plr

## Tests

```bash
pytest -q
```

Covers HERP core, regressions from pilot v2, snapshot/restore
round-trip, and vocabulary counting. `tests/test_env_adapter.py` and
`tests/test_batched_parity.py` are added as part of the refactor
(`IMPLEMENTATION.md §5`).

## Layout (HERP side)

```
train.py                     — PPO + HERP hooks entry point
requirements.txt             — pinned deps
IMPLEMENTATION.md            — Claude-CLI operating manual (install → refactor → tests)
THEORY.md                    — math (definitions, allocation derivation)
PLAN.md                      — separate SAC-population line (out of HERP scope)

src/herp/
  envs/{base,maniskill,metaworld,fetch}.py   — EnvAdapter, batched-first (metaworld/fetch stubbed)
  {allocator,archive,gradient_signature}.py  — HERP core
  {regions,relevance,rollout_buffer}.py
  {sigma,probe,eval,logging}.py
  baselines/{rnd,disagreement}.py

scripts/
  run_suite.py         — bounded, restartable grid runner (IMPL §4.16)
  ppo_official.py      — patched upstream ManiSkill PPO (IMPL §4.14)
  check_restore.py     — snapshot restore ≤ 1e-7 gate (IMPL §5.2)
  run_ablation.py, run_mechanisms.py, …
  run_ppo_official_50k.sh — 50k probe suite driver

analysis/
  aggregate.py         — post-suite aggregation
  mechanisms.py        — σ mechanism test; p mechanism (controlled PPO-delta) in IMPL §4.13
  plot_mechanisms.py

tests/                 — HERP unit + regression tests
docs/                  — pilot reports (v2, PPO-official 50k, reproducibility)
outputs/               — run artifacts (gitignored)
runs/                  — tensorboard logs from scripts/ppo_official.py (gitignored)
third_party/           — sparse-cloned upstream ManiSkill (gitignored)
```

## Where the PLAN.md pipeline lives

`src/experience_routing/` is a **separate** codebase for the SAC-based
donor→receiver routing project. It shares no code with HERP core and is
out of scope for the refactor described in `IMPLEMENTATION.md`. See
`PLAN.md` for its scope, and its own scripts:

```bash
python scripts/run_full.py --env synthetic --router uot        # PLAN.md line
python scripts/run_baseline.py --routers uot greedy td_priority
```

## Docs

- [`THEORY.md`](THEORY.md) — math (σ, p, allocation, why HERP works).
- [`IMPLEMENTATION.md`](IMPLEMENTATION.md) — Claude-CLI operating manual;
  the single source of truth for install, refactor tasks, tests,
  pilot-v3 validation, and reproducibility.
- [`PLAN.md`](PLAN.md) — separate SAC-population pipeline (not HERP).
- [`docs/HERP_PILOT_V2_RESULTS.md`](docs/HERP_PILOT_V2_RESULTS.md) — last
  before-refactor pilot.
- [`docs/PPO_OFFICIAL_50K.md`](docs/PPO_OFFICIAL_50K.md) — evidence that
  the bottleneck is CPU single-env sim, not the algorithm.
- [`docs/HERP_REPRODUCIBILITY.md`](docs/HERP_REPRODUCIBILITY.md) — the
  reproducibility contract inherited from pilot v2 (superseded piecewise
  by `IMPLEMENTATION.md §8`).
