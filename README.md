# HERP — behavior-aware interaction allocation framework

HERP chooses **where** new environment interactions are collected under a
fixed budget; PPO / SAC / MBRL remain the downstream learners. See
[`THEORY.md`](THEORY.md) for the full formal specification (partition,
`σ_v` estimator, gradient-alignment `p_v`, allocation `n_v ∝ p_v · σ_v`).

## Allocation rule

For every behavioral region `v`, HERP estimates:
- **relevance `p_v`** — gradient-alignment of region-v data with an
  ordinary-reset reference batch (THEORY §20-21);
- **dispersion `σ_v`** — square root of unbiased fixed-M future
  variance (THEORY §9-19).

Allocation is `n_v ∝ p_v · σ_v` with rank-normalization to prevent
either factor from swamping the other. The root region (ordinary reset
from `ρ_0`) participates as an ordinary candidate; an optional
`root_floor` guarantees a minimum share (EXPERIMENTS §26 V1).

## Repo layout

```
src/herp/                     HERP core: partition, archive, allocator,
                              controller, learners, env adapters
scripts/train_v3.py           HERP-PPO trainer (main entry point)
scripts/train_herp_sac_vector.py   Vectorized HERP-SAC (default)
scripts/train_herp_sac.py     Serial HERP-SAC (num_envs=1, backup)
scripts/sac_official.py       Vanilla SAC wrapper (ManiSkill upstream)
scripts/tdmpc2_official.py    TD-MPC2 wrapper (isolated venv)
scripts/run_external_native.py    BRO / MaxInfoRL runners
scripts/run_framework_campaign.py Master orchestrator (pilot/ablation/perf)
scripts/server_deploy.sh      One-command pipeline for the server
scripts/preflight_check.py    11-check GPU/CUDA/wandb readiness gate
tests/                        pytest suite (55 pass)
Dockerfile + .github/workflows/docker.yml   CI-built image (ghcr.io)
docs/                         experiment reports
```

## Running

**Docker (recommended):**
```bash
docker pull ghcr.io/thanh124pav/herp:latest
docker run --gpus all --rm \
    -v $HOME/herp_outputs:/workspace/herp/outputs \
    -v $HOME/.sapien:/root/.sapien \
    -v $HOME/.netrc:/root/.netrc:ro \
    ghcr.io/thanh124pav/herp:latest
```

**Bare metal:**
```bash
./scripts/server_setup.sh    # conda env + third_party + secondary venvs
./scripts/server_deploy.sh   # preflight → pilot → phase1 → phase2 → merge
```

Overridable env vars: `STAGE`, `OUTPUT_ROOT`, `SEEDS`, `NUM_ENVS_PPO`,
`NUM_ENVS_SAC`, `PILOT_TASK`, `PILOT_BUDGET`, `PHASE1_T`,
`PHASE2_TASKS`, `PHASE2_T`, `WANDB_PROJECT`, `WANDB_MODE`.

**Tests:**
```bash
pytest -q            # 55 pass, 7 skipped (GPU/dm_control conditional)
```

## Documentation

- [`THEORY.md`](THEORY.md) — formal spec (partition, σ, p, allocation, propositions)
- [`docs/FULL_MATRIX_REPORT_2026-09-11.md`](docs/FULL_MATRIX_REPORT_2026-09-11.md)
- [`docs/RESULTS_BOTH_METRICS_2026-09-11.md`](docs/RESULTS_BOTH_METRICS_2026-09-11.md)
