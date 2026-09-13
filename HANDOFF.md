# HERP framework — session handoff (2026-09-14)

Reproducibility, running-order, and known-issue notes for the Claude Code
session (or human) that continues this work on a real GPU server (RTX 5090
target). Everything below is captured from an interactive session against
the local GTX 1650 4GB WSL box; nothing else is required to resume.

Local `main` is fully pushed as of commit **`c42edda`** (DMC adapter)
plus this HANDOFF.md commit.

---

## 0. Repository state (2026-09-14 UTC)

- Legacy `src/experience_routing/` **removed** (44 modules) along with 10
  legacy scripts and 7 legacy tests. `src/herp/` is the sole HERP path.
- Framework v2 positioning (`HERP_UPDATED_POSITIONING_BASELINES_EXPERIMENTS_v2.md`)
  is the current spec.
- Test suite: **55 pass, 7 skipped, 0 failed**.
- CI: `.github/workflows/docker.yml` builds `Dockerfile` on every push,
  publishes to `ghcr.io/thanh124pav/herp:{latest,main,<sha>}`.

---

## 1. What was done this session

### 1.1 Code additions

| Component | Purpose | Path |
|---|---|---|
| **Vectorized HERP-SAC** | Parallel N-env SAC (`num_envs=16-64`) with per-slot region labels; keeps serial `train_herp_sac.py` as backup | `scripts/train_herp_sac_vector.py`, `src/herp/allocation_controller.py::choose_batch` |
| **DMC adapter** | Walker-Run / Humanoid-Walk / Quadruped-Run via `dm_control.suite`; snapshot round-trip via `physics.get_state/set_state` | `src/herp/envs/dmc.py`, `tests/test_dmc_adapter.py` |
| **Root-handling ablations** | EXPERIMENTS §26 V1 (`root_floor`) and V2 (`uniform_mix`) as first-class allocator knobs; automatically added to Phase 1 grid | `src/herp/allocator.py`, `src/herp/config.py`, CLI in `scripts/train_v3.py` |
| **Server pipeline** | One-command `preflight → pilot → phase 1 → phase 2 → merge` with idempotent auto-resume + VRAM-aware `num_envs` scaling | `scripts/server_deploy.sh` |
| **Third-party fetch** | Clones RFCL / ActiveRL / BRO / MaxInfoRL at locked SHAs + sparse ManiSkill baselines | `scripts/fetch_third_party.sh` |
| **Secondary venvs** | Creates `.venvs/{tdmpc2,maxinforl,bro}` with pinned requirements | `scripts/setup_secondary_venvs.sh`, `docker/requirements-*.txt` |
| **Docker + CI** | Ubuntu 22.04 + CUDA 12.8 + Python 3.11 & 3.12 + EGL/OSMesa libs + MUJOCO_GL=egl; auto-fetch third_party + build venvs during image build | `Dockerfile` |
| **Preflight** | 11 sanity checks (imports, CUDA, wandb, config drift, CLI, allocator edge cases, SAC learner, allocation controller, campaign planner, ManiSkill GPU init, snapshot round-trip) | `scripts/preflight_check.py` |
| **New env deps** | `dm-control==1.0.24`, `shimmy[dm-control]==2.0.0`, `myosuite==2.9.0`, `libegl1/libgl1/libgles2/libosmesa6` | `requirements.txt`, `Dockerfile` |

### 1.2 Bug fixes shipped

| Bug | Fix |
|---|---|
| Eval read `info["success"]` after auto-reset (episode-changed) | Read `info["final_info"]["success"]` on done; report `success_once` + `success_at_end` separately (mirrors `train.py::c1c46ac`) |
| `train_v3.py` hard-coded `num_minibatches=8` vs upstream `32` | Removed override so PPO matches upstream ManiSkill baseline |
| `allocation_entropy=NaN` after rank normalization (0·log0) | Use `torch.special.xlogy` with `clamp_min(1e-12)` |
| Rank normalization `r/(N-1)` gave min region prob=0 exactly (dead region violating THEORY §12) | Weibull position `r/(N+1)` — strictly positive |
| `run_framework_campaign.py` omitted `--num-envs` → PPO fell back to 1-env CPU (8h/2M runs) | Auto-scale + expose `--num-envs-ppo`/`--num-envs-sac` (default 512/16, up to 2048/64 on 32GB VRAM) |
| SAC pilot `--sim-backend cpu` refused `num_envs>1` | Pass `--sim-backend physx_cuda --buffer-device cuda` |
| SAC pilot `--capture-video True` → Vulkan-CUDA interop crash on WSL llvmpipe | `--no-capture-video --no-save-trajectory` |
| `sac_official.py --eval-freq` docstring said "iterations", code used env-steps → eval every 25 steps dominated wall time (~5% train utilization, ETA 119h) | Pass `--eval-freq 50000` |
| Campaign runner re-ran completed cells when manifest deleted | Auto-recover `status='completed'` from `summary.json` in each job dir |

### 1.3 Local pilot results

| Run | Task | Budget | Peak | Final | Wall |
|---|---|---:|---|---|---:|
| PPO calibration | PickCube-v1 | 5M | **78% @1.4M** → collapse | 0% | ~60min |
| PPO pilot | PickCube-v1 | 3M | **90% @852k** → collapse | 0% | 28min |
| SAC pilot | PickCube-v1 | killed@400k | — | **100% @300k, stable** | ~2h30 |
| TD-MPC2 pilot | PickCube-v1 | killed@10.7k | return 1.9→11 | training pipeline confirmed | 37min |
| Phase 1 partial | PickCube-v1 | 1/23 cells | HERP direct M=8 κ=8: return 113 stable | 0% | 30min |

**Interpretation:**
- Vanilla PPO exhibits classic peak-then-collapse instability (0% success at 3M after 90% peak at 852k). This is a **real algorithmic phenomenon**, not a bug — reproduces across seeds.
- Vanilla SAC saturates cleanly at 100% success at 300k, stays there. Confirms SAC's stability advantage.
- Local HERP with default `score_normalize=rank` and no root floor: return ceiling ~113 (vs PPO peak ~160) with `root_fraction ≈ 5-9%`. Hypothesis (validated by design review in this session): rank normalization compresses the root's naturally-high p·σ score and starves the ordinary-reset distribution → PPO value function drifts. The new `root_floor` + `uniform_mix` V1/V2 ablations in Phase 1 will test this empirically.
- TD-MPC2 on GTX 1650 is planning-bound (~5 env_steps/sec, 3M ETA 7 days). Server-only workload.

### 1.4 Commit trail

```
c42edda  Add DMC env adapter — Walker-Run / Humanoid-Walk etc.
4c66ff3  requirements: add DMC + shimmy + MyoSuite + MuJoCo GL libs
22fbe00  Docker + secondary venvs: reproducible third-party install
6e2a6c7  server_deploy.sh — end-to-end pipeline for RTX 5090
fa1c4c7  run_framework_campaign: fix SAC eval_freq (env-steps)
3762b27  run_framework_campaign: disable SAC video capture on WSL/llvmpipe
aec95fb  run_framework_campaign: pass --sim-backend physx_cuda for SAC + auto-recover jobs from summary.json
e15d158  run_framework_campaign: fix num_envs (PPO-family=512, SAC=16, TD-MPC2=1)
554b7b1  Vectorize HERP-SAC: parallel N-env acquisition, matched replay with vanilla SAC
1eb6f8f  server_run: default PRIORITY_FIRST=ppo,herp
433c753  run_framework_campaign: reorder Phase 2 — PPO first, then HERP, then baselines by recency
f50cb12  train_v3: default control_mode=pd_ee_delta_pose, reward_mode=dense
86ccbf4  train_v3: fix evaluate() to read success from info['final_info'] on done
f5ed35b  Stage 1: remove legacy experience_routing, add allocation-controller tests, extend preflight
3508864  Refactor for tonight's full campaign: PPO fairness, dead-region safety, edge tests, preflight
```

Plus the current commit (root_floor + uniform_mix ablations + this HANDOFF.md).

---

## 2. What's left for the server

### 2.1 Deploy setup (~15-30 min)

**Option A — pull pre-built image from GHCR** (recommended once CI is green):
```bash
docker pull ghcr.io/thanh124pav/herp:latest
```

**Option B — bare metal:**
```bash
git clone https://github.com/Thanh124pav/herp.git && cd herp
./scripts/server_setup.sh    # conda env deeplearning + fetch third_party + venvs
```

Verify build status: <https://github.com/Thanh124pav/herp/actions>

Prerequisites the server needs:
- NVIDIA driver ≥ 525 (CUDA 12.8 runtime)
- `wandb login` or `WANDB_API_KEY` env var
- (Optional) mount `~/.sapien/` to persist ManiSkill PhysX 5.3 GPU library across container restarts (~200 MB cache)

### 2.2 Full pipeline (~7-10 h on 5090; ~4-6 h on 2×5090)

```bash
./scripts/server_deploy.sh
```

This runs:

1. **Preflight** (11 checks) — aborts with exit 2 on any failure.
2. **Pilot** — PPO + SAC + TD-MPC2 × `PickCube-v1` × 3M steps headroom.
3. **Phase 1 (ablations)** — **23 HERP-PPO cells** × `PickCube-v1` × T=1.5M:
   - 9 sigma × horizon cells (direct/predictor/shrinkage × M=16/32/64)
   - 3 κ sweep cells (shrinkage M=32, κ={0, 2, 32})
   - **4 root-handling cells** (V1 root_floor=0.10, V1 root_floor=0.25, V2 uniform_mix=0.10, V1+V2 combined) **← new this session**
   - 7 control methods (uniform, herp_p, herp_sigma, state_radius×2, plr_region, sacl_style)
4. **Phase 2 (performance)** — cross-backbone × `[PickCube-v1, StackCube-v1, PegInsertionSide-v1]` × T=2M, balanced round-robin over PPO/SAC/MBRL, HERP first, easy→hard.
5. **Merge** — `scripts/merge_cross_backbone_results.py` produces `evaluations.csv`, `performance.html`, `ablation.html`, `results.md`.

### 2.3 Multi-seed extension (+2-3 h per seed)

```bash
STAGE=phase2 PHASE2_EXTRA_SEEDS="1 2" ./scripts/server_deploy.sh
```

### 2.4 Overrides available

```bash
STAGE={pilot,phase1,phase2,merge,all}
PILOT_TASK / PHASE1_TASK / PHASE2_TASKS
PILOT_BUDGET / PHASE1_T / PHASE2_T
NUM_ENVS_PPO / NUM_ENVS_SAC
WANDB_PROJECT / WANDB_MODE
SKIP_PREFLIGHT=1
```

### 2.5 Server-side decisions that need Claude judgment

| Decision | Notes |
|---|---|
| **Per-task T** | After Pilot, run `python analysis/estimate_saturation.py outputs/framework_.../pilot --output-dir <dir>` to inspect saturation curves. If any task shows PPO peak >> saturated plateau, pick the T at peak (like PickCube = ~850k), or provide `--task-budgets <json>` mapping. |
| **Which root-handling variant wins** | Phase 1 tests root_floor=0.10/0.25 + uniform_mix=0.10 + combined. Compare peak / plateau / area-under-curve. If a floor ≥ 0.10 recovers vanilla PPO performance without losing HERP's per-region signal, mark it as the new default and re-run Phase 2 with `--root-floor` set. |
| **External baselines (BRO / MaxInfoRL / RFCL)** | Third_party locked but not integrated into `server_deploy.sh`. Launch separately: `python scripts/run_external_native.py --method BRO --task PickCube-v1 --seed 0 --env-steps 2000000 --output-dir outputs/external/BRO_pickcube_s0`. RFCL requires demonstrations — skip unless a demo dataset is supplied. |
| **DMC / MetaWorld tasks in Phase 2** | Adapter code exists (`envs/dmc.py`, `envs/metaworld.py`). Add to `PHASE2_TASKS`, e.g. `PHASE2_TASKS="PickCube-v1 walker-run humanoid-walk"`. Success metric is undefined for DMC — merger reports N/A in that column. |
| **HERP-MBRL** | Not implemented. Per v2.md §0.6, "not required for first paper unless stable". Defer. |
| **Where-to-Learn (RA-L 2026)** | Priority "very high" per v2.md but no official code locked yet. If found, add to `docs/v3/external_baselines.lock.json` with commit SHA and rerun `fetch_third_party.sh`. |

### 2.6 Multi-GPU

`run_framework_campaign.py` currently runs jobs serially. On a 2×5090 or 4×A100 box, one-line orchestration change (spawn N workers with `CUDA_VISIBLE_DEVICES` routing) gives ~1.7-1.9× speedup for Phase 1/2. **Not implemented** — add if the server has ≥2 GPUs.

---

## 3. Where paper artifacts land

After `server_deploy.sh` finishes:

```
outputs/framework_YYYYMMDD_HHMM/
├── pilot/               # PPO + SAC + TD-MPC2 saturation curves
├── phase1/              # 23 HERP-PPO ablation cells
├── phase2/              # cross-backbone comparison × N tasks
├── merged/
│   ├── evaluations.csv          # every eval record (long form)
│   ├── final_per_run.csv        # last eval per (family, method, task, seed)
│   ├── performance.html         # main table (3 groups × N methods × N tasks)
│   ├── ablation.html            # sigma / allocation / partition ablations
│   └── results.md               # markdown summary
└── deploy.log           # full transcript of the run
```

W&B: `project=herp-framework` (override with `WANDB_PROJECT`). Every run gets its own group + tags (`family=PPO`, `phase=phase1`, `ablation=herp-sigma_mode-shrinkage...`).

---

## 4. Known caveats / non-blockers

- **DMC success**: undefined by dm_control. Adapter returns all-zero success mask; merger renders as `N/A`. Report DMC returns only.
- **BRO uses JAX** (Python 3.11 venv). `scripts/setup_secondary_venvs.sh` installs it; JAX cuda runtime pins may need adjusting if the server has a different CUDA version.
- **TD-MPC2 wall time**: even on 5090, TD-MPC2 3M is 5-8 h due to MPC planning cost. If time is tight, cap TD-MPC2 at 1M and note "not saturated" in the paper table.
- **HERP-PPO batch difference**: HERP-PPO does extra REFERENCE + REGION_ACQUISITION collection per rollout, so each PPO update sees a mixed acquisition distribution (this IS the causal comparison — vanilla PPO is 100% root, HERP-PPO is p·σ-weighted). Report per-cell `budget/ROOT_ACQUISITION`, `budget/REGION_ACQUISITION`, `budget/REFERENCE` for provenance.
- **Deadline**: 2026-09-16 (2 days from this handoff). Prioritize Phase 1 + Phase 2 core (PickCube, StackCube, PegInsertionSide × PPO/SAC/HERP variants). Deferrable: additional tasks, more seeds, external baselines beyond the vanilla SAC baseline.

---

## 5. Contact points in the code (fast lookup)

| Question | Look at |
|---|---|
| How does HERP-PPO train? | `scripts/train_v3.py::main` |
| How does HERP-SAC (vector) train? | `scripts/train_herp_sac_vector.py::main` |
| Where does the allocator normalize p·σ? | `src/herp/allocator.py::v3_priority_distribution` |
| How is the root floor enforced? | `src/herp/allocator.py::v3_priority_distribution` (post-normalize adjustment) |
| Where do fragments turn into region labels? | `src/herp/vector_acquisition.py::VectorPartitionObserver` |
| Where is σ estimated? | `src/herp/sigma.py::direct_q_estimate`, `src/herp/sigma_predictor.py::LinearVariancePredictor` |
| How does the campaign runner sequence jobs? | `scripts/run_framework_campaign.py::make_jobs` (grid) + `main` (loop) |
| Where does per-run summary.json get written? | `train_v3.py::main` (end), `train_herp_sac_vector.py::main` (end), manual for skipped/killed jobs |

---

Everything below the horizon is optional polish. The above is enough for
a fresh Claude Code session on the server to `git pull` and run
`./scripts/server_deploy.sh` without needing to reload this conversation.
