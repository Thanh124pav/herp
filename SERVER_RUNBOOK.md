# Server runbook — HERP framework paper pipeline

**Target reader:** Claude Code (or similar assistant) on the RTX 5090 server that
needs to reproduce the HERP paper tables. Every step below is copy-paste ready.
When in doubt, do exactly what this file says — do not improvise.

If a step's output does NOT match the "expected" bullet, JUMP to
[§ Troubleshooting](#7-troubleshooting) — do not continue.

---

## 0. Server prerequisites (one-time)

Verify:

```bash
docker --version                                    # >= 24.0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
```

**Expected:**
- Docker present.
- `nvidia-smi` prints an RTX 5090 (32760 MiB), or A100 (40960/81920 MiB), or similar CUDA-capable GPU with ≥ 16 GB VRAM.
- `docker run --gpus all` succeeds and prints GPU inside the container.

If any of these fail, STOP and ask the user. Do not proceed.

## 0.1. WandB auth (one-time)

The pipeline logs to a WandB project called `herp-framework`. You need one of:
- `~/.netrc` with wandb credentials (created by `wandb login` on any machine and
  copied to the server), OR
- environment variable `WANDB_API_KEY` set at container run.

Verify:
```bash
grep api.wandb.ai ~/.netrc && echo "netrc OK" || echo "no netrc — will need WANDB_API_KEY"
```

If no auth, use `-e WANDB_MODE=offline` on every `docker run` below and
manually `wandb sync outputs/framework_.../wandb/offline-run-*` later.

---

## 1. Pull the image

```bash
docker pull ghcr.io/thanh124pav/herp:latest
```

**Expected:** ~4 GB download, one layer. `docker images | head` shows the image.

If the pull 404s (image not published), fall back to source build:
```bash
git clone https://github.com/Thanh124pav/herp.git && cd herp
docker build -t herp:local .
IMAGE=herp:local   # use this everywhere below instead of the GHCR tag
```

Otherwise:
```bash
IMAGE=ghcr.io/thanh124pav/herp:latest
```

---

## 2. Prepare an output directory on the host

The container writes results to `/workspace/herp/outputs` (its working dir).
Mount a host dir to persist the results:

```bash
export HERP_OUTPUTS=$HOME/herp_outputs
mkdir -p "$HERP_OUTPUTS" "$HOME/.sapien"
```

`$HOME/.sapien` is where ManiSkill caches the PhysX GPU library (~200 MB on
first `physx_cuda` init). Mounting it means the cache survives container
restarts and every subsequent run boots ~1 min faster.

---

## 3. Full pipeline (the main command)

```bash
docker run --gpus all --rm \
    -v "$HERP_OUTPUTS:/workspace/herp/outputs" \
    -v "$HOME/.sapien:/root/.sapien" \
    -v "$HOME/.netrc:/root/.netrc:ro" \
    -e WANDB_PROJECT=herp-framework \
    "$IMAGE"
```

This runs `./scripts/server_deploy.sh` (the default CMD).

**What happens inside, in order:**

### 3.1 Preflight (~30 s)

`scripts/preflight_check.py` runs 11 sanity checks:

```
* imports              OK (torch X.Y.Z, mani_skill 3.0.1, wandb 0.23.0)
* cuda                 OK (NVIDIA GeForce RTX 5090, 32760 MiB)
* wandb login          OK (wandb login via ~/.netrc)
* config consistency   OK (M=32, radius=2.0, norm=rank)
* CLI parse            OK (all expected flags present)
* allocator edge cases OK ([...])
* SAC learner construction OK (signature dim=133636)
* allocation controller smoke OK (root fallback -> probs.sum=1.000)
* framework campaign planner OK (3 jobs planned)
* maniskill GPU init   OK (PickCube-v1 obs_dim=42 action_dim=8)
* snapshot roundtrip   OK (obs diff max<1e-6)
11/11 checks passed
```

If any check FAILs, the pipeline aborts here with `exit 2`. Do NOT re-run —
inspect the failed check's output and fix. Common:
- `wandb login`: no auth → set `WANDB_MODE=offline` (see §0.1).
- `cuda`: `--gpus all` not passed to docker run.
- `maniskill GPU init`: driver mismatch, or `~/.sapien` volume missing.

### 3.2 Pilot (~1-2 h on 5090)

**Purpose:** confirm PPO + SAC + TD-MPC2 all train something on the easiest
task (PickCube-v1). Budget 3M steps each. Determines saturation T candidate.

Runs sequentially: PPO first, SAC second, TD-MPC2 third.

**Expected outputs:**
```
outputs/framework_<STAMP>/pilot/
├── pilot-jobs.json                        # manifest, one row per job
├── PPO/PickCube-v1/ppo-seed0/
│   ├── config.json, provenance.json
│   ├── metrics.jsonl                       # every eval + training record
│   ├── summary.json                        # only present when the run is done
│   └── checkpoint_*.pt, checkpoint_final.pt
├── SAC/PickCube-v1/sac-seed0/
│   └── (same)
└── MBRL/PickCube-v1/tdmpc2-seed0/
    └── (same)
```

**Interpret `summary.json` for each:**
```bash
python3 -c "
import json
d = json.load(open('outputs/framework_<STAMP>/pilot/PPO/PickCube-v1/ppo-seed0/summary.json'))
print(f'PPO: steps={d[\"training_steps\"]}, return={d[\"final_evaluation\"][\"eval_return\"]:.2f}, success={d[\"final_evaluation\"][\"eval_success\"]:.3f}')"
```

**Expected on 5090:**
- **PPO**: peak `eval_success ≈ 0.90` somewhere between 400k-1M, then may
  collapse to 0.0 by 2M (known PPO instability, not a bug).
- **SAC**: `eval_success` reaches `1.0` at ~200-400k and stays there.
- **TD-MPC2**: `eval_success` reaches `~0.5-1.0` sometime in [500k, 3M];
  slower per step because of MPC planning.

If PPO peak stays 0 → training is broken; abort and inspect logs.
If SAC never learns → `--sim-backend physx_cuda` failed or CUDA-Vulkan interop
issue. Look for "Vulkan Device is not visible to CUDA" in the console log.

### 3.3 Phase 1 ablations (~7-13 h on 5090)

**Purpose:** ablate the σ estimator + allocation controls with HERP-PPO on
PickCube-v1 at budget T=1.5M. 23 cells total:

| Group | Cells | Question |
|---|---:|---|
| σ estimator × horizon | 9 | direct/predictor/shrinkage × M∈{16,32,64} |
| κ sweep | 3 | shrinkage M=32, κ∈{0,2,32} |
| **Root-handling (§26 V1/V2)** | **4** | root_floor∈{0.10, 0.25}, uniform_mix=0.10, V1+V2 |
| Baseline controls | 7 | uniform / herp_p / herp_σ / state_radius×2 / plr_region / sacl_style |

**Expected outputs:** `outputs/framework_<STAMP>/phase1/ablation/PPO/PickCube-v1/<cell>/`
same structure as pilot.

**Watch for:** root-handling cells (V1/V2) should recover most of vanilla PPO's
peak (0.9). If default HERP (root_floor=0) stays plateau at ~0.5 but
root_floor=0.15 hits 0.9 → confirms hypothesis and root_floor should become
Phase 2's default.

### 3.4 Phase 2 performance (~3-5 h on 5090)

**Purpose:** cross-backbone comparison at T=2M on 3 tasks:
`PickCube-v1`, `StackCube-v1`, `PegInsertionSide-v1`.

Order (balanced round-robin, HERP-first, easy→hard):
```
1. PPO herp     × PickCube          6. MBRL tdmpc2    × PickCube
2. SAC sac_herp × PickCube          7. PPO ppo        × StackCube
3. MBRL tdmpc2  × PickCube          8. SAC sac        × StackCube
4. PPO herp     × StackCube         9. MBRL tdmpc2    × StackCube
5. SAC sac_herp × StackCube        10. PPO herp       × PegInsertion
   ... (etc, 15 jobs total for 1 seed)
```

**Expected outputs:** `outputs/framework_<STAMP>/phase2/performance/{PPO,SAC,MBRL}/<task>/<method>-seed0/`

If TD-MPC2 wall time overshoots deadline, kill it and write a manual
`summary.json` marking it partial:
```bash
python3 -c "
import json, pathlib
p = pathlib.Path('outputs/framework_<STAMP>/phase2/performance/MBRL/PickCube-v1/tdmpc2-seed0/summary.json')
p.write_text(json.dumps({'status':'completed','method':'tdmpc2','env_id':'PickCube-v1',
    'seed':0,'training_steps':<STEPS_REACHED>,'early_stopped':True,
    'reason':'wall-time cap; report partial curve only'}, indent=2))"
```
The campaign runner will treat it as completed and move on.

### 3.5 Merge (~30 s)

Runs `scripts/merge_cross_backbone_results.py` over all three phase dirs.

**Expected outputs:**
```
outputs/framework_<STAMP>/merged/
├── evaluations.csv        # every eval record (long form)
├── final_per_run.csv      # last eval per (family, method, task, seed)
├── performance.html       # MAIN TABLE for the paper
├── ablation.html          # ABLATION TABLE for the paper
└── results.md             # markdown summary
```

Copy `performance.html` and `ablation.html` to your local machine to open in a
browser — those are the paper-ready tables.

---

## 4. Monitoring progress mid-run

The main `docker run` command streams stdout to your terminal. Prefer running
it inside `tmux` or `nohup` so a dropped SSH doesn't kill the run:

```bash
tmux new -s herp
# inside tmux
docker run ... "$IMAGE" 2>&1 | tee $HERP_OUTPUTS/deploy.stream.log
# detach: Ctrl-B then D
# reattach later: tmux attach -t herp
```

To peek at progress without touching the running container:

```bash
# Which stage is running and how far?
find $HERP_OUTPUTS -name summary.json 2>/dev/null | wc -l
# ^ counts completed cells across all phases.

# Latest metrics from every running cell:
for d in $(find $HERP_OUTPUTS -name metrics.jsonl -type f); do
    echo "=== $d ==="
    tail -1 "$d" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print({k:v for k,v in d.items() if k in ('type','step','eval_return','eval_success','wall_seconds')})"
done

# WandB dashboard:
echo "https://wandb.ai/$(cat ~/.netrc | grep -A1 wandb.ai | grep login | awk '{print $2}')/herp-framework"
```

The pipeline is **idempotent**: kill the `docker run` at any point, then
re-run the exact same command. It picks up from the last completed
`summary.json` — no work is redone.

---

## 5. Decision points that need judgment

The pipeline surfaces THREE decisions where a human (or capable assistant)
must inspect the data before proceeding:

### 5.1 After pilot: pick T per task

The pilot uses budget=3M for headroom. Real Phase 2 uses T=2M by default. If
the pilot shows any method saturates earlier (say 800k), Phase 2 can be
shortened for that task. Run:

```bash
docker run --rm -v "$HERP_OUTPUTS:/workspace/herp/outputs" "$IMAGE" \
    python analysis/estimate_saturation.py \
        outputs/framework_<STAMP>/pilot \
        --output-dir outputs/framework_<STAMP>/pilot/saturation
```

Output: `pilot/saturation/saturation.json` with per-run `candidate_T`.

**Decision rule:**
- If every method saturates by step S with tolerance 0.02 (sustained for 5
  evals): set `PHASE2_T = round(1.2 * S)` (20% headroom).
- If any method never saturates within 3M: keep default T=2M and note in the
  paper that "SOTA X did not saturate at 2M".

Re-run Phase 2 with the chosen T:
```bash
docker run ... -e STAGE=phase2 -e PHASE2_T=<chosen_T> "$IMAGE"
```

### 5.2 After Phase 1: which root-handling variant wins

Inspect `phase1/ablation/PPO/PickCube-v1/herp-*-root_floor-*-seed0/summary.json`
for cells:
- `root_floor=0.10`
- `root_floor=0.25`
- `uniform_mix=0.10`
- `root_floor=0.15 uniform_mix=0.10`

vs the baseline `herp-sigma_mode-shrinkage-future_horizon-32-sigma_kappa-8.0-seed0`
(default, root_floor=0).

**Decision rule:** the variant with highest `success_once` at T=1.5M becomes
Phase 2's default. Re-run Phase 2 with that variant:

```bash
# example: if root_floor=0.15 wins
docker run ... -e STAGE=phase2 \
    -e "HERP_EXTRA_ARGS=--root-floor 0.15" "$IMAGE"
```

(NOTE: `HERP_EXTRA_ARGS` is not wired into `run_framework_campaign.py` yet
— you may need to edit `scripts/run_framework_campaign.py::make_jobs` to
append `--root-floor 0.15` to the PPO `herp` command line, then re-run
Phase 2. Grep for `'--method',method,'--phase'` in that file — extra
flags go there.)

### 5.3 After Phase 2: are external baselines needed

If the paper needs BRO / MaxInfoRL / TD-MPC2 columns beyond what came from
the pilot + Phase 2, install the secondary venvs inside the container and
run them separately:

```bash
docker run --rm --gpus all -v "$HERP_OUTPUTS:/workspace/herp/outputs" \
    -v "$HOME/.sapien:/root/.sapien" "$IMAGE" \
    ./scripts/setup_secondary_venvs.sh
```

This installs `.venvs/{tdmpc2,maxinforl,bro}` inside the container. **The
venvs are ephemeral — they die with `--rm`**. Either drop `--rm` and keep
the container, or persist the venvs by mounting `-v $HOME/herp_venvs:/workspace/herp/.venvs`
BEFORE calling setup.

Then launch a specific external baseline:
```bash
docker run --rm --gpus all -v "$HERP_OUTPUTS:/workspace/herp/outputs" ... "$IMAGE" \
    python scripts/run_external_native.py --method BRO \
        --task PickCube-v1 --seed 0 --env-steps 2000000 \
        --output-dir outputs/external/BRO_pickcube_s0
```

Re-run the merger to pick up the new results:
```bash
docker run --rm -v ... "$IMAGE" \
    python scripts/merge_cross_backbone_results.py \
        outputs/framework_<STAMP>/pilot outputs/framework_<STAMP>/phase1 \
        outputs/framework_<STAMP>/phase2 outputs/external \
        --output-dir outputs/framework_<STAMP>/merged \
        --budget 2000000
```

---

## 6. Extending with more seeds

Once Phase 2 seed 0 is done, run seeds 1 and 2:

```bash
docker run ... -e STAGE=phase2 -e SEEDS="1 2" "$IMAGE"
```

This spawns 2 × 15 = 30 more jobs (all methods × all tasks × 2 seeds).
Wall time ~6-10 h on 5090.

After each extra seed batch, re-run merge (§3.5) — the merger will
group by seed and pick the last eval per `(family, method, task, seed)`.

---

## 7. Troubleshooting

### 7.1 Preflight fails

```
* wandb login          FAIL
    RuntimeError: no WANDB_API_KEY, no ~/.netrc; run `wandb login`
```
→ Either mount `~/.netrc` from a machine where `wandb login` was run, or
`-e WANDB_MODE=offline`.

```
* maniskill GPU init   FAIL
    RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver
```
→ NVIDIA driver too old (need ≥ 525). Update the host driver. Do NOT
try to patch the container.

### 7.2 SAC crashes with "Vulkan Device is not visible to CUDA"

The container defaults to headless MuJoCo (`MUJOCO_GL=egl`) but SAC's
`sac_official.py` may still try to open a rendering window. If you see this
after our fixes, override:
```bash
docker run ... -e MUJOCO_GL=osmesa "$IMAGE"
```
Slower but works everywhere.

### 7.3 A single cell in Phase 1 or 2 fails but you want the rest

`scripts/run_framework_campaign.py::main` currently RAISES on the first
failed job (`raise SystemExit(f'Failed job: {run}; inspect before continuing')`).
That's intentional so bad configs surface loudly.

To skip a broken cell:
1. Write a manual `summary.json` in that cell's output dir with
   `{"status":"completed","reason":"manually skipped: <why>"}`.
2. Re-run the exact same `docker run ...` command.
3. The auto-recover logic in `make_jobs` will treat that cell as
   completed and move on.

### 7.4 Out of GPU memory

If you see `CUDA out of memory` during Phase 1 or 2:
```bash
docker run ... -e NUM_ENVS_PPO=1024 -e NUM_ENVS_SAC=32 "$IMAGE"
```
(Halves the default 2048/64 batch — falls back to what was validated on
GTX 1650 4 GB.)

### 7.5 The merger renders "No completed evaluations" in a cell

That cell either hasn't finished yet or its `metrics.jsonl` never
contained an evaluation line. Check `find outputs -name metrics.jsonl -exec wc -l {} \;`
— a cell with < 5 lines probably crashed during init; look at its
`console.log` for the traceback.

### 7.6 GHCR pull returns 404

Either the CI build hasn't completed yet or the image is private.
Check <https://github.com/Thanh124pav/herp/pkgs/container/herp>. If
private, use the source-build fallback in §1. If the CI build is red,
inspect the failure at <https://github.com/Thanh124pav/herp/actions>
before doing anything else.

---

## 8. Definition of done

The pipeline is finished when ALL of the following hold:

- [ ] `outputs/framework_<STAMP>/pilot/pilot-jobs.json` — every job has
      `status: "completed"`.
- [ ] `outputs/framework_<STAMP>/phase1/ablation-jobs.json` — same.
- [ ] `outputs/framework_<STAMP>/phase2/performance-jobs.json` — same.
- [ ] `outputs/framework_<STAMP>/merged/performance.html` and
      `merged/ablation.html` exist and open in a browser.
- [ ] `outputs/framework_<STAMP>/merged/results.md` lists at least one row
      per (family, method, task) triple you want in the paper.
- [ ] Phase 2 has at least 1 seed. If time permits, 3 seeds for the
      cells that go into the main claim.

When done, tar and hand back to the user:

```bash
tar -czf herp_results_<STAMP>.tar.gz \
    outputs/framework_<STAMP>/merged \
    outputs/framework_<STAMP>/*/pilot-jobs.json \
    outputs/framework_<STAMP>/*/ablation-jobs.json \
    outputs/framework_<STAMP>/*/performance-jobs.json \
    outputs/framework_<STAMP>/deploy.log
```

---

## 9. Absolute don'ts

- **Do not** modify `src/herp/allocator.py`, `train_v3.py`, or any file
  under `src/herp/envs/` in the container. If a bug looks like it's in
  the code, STOP and ask a human — the code passed 55 tests locally.
- **Do not** delete `checkpoint_*.pt` files mid-run. If a cell dies, the
  campaign runner uses its last checkpoint to resume via `--auto-resume`.
- **Do not** run two full pipelines in parallel on the same GPU — they
  will fight for VRAM and both will collapse. Serialise via `tmux`.
- **Do not** rebuild the image on the server unless the GHCR pull fails
  — the CI build is the reference.
- **Do not** upgrade any Python package inside the container. Every pin
  in `requirements.txt` was chosen so the resolver produces a working
  install; loosening any of them re-enters the whack-a-mole from
  attempts 2-5 of the Docker debug.

---

**Quick reference (paste into terminal):**

```bash
# --- one-time ---
export HERP_OUTPUTS=$HOME/herp_outputs
mkdir -p "$HERP_OUTPUTS" "$HOME/.sapien"
docker pull ghcr.io/thanh124pav/herp:latest
IMAGE=ghcr.io/thanh124pav/herp:latest

# --- full pipeline (7-10 h on 5090) ---
tmux new -s herp
docker run --gpus all --rm \
    -v "$HERP_OUTPUTS:/workspace/herp/outputs" \
    -v "$HOME/.sapien:/root/.sapien" \
    -v "$HOME/.netrc:/root/.netrc:ro" \
    -e WANDB_PROJECT=herp-framework \
    "$IMAGE" 2>&1 | tee "$HERP_OUTPUTS/deploy.stream.log"
# Ctrl-B then D to detach. tmux attach -t herp to reattach.

# --- single stage re-run ---
docker run ... -e STAGE=phase1 "$IMAGE"
docker run ... -e STAGE=phase2 -e PHASE2_T=1500000 "$IMAGE"

# --- multi-seed ---
docker run ... -e STAGE=phase2 -e SEEDS="1 2 3" "$IMAGE"

# --- external baselines (optional) ---
docker run --gpus all -v "$HERP_OUTPUTS:/workspace/herp/outputs" \
    -v "$HOME/herp_venvs:/workspace/herp/.venvs" "$IMAGE" \
    ./scripts/setup_secondary_venvs.sh
# then a real run:
docker run --gpus all -v ... "$IMAGE" \
    python scripts/run_external_native.py --method BRO \
        --task PickCube-v1 --seed 0 --env-steps 2000000 \
        --output-dir outputs/external/BRO_pickcube_s0

# --- final merge (always re-run after adding results) ---
docker run --rm -v "$HERP_OUTPUTS:/workspace/herp/outputs" "$IMAGE" \
    python scripts/merge_cross_backbone_results.py \
        outputs/framework_<STAMP>/{pilot,phase1,phase2} \
        --output-dir outputs/framework_<STAMP>/merged --budget 2000000

# --- ship results ---
tar -czf herp_results.tar.gz outputs/framework_<STAMP>/merged \
    outputs/framework_<STAMP>/*/pilot-jobs.json \
    outputs/framework_<STAMP>/*/ablation-jobs.json \
    outputs/framework_<STAMP>/*/performance-jobs.json
```

That's it. Do not deviate.
