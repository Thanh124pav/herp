# HERP IMPLEMENTATION.md — Claude-CLI operating manual

> **Audience.** This file is a *prompt* for a Claude CLI session that will
> refactor and validate HERP on a GPU-capable server. Read it top-to-bottom
> **before** running any command or writing any code. All theory (definitions
> of `σ_v`, `p_v`, allocation rule, why HERP works) lives in `THEORY.md`;
> this file contains only instructions and code-shape contracts.
>
> **Model expectation.** Claude Opus 4.7 or Fable 5 (or newer). The refactor
> in §4 is ~1500–2500 lines of edits plus tests and must be done inside a
> single conversation on the server so the write-run-fix loop is tight.
>
> **You must not deviate from the file layout, adapter signatures, or test
> gates below without asking the user first.** External code
> (`docs/HERP_PILOT_V2_RESULTS.md`, `scripts/*.py`) references section
> numbers here; **keep the § numbering stable** even if you edit body text.

---

## §0. Session opening checklist

Before writing any code, do all of this:

1. `git status` — confirm the tree is clean. If not, stash or ask the user.
2. `git log --oneline -5` — you should be on `main` at or newer than
   `877ac9e` (`Merge pull request #4 …`).
3. `nvidia-smi` — confirm the target GPU is visible and idle. See §2.1 for
   accepted GPUs.
4. Read the three sibling docs once each: `THEORY.md` (math),
   `PLAN.md` (population-routing baselines, unrelated to HERP core but
   shares repo), and `README.md` (project pitch).
5. Read this file end-to-end, including the appendix.
6. Run the verification block in §2.8 and paste the output back into the
   session before starting §4. If any check fails, fix the environment
   before touching source.

Do **not** start editing on your own initiative before those six steps.

---

## §1. Starting state (as of 2026-09-10, commit `877ac9e`)

Working:

- `train.py` — single-env PPO training loop with HERP hooks (probing,
  region signatures, allocator). Uses `sim_backend=physx_cpu, num_envs=1`.
  ~64 steps/sec on an 8-core CPU.
- `src/herp/envs/maniskill.py` — `ManiSkillAdapter` for the 4 tabletop tasks
  in §7, single-env only.
- `src/herp/{allocator,archive,gradient_signature,regions,relevance,rollout_buffer,sigma,probe,eval,logging}.py`
  — HERP core, single-env semantics.
- `src/herp/baselines/{rnd,disagreement}.py` — intrinsic-reward baselines.
- `scripts/run_suite.py` — restartable (benchmark, task, method, seed) grid,
  each cell subprocess-invokes `train.py`.
- `scripts/ppo_official.py` — a patched copy of the upstream ManiSkill PPO
  baseline. Runs standalone; not yet integrated as the HERP backbone.
- `tests/test_herp_core.py`, `test_herp_regressions.py`,
  `test_checkpoint_qmp.py`, `test_vocabulary.py` — pass on single-env CPU.

Stubbed / broken:

- `src/herp/envs/metaworld.py` — every method raises `NotImplementedError`.
- `src/herp/envs/fetch.py` — every method raises `NotImplementedError`.
- Vectorized GPU sim: `physx_cuda` cannot be used on this repo yet because
  `train.py` assumes `num_envs=1`. The upstream ManiSkill vectorized env
  API (`num_envs=1024, sim_backend=physx_cuda`) is what the refactor in §4
  targets.
- The `experience_routing/` package is a separate SAC-based population
  pipeline (PLAN.md); it is **out of scope for this refactor**. Do not
  edit anything under `src/experience_routing/`.

Recent evidence:

- `docs/HERP_PILOT_V2_RESULTS.md` — 32k-step pilot, all methods still at
  ≈0% success. The σ mechanism test correlates 0.63±. The p mechanism test
  inverted sign under the single-step SGD proxy; the fix is the controlled
  PPO-delta protocol in §4.13.
- `docs/PPO_OFFICIAL_50K.md` — official PPO CPU-single-env at 50k steps
  also 0% success. The result is not an algorithm bug; it is a
  throughput bug. GPU sim is the fix.

---

## §2. Environment setup on a GPU server

### §2.1 Hardware requirements

- **GPU (required).** NVIDIA GPU with compute capability ≥ 7.0
  (Volta / Turing / Ampere / Ada / Blackwell). Recommended:
  RTX 3090 / 4090 / 5090 / A10 / A100 / H100. **≥ 12 GB VRAM** for
  `num_envs=1024`; **≥ 24 GB** for `num_envs=4096`. Consumer 4 GB cards
  (e.g. GTX 1650) do not run PhysX 5.3 GPU sim reliably — do not attempt.
- **Driver.** NVIDIA driver ≥ 550, CUDA runtime 12.1 or newer.
- **CPU / RAM.** ≥ 8 cores, ≥ 16 GB. WSL2 works but native Linux is
  preferred for GPU sim.
- **Disk.** ≥ 20 GB free for the checkpoint & video suite.

### §2.2 System packages (Ubuntu 22.04 baseline)

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential git curl unzip \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    libglfw3 libglew-dev libglvnd-dev \
    libvulkan1 mesa-vulkan-drivers
```

### §2.3 Python environment (choose one)

**Option A — conda (recommended, matches dev setup):**

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
bash /tmp/mc.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n herp python=3.12 -y
conda activate herp
```

**Option B — uv / venv:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### §2.4 Install PyTorch with CUDA

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0+cu128
```

Adjust `cu128` to your CUDA runtime (`cu121`, `cu124`, …) if needed. Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### §2.5 Install project dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins ManiSkill 3.0.1, Meta-World 3.1.1, gymnasium-robotics
1.5.0, mujoco 3.3.0. See that file for the full list.

### §2.6 Post-install downloads

ManiSkill needs SAPIEN's PhysX GPU binary and asset packs the first time
it starts a GPU sim:

```bash
python -c "
import mani_skill.envs, gymnasium as gym
env = gym.make('PickCube-v1', num_envs=4, sim_backend='physx_cuda',
               obs_mode='state', render_mode=None)
env.reset(); env.close()
print('ManiSkill GPU init OK')
"
```

That command:

- downloads `~/.sapien/physx/…` (PhysX 5.3 GPU library, ~200 MB, one-time),
- warms up asset caches,
- proves that `physx_cuda` initialises on this box.

If it prints `RuntimeError: CUDA failed` or a `PhysxGpuSystem` error, stop.
The GPU is not compatible; go to §2.9 fallback.

Meta-World assets are shipped with the wheel; no download.

Gymnasium-Robotics Fetch uses MuJoCo native meshes; no download.

### §2.7 Verify Meta-World and Fetch

```bash
python -c "
import gymnasium as gym
import metaworld
mt1 = metaworld.MT1('reach-v3', seed=0)
env = mt1.train_classes['reach-v3']()
env.set_task(mt1.train_tasks[0])
o, _ = env.reset()
print('Meta-World reach-v3 OK, obs', o.shape)
env.close()
"

python -c "
import gymnasium as gym, gymnasium_robotics  # noqa: F401
env = gym.make('FetchPush-v4')
o, _ = env.reset()
print('Fetch OK, keys', list(o.keys()))
env.close()
"
```

### §2.8 Full green-light block

Paste this exact block into your session before starting the refactor.
All four lines must print `OK`.

```bash
python -c "import torch; assert torch.cuda.is_available(); print('torch cuda OK')"
python -c "
import mani_skill.envs, gymnasium as gym
env = gym.make('PickCube-v1', num_envs=8, sim_backend='physx_cuda',
               obs_mode='state', render_mode=None)
env.reset(); env.close()
print('mani_skill GPU sim OK')
"
python -c "
import metaworld
mt1 = metaworld.MT1('reach-v3', seed=0)
env = mt1.train_classes['reach-v3']()
env.set_task(mt1.train_tasks[0]); env.reset()
print('metaworld OK')
"
python -c "
import gymnasium as gym, gymnasium_robotics  # noqa: F401
env = gym.make('FetchPush-v4'); env.reset()
print('fetch OK')
"
```

### §2.9 Fallback: GPU sim unavailable

If §2.6 or §2.8 second line fails despite following §2.1–§2.5:

1. Ask the user before touching the refactor. GPU sim is a hard
   prerequisite for §4; without it the refactor cannot be
   integration-tested.
2. If the user still wants you to proceed, restrict scope to §4.1–§4.4
   and §4.16 (interfaces + adapters + suite runner) and leave §4.5–§4.15
   for later. Mark that clearly in the PR description.

---

## §3. Refactor overview

**Goal.** Move HERP from single-env CPU sim (`num_envs=1, physx_cpu`) to
vectorized GPU sim (`num_envs≥1024, physx_cuda`) on ManiSkill, while
keeping the same HERP hooks (probes, region signatures, allocator) and
without regressing the σ-mechanism-test correlation reported in
`docs/HERP_PILOT_V2_RESULTS.md`.

**Non-goals.**

- Do **not** change `experience_routing/` (the SAC/UOT pipeline for PLAN.md).
- Do **not** rewrite the PPO algorithm inside `train.py` from scratch; port
  the official ManiSkill PPO backbone as the *inner* loop (§4.14).
- Do **not** batch Meta-World or Fetch; those adapters stay single-env but
  must fit the same interface (§4.1).

**Deliverables.**

1. Batched `EnvAdapter` interface (§4.1).
2. `ManiSkillAdapter` batched (§4.2).
3. `MetaWorldAdapter`, `FetchAdapter` implemented, single-env, same
   interface (§4.3, §4.4).
4. `train.py` rewritten around vectorized rollout + per-env-slot HERP
   modes (§4.5) with strict budget accounting (§4.6).
5. Region archive + probes work per-env-slot (§4.7, §4.8).
6. Reference batch collection batched (§4.9).
7. Gradient signature batched (§4.10).
8. All `p_v` estimators batched (§4.11).
9. σ mechanism test unchanged behaviourally on ManiSkill; new K=4 vs K=64
   correlation figure at 500k + 5M checkpoints (§4.12).
10. Controlled PPO-delta mechanism test (§4.13) — this fixes the sign flip
    reported in the v2 pilot.
11. PPO backbone matches upstream ManiSkill `ppo.py` in Agent architecture,
    init, and hyperparams (§4.14). RND / Disagreement re-baselined against
    the same backbone (§4.15).
12. Suite runner supports vectorized cells (§4.16).
13. Tests pass (§5). Snapshot restore is ≤ 1e-7 obs error per env slot on
    ManiSkill, ≤ 1e-6 on Meta-World and Fetch (§5.2).
14. Pilot v3 report at `docs/HERP_PILOT_V3_RESULTS.md` (§6) showing that
    HERP does not regress the σ-mechanism correlation and that PPO now
    solves PickCube-v1 well inside the 5M budget.

**Order.** Follow §4 top-to-bottom. Do **not** interleave — the interface
in §4.1 is a hard boundary that every later section depends on.

---

## §4. Refactor tasks

### §4.1 Batched `EnvAdapter` interface

File: `src/herp/envs/base.py`. Rewrite so every method is batched. Keep
the class name `EnvAdapter`; keep `benchmark: str` class attribute.

Required methods (all torch-tensor first; no numpy in the public API):

```python
class EnvAdapter:
    benchmark: str
    num_envs: int
    obs_dim: int
    action_dim: int
    max_episode_steps: int

    def make(self, num_envs: int, seed: int, **kwargs) -> "EnvAdapter": ...
    # Batched reset — returns obs of shape (num_envs, obs_dim) and a per-env info dict.
    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]: ...
    # Batched step. Actions and returns are torch tensors on self.device.
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # returns obs, reward, terminated, truncated, info
        ...
    # Snapshot API — save/restore a SUBSET of env slots.
    # env_ids is a LongTensor of slot indices (0..num_envs-1); return one
    # Snapshot per id.
    def save_state(self, env_ids: torch.Tensor) -> list[Snapshot]: ...
    def restore_state(self, env_ids: torch.Tensor, snapshots: list[Snapshot]) -> torch.Tensor:
        # returns the obs of the restored slots, shape (len(env_ids), obs_dim)
        ...
    # Feature extraction used by regionizer.
    def obs_tensor(self, raw_obs) -> torch.Tensor:
        # Called by the rollout loop; kept for parity with pre-refactor code.
        ...
    def region_features(self, obs: torch.Tensor) -> torch.Tensor:
        # Slice/transform obs → region-feature vector. Default: identity.
        ...
    def elapsed_steps(self) -> torch.Tensor:  # shape (num_envs,)
        ...
    def success_from_info(self, info) -> torch.Tensor:  # shape (num_envs,), 0/1
        ...
    def close(self) -> None: ...
```

`Snapshot` is a per-benchmark dataclass; keep it opaque outside the
adapter. It must be `copy.deepcopy`-safe and JSON-serialisable only via
the benchmark's own helper.

Single-env benchmarks (Meta-World, Fetch) implement this by holding an
internal list of `num_envs` MuJoCo envs and looping serially. That is
OK; the interface is what matters.

**Test gate.** Every adapter must pass `tests/test_env_adapter.py` (new;
see §5) with `num_envs ∈ {1, 4}` on the tasks listed in §7.

### §4.2 ManiSkill batched adapter

File: `src/herp/envs/maniskill.py`. Rewrite around
`ManiSkillVectorEnv` (from
`mani_skill.vector.wrappers.gymnasium.ManiSkillVectorEnv`), same wrapper the
upstream `ppo.py` uses.

Snapshot: use `env.unwrapped.get_state_dict(env_ids)` — ManiSkill 3
supports per-slot state dict extraction. Restore via
`env.unwrapped.set_state_dict(env_ids, dicts)`. Controller state and
`_elapsed_steps` must round-trip; verify with §5.2.

Control mode default: `pd_joint_delta_pos` (upstream PPO default). Do not
override to `pd_ee_delta_pose` unless the user asks — the choice affects
the PPO reference curves.

Obs mode: `state`. Reward mode: `dense`. `render_mode=None` unless the
suite runner is capturing eval videos.

### §4.3 Meta-World adapter

File: `src/herp/envs/metaworld.py`. Implement against Farama Meta-World
V3 (`metaworld==3.1.1`). Fall back to V2 only if V3 class is missing.

Tasks (§7):

- `button-press-v3`, `drawer-open-v3`, `pick-place-v3`, `peg-insert-side-v3`

Snapshot (dataclass `MetaWorldSnapshot`):

- `qpos`, `qvel` — numpy arrays from `env.data.qpos.copy()`,
  `env.data.qvel.copy()`.
- `mocap_pos`, `mocap_quat` — `env.data.mocap_pos.copy()`, `.mocap_quat.copy()`.
- `task_state` — `dict` of goal_pos / goal_quat / hand init / obj init
  (task-specific). Grab it from `env._get_pos_objects()` etc. and from
  `env._target_pos` when available. If the env exposes
  `get_env_state()`/`set_env_state()`, use those directly instead.
- `elapsed_steps` — int, from an internal counter you maintain (Meta-World
  does not expose one in a stable API).

Restore: `env.data.qpos[:] = snap.qpos`; `env.data.qvel[:] = snap.qvel`;
mocap likewise; `mujoco.mj_forward(env.model, env.data)`; restore any
task_state. Verify obs error ≤ 1e-6 with §5.2.

`num_envs > 1` is implemented as a list of independent envs stepped
serially. Do not attempt subprocess vectorisation.

### §4.4 Fetch adapter

File: `src/herp/envs/fetch.py`. Implement against
`gymnasium-robotics==1.5.0`.

Tasks (§7):

- `FetchPush-v4`, `FetchPickAndPlace-v4`

Obs is a dict `{observation, achieved_goal, desired_goal}`. Flatten as
`torch.cat([observation, achieved_goal, desired_goal])` into `obs_dim`.

Snapshot (dataclass `FetchSnapshot`):

- `qpos`, `qvel`, `mocap_pos`, `mocap_quat` — same as Meta-World.
- `desired_goal` — from `env.goal.copy()`.
- `elapsed_steps` — your counter.

Restore: same qpos/qvel/mocap procedure, then `env.goal = snap.desired_goal`.
Regenerate the obs by calling `env._get_obs()`. Verify §5.2.

`success_from_info(info)`: `info["is_success"]` when present; otherwise
compute `env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info) == 0`
(dense reward mode). Prefer the `info["is_success"]` flag.

### §4.5 Rollout loop with per-env modes

File: `train.py` main function. Replace the current single-env loop with
a vectorized loop over the ManiSkillAdapter. Every env slot at every step
has a **mode**:

- `NORMAL` — ordinary policy rollout, counted against the training budget.
- `PROBE` — a slot that has just been restored to a region snapshot and is
  running a short probe (`args.probe_horizon` steps) to sample `σ_v`.
- `ALLOCATED` — like NORMAL but its transitions carry a routing tag so
  they are used to update `p_v` via the controlled PPO-delta test (§4.13).
- `REFERENCE` — slots dedicated to the reference batch used by `p_v`
  (§4.9).

At each rollout iteration:

1. Choose how many slots each mode gets. Defaults:
   - 75% NORMAL, 15% PROBE, 10% ALLOCATED (a full ALLOCATED region is
     redistributed across the ALLOCATED slots according to the allocator).
   - REFERENCE runs in a separate small `num_envs_ref` vectorized env (~64)
     rather than stealing slots from the main rollout.
2. For PROBE slots, sample regions via `allocator.priority_distribution`,
   pull snapshots from the archive, restore, then step.
3. Collect all transitions into a single buffer with a `mode` column.

Budget accounting: increment `cumulative_normal_steps`, `cumulative_probe_steps`,
`cumulative_allocated_steps`, `cumulative_reference_steps` by the number
of **individual env transitions** in that mode. `sum(cumulative_*)` must
always equal `global_steps`, which is the total number of env transitions
consumed (across all slots and both the main and reference envs).

### §4.6 Budget accounting

Assertion at end of every rollout iter:

```python
assert cumulative_normal + cumulative_probe + cumulative_allocated + cumulative_reference == global_steps
assert global_steps <= args.total_timesteps
```

`global_steps` grows by `num_envs * num_steps + num_envs_ref * num_steps_ref`
per iter. This is the standard vectorized-env accounting and matches
upstream ManiSkill PPO.

The pilot v2 doc (§23) said the 32k pilot is not a final benchmark; keep
that spirit — the smallest useful budget for the *paper* is now 5M for
PickCube-v1 and 50M–75M for PegInsertionSide-v1, matching the upstream
recipe in `third_party/ManiSkill/examples/baselines/ppo/baselines.sh`.

### §4.7 Region archive + snapshots per env slot

File: `src/herp/archive.py`. Each `Region` still holds up to
`args.max_snapshots_per_region` snapshots. Extend to a batched save:
`archive.add(env_ids, snapshots, region_ids)`. Sampling for probes stays
per-region.

The archive is process-local; do not shard across GPUs.

### §4.8 Probe scheduling

Once per rollout iter, choose which PROBE slots restore to which region.
Priority weights come from `allocator.priority_distribution(regions, args)`.
Sample with replacement; multiple slots may probe the same region in
parallel. That is desirable when `p_v * σ_v` is concentrated on a few
regions.

`args.num_probes` is now the *total* PROBE slots per iter, not per region.
Adjust default to `int(0.15 * num_envs)`.

### §4.9 Reference batch

File: `src/herp/reference.py` (new). Reference envs run their own small
vectorized ManiSkill env (`num_envs_ref = 64`) that resets every episode
and is stepped by the *current* policy in deterministic mode. Its
trajectories feed the `p_v` cosine / dot / fisher / hybrid estimators.

The reference rollout does **not** update policy weights and does **not**
share buffers with the main rollout.

### §4.10 Gradient signature

File: `src/herp/gradient_signature.py`. Already uses full-actor +
logstd parameters (v2 refactor). Batched update: compute per-region
signature by looping over regions in a `torch.no_grad` block with autograd
for the small policy-gradient sub-graph; the outer batch dimension is
already handled.

Empirical Fisher diagonal (`empirical_fisher_diagonal`) also stays; the
input batches are now large enough that FIM approximation quality
improves.

### §4.11 `p_v` estimators

Files: `src/herp/relevance.py`, `src/herp/gradient_signature.py`. Keep the
five estimators (`occupancy`, `cosine`, `dot`, `fisher`, `hybrid`).
Nothing structural changes; only the batch sizes going in are bigger.

### §4.12 σ mechanism test

Compare `K=4` vs `K=64` oracle probe correlations at two checkpoints:

- Early: 500 000 env steps.
- Converged: 5 000 000 env steps (PickCube) or 50 000 000 (Peg).

Report ρ + bootstrap 95% CI. The pilot v2 baseline was 0.62 ± at 32k;
we expect ≥ 0.6 at the early checkpoint and ≥ 0.7 at the converged one.
Write results to `docs/HERP_PILOT_V3_RESULTS.md` (see §6).

### §4.13 Controlled PPO-delta mechanism (`p_v`)

File: `analysis/mechanisms.py`. Replace the single-step SGD proxy with
the controlled PPO-delta protocol from `THEORY.md`:

1. Clone the current policy → `π_A` and `π_B`.
2. On `π_A`: apply one full PPO update on a matched *base* batch
   (uniformly sampled main-rollout transitions).
3. On `π_B`: apply one full PPO update on `base ∪ region_batch`, where
   `region_batch` is transitions collected inside the target region only.
4. Evaluate both on the reference batch → get `J_ref(A)`, `J_ref(B)`.
5. `Δ_v = J_ref(B) − J_ref(A)`. This is the ground-truth quantity `p_v`
   is supposed to predict.

Sample size: ≥ 30 regions, ≥ 50 reference episodes per side.
Report ρ(`p_v`, `Δ_v`) + bootstrap 95% CI.

The old single-step SGD proxy in mechanisms.py **must be removed**, not
merely disabled. Its sign flip on undertrained policies is documented in
`docs/HERP_PILOT_V2_RESULTS.md`.

### §4.14 PPO backbone → official style

Import the network architecture and update loop from
`scripts/ppo_official.py` (a lightly patched copy of upstream
`ManiSkill/examples/baselines/ppo/ppo.py`). Specifically:

- `Agent`: 3-layer MLP width 256, `Tanh`, **orthogonal init** with the
  actor head at `std=0.01*sqrt(2)`. Do **not** keep the old Xavier init.
- Update: clip-vloss, `target_kl=0.1` early-stop, advantage normalisation,
  linear LR anneal from `args.learning_rate` to 0.
- Reward scale: `args.reward_scale = 1.0` default.
- No obs normalization (upstream default; matches pilot).

Keep HERP-specific extensions (buffer with `mode` column, `adv_scale`
logging, shared advantage scaler across NORMAL / PROBE / ALLOCATED) but
built on top of the upstream algorithm.

### §4.15 RND and Disagreement

Files: `src/herp/baselines/rnd.py`, `src/herp/baselines/disagreement.py`.
Re-baseline against the new PPO backbone. Intrinsic reward is added to
extrinsic reward at the *transition* level, weight `args.intrinsic_coef`.
RND target dim = 128; Disagreement ensemble size = 5. Update the
predictor every rollout iter for `args.intrinsic_epochs` epochs.

### §4.16 Suite runner

File: `scripts/run_suite.py`. Extend `--num-envs` and `--sim-backend` CLI.
Add a per-benchmark default:

- `maniskill` → `num_envs=1024, sim_backend=physx_cuda`
- `metaworld`, `fetch` → `num_envs=8, sim_backend=cpu` (serial vectorization)

The restart contract (skip a cell if its `complete.json` exists) is
already correct; do not change it.

---

## §5. Testing

### §5.1 Unit tests to keep

Do not delete existing tests unless a specific test asserts single-env
behaviour that no longer applies. Instead, generalise them to
`num_envs > 1`. The four existing files under `tests/` must all pass
after the refactor.

### §5.2 Snapshot restore precision

New file: `tests/test_env_adapter.py`. For each benchmark in §7 and each
of a small set of task IDs (one per benchmark is enough):

1. Reset the env.
2. Step for a random number of steps `k ∈ [0, 20]`.
3. Snapshot `env_ids = [0, 1, ..., num_envs-1]`.
4. Step for another `m ∈ [0, 20]` steps.
5. Restore all env_ids.
6. Compute `obs_after_restore - obs_before_restore` L∞ norm per env slot.
   Assert: ≤ 1e-7 for ManiSkill state obs, ≤ 1e-6 for Meta-World and
   Fetch (MuJoCo has slightly larger tolerance because of float ↔ double
   round-tripping).

Also assert: `elapsed_steps` after restore equals the step count at
snapshot time.

### §5.3 Batched-vs-single-env parity

New file: `tests/test_batched_parity.py`. Run one PPO rollout iter with
`num_envs=1` and `num_envs=8` on `PickCube-v1`, same seed, same policy
init. Assert that:

- The first env slot of the batched run matches the single-env run in
  observations for at least 200 steps (allow 1e-5 tolerance in reward).
- The optimizer step, if run on both, produces gradients whose cosine
  similarity ≥ 0.99.

This is the highest-value regression barrier for the whole refactor.

### §5.4 Test on the smallest possible GPU footprint

For CI or a quick smoke check, use `num_envs=8, sim_backend=physx_cuda`.
It exercises every code path without needing multi-GB VRAM budgets.

---

## §6. Pilot v3 validation

Once §4 and §5 are green, run:

- `scripts/run_suite.py --benchmarks maniskill --tasks PickCube-v1 PushCube-v1 --methods ppo rnd disagreement herp_sigma herp_p herp --seeds 0 1 2 --steps 5_000_000 --eval-interval 250_000 --eval-episodes 100 --workers 1 --device cuda --sim-backend physx_cuda --extra --num-envs 1024`

Wall clock estimate on RTX 4090: ~10 min per PickCube run, ~15 per
PushCube. 36 runs total → ~5 hours. Expect ≥ 80% success on PickCube for
all methods at ≥ 3M steps.

Write `docs/HERP_PILOT_V3_RESULTS.md` with the same structure as
`HERP_PILOT_V2_RESULTS.md` (main table, σ correlation, controlled
PPO-delta correlation, hyperparam sensitivity).

If any of the following fails, stop and ask the user:

- PPO baseline does not reach ≥ 80% on PickCube-v1 by 5M steps.
- σ mechanism ρ < 0.5 at either checkpoint.
- Controlled PPO-delta mechanism ρ is not significantly positive (95% CI
  crosses 0) for at least one gradient-based `p_v` estimator.

---

## §7. Full experiment matrix (target for the paper)

Benchmarks × tasks (final, do not add/remove without asking):

- **ManiSkill**: `PushCube-v1`, `PickCube-v1`, `StackCube-v1`,
  `PegInsertionSide-v1`.
- **Meta-World**: `button-press-v3`, `drawer-open-v3`, `pick-place-v3`,
  `peg-insert-side-v3`.
- **Fetch**: `FetchPush-v4`, `FetchPickAndPlace-v4`.

Methods (final):

Required: `ppo`, `rnd`, `disagreement`, `herp_sigma`, `herp_p`, `herp`.
Optional: `go_explore`, `plr` — do not block the paper on these.

Seeds: 0, 1, 2 (three seeds is the paper minimum; five is nicer if
compute allows).

Budget:

- ManiSkill: 5M for `PushCube-v1`, `PickCube-v1`, `StackCube-v1`; 75M
  for `PegInsertionSide-v1`.
- Meta-World: 2M per task.
- Fetch: 1M per task.

Total: 10 tasks × 6 methods × 3 seeds = **180 runs**. Reduce to 108
(6 tasks) if compute is tight — see §3 non-goals.

---

## §8. Reproducibility

### §8.1 Provenance

Every run writes `provenance.json` next to its `metrics.csv` with:

```json
{
  "python": "3.12.3",
  "torch": "2.11.0+cu128",
  "numpy": "…",
  "mani_skill": "3.0.1",
  "metaworld": "3.1.1",
  "gymnasium_robotics": "1.5.0",
  "mujoco": "3.3.0",
  "git_sha": "…",
  "git_dirty": false,
  "seed": 0,
  "device": "cuda:0",
  "sim_backend": "physx_cuda",
  "num_envs": 1024,
  "total_timesteps": 5000000,
  "hostname": "…",
  "started_at": "ISO-8601",
  "cli": [ "python", "train.py", …]
}
```

Do not skip any field. `docs/HERP_PILOT_V2_RESULTS.md` §33 was already
following this contract; keep it.

### §8.2 Configs

Every run must be reproducible from `configs/<benchmark>/<method>.yaml`
+ CLI overrides. Do not accept CLI-only runs for anything that ends up in
the paper.

### §8.3 Determinism

Set:

```python
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Full bit-exact determinism on GPU sim is not guaranteed by PhysX; note
that in the provenance's `notes` field if you observe reproducibility
drift > 1e-3 in eval return across seeds you thought were identical.

---

## §9. Instructions for the CLI when uncertain

- **Ask the user before**: (a) changing this file's § numbering,
  (b) editing anything under `src/experience_routing/`,
  (c) killing a long-running training job,
  (d) force-pushing or amending an existing commit on `main`,
  (e) deleting any `outputs/` directory that predates this session.
- **Just decide** when: (a) picking between two hyperparameters that
  match the upstream recipe within a factor of 2, (b) renaming a private
  helper, (c) fixing a lint / type error that the CI would catch anyway.
- **Never**: bypass `pre-commit` hooks (`--no-verify`), commit files
  under `outputs/` or `third_party/` or `.venv/` (they are gitignored;
  keep them gitignored), or hard-code paths that require WSL.

If a section of this document contradicts something you observe in the
code (e.g., a method name changed), assume the code is right in the
moment but *flag the contradiction to the user before you edit*.

Save recurring feedback to
`~/.claude/projects/-<slug>/memory/` per the auto-memory contract.

---

## Appendix A. Legacy anchor map (pre-refactor → this file)

External docs (`docs/HERP_PILOT_V2_RESULTS.md`, `scripts/*.py`) still
reference the pre-refactor section numbers. Where possible I kept the
same anchor topic in the same § so old references still resolve:

| Old anchor | Topic | New section |
|---|---|---|
| §7 | Gradient signature | §4.10 |
| §8 | Advantage normalization | §4.14 (inside PPO update) |
| §10 | `p_v` estimators | §4.11 |
| §23 | "32k pilot is not a final benchmark" | §4.6 note |
| §25 | Controlled PPO-delta mechanism | §4.13 |
| §27 / §28 | Estimator / hyperparameter ablations | §6 pilot v3 report |
| §29 | Standardised restore tests | §5.2 |
| §32 | Suite runner | §4.16 |
| §33 | Provenance | §8.1 |
| §36 Phase 1 | EnvAdapter | §4.1 |
| §36 Phase 2 | Batched adapter (this refactor) | §4.2 |
| §39 | Paper success criteria | §6 pilot-v3 gates |

New sections have no legacy anchor; they were added by this refactor
manual.

---

*End of IMPLEMENTATION.md. If you are the Claude CLI reading this, the
next thing you should do is run the §2.8 green-light block and paste
its output. Do not start §4 before that.*
