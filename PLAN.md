# HERP Experimental Plan

**Updated:** 2026-09-15  
**Scope:** final experiment campaign for the HERP paper  
**Primary constraint:** interaction training is much slower than initially expected, and some task/method pairs can remain at zero success for millions of transitions. The campaign therefore uses a staged **screen -> freeze -> promote -> full-seed** protocol instead of launching a full Cartesian grid.

The current server profile is CPU-heavy (approximately 88 CPU cores/vCPUs, 256 GB RAM, RTX 3060). The experiment schedule should exploit this asymmetry: **DMC runs on CPU in parallel while ManiSkill uses the GPU**, rather than leaving the CPU idle during GPU simulation.

---

## 1. Research questions

Every main-paper table/figure must answer one of four questions directly.

### RQ1 — Performance

**Does HERP improve downstream performance under a fixed environment-interaction budget?**

HERP is an interaction-allocation framework rather than a policy architecture. Evaluate it across:

- PPO,
- SAC,
- model-based RL (MBRL).

The claim is within-backbone: HERP-PPO vs PPO-based competitors, HERP-SAC vs SAC-based competitors, and HERP-TD-MPC2 vs TD-MPC2. The paper does not need to claim that one backbone is globally superior to another.

### RQ2 — Non-trivial resource allocation

**Does HERP produce a non-trivial distribution of interaction budget across behavioral regions rather than collapsing onto one region or staying effectively uniform?**

Primary metric: normalized allocation entropy over training.

### RQ3 — Contribution of relevance and dispersion

**Does the gain come from jointly using relevance `p` and dispersion `sigma`, or essentially from only one factor?**

Run a controlled allocation ablation with identical learner, partition, budget, and evaluation protocol.

### RQ4 — Quality of the sigma estimator

**Does the inexpensive online `sigma` estimator agree with an expensive high-sample Monte-Carlo oracle computed from additional frozen-policy rollouts?**

The oracle must estimate the **same fixed-window future-trajectory dispersion quantity** as HERP. Do not replace it with an unrelated full-network gradient-variance target.

---

## 2. Global experimental rules

### 2.1 Count raw environment interactions

All methods are compared using **raw environment transitions**.

All HERP acquisition, probing, and training-time reference interactions that influence training are charged to the interaction budget. Evaluation episodes are diagnostic and excluded from the training budget, but their protocol must be identical across compared methods.

For baselines with action repeat or native counters, convert to raw environment transitions before comparing budgets.

### 2.2 Equal-budget rule

Once a task and budget are frozen, every method in that comparison receives the same raw interaction budget. Do not early-stop a losing method in the final RQ1 matrix.

Early stopping is allowed only during pilot/screening.

### 2.3 Seeds

1. Pilot/screening: seed 0.
2. Final promoted cells: seeds 0, 1, 2.
3. If a key result has unusually high variance, extend **all relevant methods on that task** to seeds 3 and 4.

Report mean +/- standard deviation for the default three-seed results.

### 2.4 Evaluation cadence

Use a fixed cadence inside each environment family. A default of roughly 50k raw interactions is acceptable when evaluation does not dominate wall time. Screening can use 10 evaluation episodes; final manipulation endpoints should preferably use 50--100 episodes if affordable.

### 2.5 No task selection based on HERP winning

Task promotion and budget selection must use task informativeness, not HERP's relative rank. Freeze the cutoff before inspecting the final HERP-vs-baseline result.

---

## 3. Baseline matrix

### 3.1 PPO group

- `PPO`
- `PPO + RND`
- `PPO + Disagreement`
- `HERP-PPO`

### 3.2 SAC group

Where natively supported:

- `SAC`
- `RFCL (SAC)`
- `MaxInfoRL`
- `BRO`
- `HERP-SAC`

RFCL is a curriculum/allocation method built around SAC-style training; it is not a fourth backbone. Do not invent RFCL-PPO or RFCL-TD-MPC2 and present them as literature baselines.

BRO and MaxInfoRL should use locked/native implementations whenever possible.

### 3.3 MBRL group

- `TD-MPC2`
- `HERP-TD-MPC2`

DreamerV3 is intentionally omitted to reduce engineering and compute cost.

### 3.4 N/A is acceptable

The table does not need every method on every task. Prefer a faithful native baseline with an `N/A` elsewhere over an unofficial port created solely to fill a cell.

---

## 4. Environment and task suite

### 4.1 MetaWorld

Core task:

- `button-press-v3`

Use one MetaWorld task initially because CPU MuJoCo wall time is comparatively expensive. Primary metric: task success.

### 4.2 ManiSkill

Core/candidate tasks:

- `PickCube-v1`: reliable learning task; use a reduced pre-saturation budget. Primary task for manipulation-side RQ3/RQ4.
- `LiftPegUpright-v1`: medium/discriminative task; previous matrix produced strong method separation.
- `PushCube-v1`: low-budget candidate; old 5M setting saturated.

Reserve/challenge:

- `PokeCube-v1`: use only if a lower pre-saturation budget is identified.
- `PlaceSphere-v1`: challenge task after the method is frozen.

Do not use in the low-budget core unless a new pilot proves feasibility:

- `StackCube-v1`,
- `PegInsertionSide-v1`.

### 4.3 DeepMind Control Suite (DMC)

Use standard CPU `dm_control`; do not refactor to MJX/Warp for this paper unless profiling later proves necessary.

Core tasks:

- `walker-run`
- `cartpole-swingup_sparse`

Reserve:

- `cheetah-run`

DMC is now used not only in RQ1 but also as a **cross-domain mechanism validation suite for RQ2--RQ4**. This is attractive on the current CPU-heavy server because DMC simulation can run concurrently with ManiSkill GPU jobs.

Primary DMC metric: episodic return. Standard DMC tasks do not use manipulation-style success probabilities.

### 4.4 Fetch

Do not include Fetch in the first final campaign. Revisit only if the completed core suite is too narrow.

---

## 5. Task-budget calibration

### 5.1 Goal

Select `T_task` where the vanilla learner clearly learns but has not already saturated.

For manipulation success tasks, a useful heuristic is:

```text
0.2 <= vanilla success(T_task) <= 0.8
```

This is a heuristic rather than a hard requirement.

### 5.2 Initial pilot ranges

| Task | Initial screening range | Intent |
|---|---:|---|
| PushCube-v1 | 0.5M--1.5M | pre-saturation regime |
| PickCube-v1 | 0.5M--2M | learnable but not trivial |
| LiftPegUpright-v1 | 2M--5M | discriminative manipulation |
| PokeCube-v1 | 2M--5M | lower than old saturated 10M |
| PlaceSphere-v1 | up to 5M | challenge only |
| button-press-v3 | staged short pilot | MetaWorld wall-time control |
| walker-run | 0.1M, then up to ~0.5M if needed | cheap DMC |
| cartpole-swingup_sparse | 0.1M, then up to ~0.5M if needed | sparse exploration |
| cheetah-run | 0.1M, then up to ~0.5M if needed | reserve DMC |

### 5.3 Pilot kill criteria

Kill a pilot after a substantial fraction of its current cap if both are true:

1. success remains zero (or DMC return stays near its initial floor), and
2. dense/episodic return shows no meaningful upward trend.

Do not kill a sparse-success task if return/progress is clearly improving.

Kill immediately for NaNs, invalid restore, broken action bounds, exploding values, or inconsistent raw-step accounting.

### 5.4 Promotion criteria

Promote if:

- the vanilla learner shows a real learning signal,
- the task is not completely saturated at `T_task`,
- evaluation is stable enough for comparisons,
- HERP restore/region logic is correct,
- required baselines can run faithfully or unsupported cells are pre-declared `N/A`.

### 5.5 Preferred core

Target approximately five RQ1 tasks:

1. MetaWorld `button-press-v3`
2. ManiSkill `PickCube-v1`
3. ManiSkill `LiftPegUpright-v1`
4. DMC `walker-run`
5. DMC `cartpole-swingup_sparse`

Reserve: `PushCube-v1`, `PokeCube-v1`, `PlaceSphere-v1`, `cheetah-run`.

---

## 6. RQ1 — Main performance experiment

### 6.1 Main table

```text
PPO backbone
  PPO
  + RND
  + Disagreement
  + HERP

SAC backbone
  SAC
  RFCL
  MaxInfoRL
  BRO
  + HERP

MBRL backbone
  TD-MPC2
  TD-MPC2 + HERP
```

Columns are promoted tasks. Use `N/A` for unsupported combinations.

### 6.2 Metrics

Manipulation:

- primary endpoint: evaluation success at frozen interaction budget,
- log `success_at_once` and `success_at_end` when available.

DMC:

- primary endpoint: mean episodic return.

All tasks:

- retain learning curves,
- optionally compute AUC offline as a supplementary sample-efficiency diagnostic.

### 6.3 Fairness

Within each task/backbone comparison:

- identical raw interaction budget,
- same core learner settings for HERP vs vanilla backbone,
- same observation/reward protocol,
- same evaluation seeds/episodes,
- no method-specific budget extension after seeing results.

---

## 7. RQ2 — Does allocation collapse?

RQ2 is intentionally narrow. It tests whether the learned resource distribution is non-trivial/non-collapsed; it does not by itself prove semantic optimality.

### 7.1 Metric

At allocation decision `t`, let `q_t(v)` be the probability mass assigned to region `v` and `K_t` the number of eligible regions:

```text
H_norm(t) = - sum_v q_t(v) log q_t(v) / log(K_t)
```

Interpretation:

- `H_norm ~ 0`: collapse onto one/few regions,
- `H_norm ~ 1`: nearly uniform allocation,
- intermediate/changing values: structured preference without complete collapse.

Exclude or shade the root-only/warmup interval before the learned allocator is active.

### 7.2 Tasks: cross-domain rather than ManiSkill-only

Use four curves if all tasks pass screening:

**ManiSkill**
- `PickCube-v1`
- `LiftPegUpright-v1`

**DMC**
- `walker-run`
- `cartpole-swingup_sparse`

This tests whether non-collapse is a property of the allocator rather than an artifact of one simulator family.

If four lines make the main plot unreadable, use two panels:

- panel (a): ManiSkill,
- panel (b): DMC.

Do not add extra entropy variants, heatmaps, effective-region-count, or max-share to the main paper unless entropy is ambiguous.

### 7.3 Reuse rule

Do not launch dedicated RQ2 training if protocol-identical HERP runs from RQ1/RQ3 already log `q_t(v)`. Generate RQ2 from those logs.

---

## 8. RQ3 — Does joint `p x sigma` help?

RQ3 now uses one representative task from each simulator family.

### 8.1 Tasks

**Primary manipulation task**

- `PickCube-v1` at its frozen reduced budget.

**Primary DMC task**

- `cartpole-swingup_sparse` at its frozen pre-saturation budget.

Rationale: PickCube gives the agreed manipulation success metrics, while sparse Cartpole provides a cheap CPU-side test of whether the same `p x sigma` interaction matters in a qualitatively different exploration problem.

If sparse Cartpole fails screening, fall back to `walker-run`; do not force an all-zero ablation.

### 8.2 Methods

For each task, same backbone and same HERP region/reset machinery:

1. `HERP` — `p x sigma`
2. `herp_p` — `p` only
3. `herp_sigma` — `sigma` only
4. `uniform` — uniform allocation over eligible regions

Use `uniform` rather than vanilla PPO as the clean mechanism control because it preserves HERP acquisition machinery while removing learned prioritization.

### 8.3 Main-paper figure

Keep this compact under the 8-page limit.

Recommended three-panel figure:

- (a) PickCube `success_at_once` vs raw interactions,
- (b) PickCube `success_at_end` vs raw interactions,
- (c) DMC `cartpole-swingup_sparse` episodic return vs raw interactions.

Each panel has four curves: HERP / p-only / sigma-only / uniform, mean over seeds 0--2 with a light standard-deviation band.

If space becomes tight, keep panels (a,b) in the main paper and report the DMC ablation as a compact endpoint row/table plus full curve in supplementary material. The DMC experiment should still be run.

### 8.4 Decision gate

Start with seed 0 on both tasks. If full `p x sigma` is clearly broken relative to `p`-only across both domains, stop and fix the method before running the expensive RQ1 matrix.

Do not hide a p-only win.

---

## 9. RQ4 — Validate sigma against a high-sample oracle

### 9.1 Quantity being validated

HERP's online estimator measures dispersion of fixed-horizon trajectory features, not full-network gradient variance.

For each region:

```text
sigma_online(v) = sqrt(q_combined(v) + sigma_floor^2)
```

where `q_combined` is the online direct/predictor shrinkage quantity used by the allocator.

### 9.2 Oracle

At a frozen checkpoint:

1. freeze policy,
2. freeze partition/regions,
3. freeze feature normalizer,
4. select eligible regions,
5. restore snapshots,
6. collect many additional independent fixed-horizon fragments,
7. compute exactly the same fixed-window trajectory-dispersion statistic from the large pool.

```text
q_oracle(v)     = high-sample fixed-window future-trajectory variance
sigma_oracle(v) = sqrt(q_oracle(v) + sigma_floor^2)
```

Oracle interactions are diagnostic only and are never fed back into training.

### 9.3 Tasks: two-domain validation

Run oracle validation on:

**ManiSkill**
- `PickCube-v1`

**DMC**
- `walker-run`

`walker-run` is preferred over sparse Cartpole for RQ4 because it provides a rich continuous-control continuation distribution and high-sample rollouts are cheap on the CPU-heavy server.

This gives a stronger claim than validating the estimator on only one manipulation task.

### 9.4 Sampling protocol

Initial target per task/checkpoint:

- oracle pool: 128 future fragments per selected region,
- selected regions: approximately 20--30 including root,
- checkpoints: roughly 25%, 50%, 75% of the frozen task budget,
- seeds: 0--2 if affordable; at minimum all three seeds at the middle checkpoint.

For DMC, because rollout collection is CPU-cheap, increase the oracle pool to 256 if the 128-sample oracle remains visibly unstable. Do not increase ManiSkill automatically just because DMC can afford it.

Use the existing `scripts/collect_sigma_oracle.py` and `analysis/sigma_diagnostics.py` design; generalize the collector to DMC rather than inventing a different estimator.

### 9.5 Required saved values

For each selected region save:

- `sigma_raw`,
- `q_direct`,
- `q_pred`,
- `q_combined`,
- direct sample count,
- `q_oracle` / `sigma_oracle`,
- checkpoint, task, seed.

### 9.6 Metric and figure

Primary metric: Spearman rank correlation `rho`.

Main-paper figure: two panels if space allows:

- (a) PickCube: `sigma_oracle` vs `sigma_online`,
- (b) WalkerRun: `sigma_oracle` vs `sigma_online`.

Each point is one region; annotate `rho`.

If page space is tight, show one representative scatter plus a tiny table reporting `rho` for both tasks/checkpoints. Keep Pearson/Kendall/NDCG/bias-variance analyses in supplementary material.

Use the same valid-mask/common-window logic as training and log censoring/early termination.

---

## 10. Engineering prerequisites

Before spending compute on seeds 1--2:

### 10.1 PPO/HERP path

Smoke-test PPO, RND, Disagreement, HERP, p-only, sigma-only, and uniform on every promoted environment. Verify:

- raw steps,
- evaluation,
- no NaNs,
- snapshot restore,
- region growth,
- allocation probabilities sum to one.

### 10.2 SAC path

Verify vanilla SAC and HERP-SAC share the same core SAC update/replay settings within each direct comparison.

### 10.3 RFCL

Use a locked/native implementation on genuinely supported tasks. Prefer `N/A` over a large unofficial port.

### 10.4 BRO / MaxInfoRL

Use locked upstream checkouts through `scripts/run_external_native.py`. Verify raw DMC interaction accounting.

### 10.5 TD-MPC2 / HERP-TD-MPC2

Before making an MBRL modularity claim, verify that HERP is truly integrated into TD-MPC2's environment interaction/reset selection layer while the world model/planner/training remain unchanged.

### 10.6 MetaWorld

Verify the known CPU-simulator / CUDA-action-bound device issue before launching the final task.

### 10.7 DMC

Use CPU `dm_control`; keep GPU use minimal/optional for the learner. Verify HERP restore/region state serialization and raw-step accounting before starting DMC RQ2--RQ4.

---

## 11. Logging requirements

### 11.1 Common

At every evaluation log:

- method,
- backbone,
- task,
- seed,
- raw environment steps,
- eval return,
- success metrics if defined,
- wall-clock time,
- policy/update count,
- git/config/provenance.

### 11.2 HERP-specific

At allocation intervals log:

- active/eligible region count,
- allocation probability per region,
- normalized allocation entropy,
- `p_raw`, `p_ema`,
- `q_direct`, `q_pred`, `q_combined`,
- `sigma_raw`,
- sigma direct sample count,
- root allocation share,
- acquisition/probe/reference interaction counts,
- region creation/merge events.

### 11.3 Checkpointing

For both RQ4 tasks (`PickCube-v1`, `walker-run`), save approximately:

- 25% of `T_task`,
- 50%,
- 75%,
- final.

Each checkpoint must include policy, regions/archive, normalizer, predictor, allocator statistics, policy version, and counters.

---

## 12. CPU/GPU parallel execution on the current server

The current machine should be treated as two concurrent resources:

```text
RTX 3060  -> ManiSkill GPU simulation + learner
88 CPU cores/vCPUs -> DMC + MetaWorld + oracle jobs + analysis
256 GB RAM -> many independent CPU jobs / replay buffers / oracle pools
```

### 12.1 ManiSkill lane

Run one primary ManiSkill training job on the RTX 3060 initially. Tune `num_envs` downward from the previous 1024-env 5080/5090 configuration until GPU memory and simulator throughput are stable; likely screen 128/256/512 rather than assuming 1024.

Do not oversubscribe the 3060 with many simultaneous ManiSkill training cells unless profiling proves it increases aggregate transitions/sec.

### 12.2 DMC lane

DMC jobs are independent and should run concurrently with the ManiSkill GPU lane.

Start conservatively with several independent DMC processes rather than one process using all CPU threads. Set per-process BLAS/PyTorch thread counts low (typically `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and small `torch.set_num_threads`) so 88 cores are used across experiments rather than by nested thread pools.

Suggested initial schedule:

- reserve roughly 8--16 CPU cores/vCPUs for OS, logging, ManiSkill orchestration, and evaluation,
- use the remaining CPU capacity for independent DMC cells,
- increase concurrency only after measuring aggregate raw steps/sec and RAM use.

Do not assume the advertised 88 cores are 88 dedicated physical cores; benchmark actual throughput on the rented host.

### 12.3 Parallel campaign rule

Whenever a ManiSkill job is running on GPU, keep the CPU lane occupied with one of:

- DMC budget pilots,
- DMC RQ3 ablations,
- DMC RQ4 oracle rollouts,
- DMC RQ1 baselines,
- MetaWorld pilot,
- analysis/post-processing.

This is the default schedule unless CPU contention measurably slows ManiSkill `physx_cuda` throughput.

---

## 13. Execution order

### Stage A — preflight

1. Unit tests.
2. 50k--100k smoke runs for all adapters.
3. Raw-step accounting.
4. MetaWorld device fix.
5. DMC CPU restore/region path.
6. External baseline locks.
7. HERP-TD-MPC2 integration check.
8. Profile concurrent ManiSkill + DMC throughput on the rented machine.

### Stage B — budget pilots

In parallel:

- GPU lane: ManiSkill vanilla pilots,
- CPU lane: DMC vanilla pilots and MetaWorld pilot.

Freeze `task_budgets.json` without looking for a HERP-favorable cutoff.

### Stage C — RQ3 gate

Run both domains:

- GPU: PickCube HERP / p-only / sigma-only / uniform,
- CPU: sparse Cartpole HERP / p-only / sigma-only / uniform.

Seed 0 first. If the joint mechanism is plausible, run seeds 1--2 and freeze the method.

### Stage D — RQ4 sigma validation

Collect oracle diagnostics on:

- PickCube (GPU-assisted ManiSkill),
- WalkerRun (CPU DMC).

Run DMC oracle work concurrently with other GPU experiments whenever possible.

### Stage E — RQ2 entropy

Produce entropy curves from already completed protocol-identical HERP runs:

- PickCube,
- LiftPegUpright,
- WalkerRun,
- sparse Cartpole.

Dedicated RQ2 training should normally be unnecessary.

### Stage F — RQ1 seed-0 matrix

Run every frozen cell once. Use CPU/GPU lanes concurrently.

### Stage G — RQ1 final seeds

After Stage F passes, launch seeds 1--2. Extend to seeds 3--4 only for objectively ambiguous key comparisons.

### Stage H — reserve/challenge

Only after core claims are complete:

- PushCube/PokeCube low-budget variants,
- PlaceSphere,
- cheetah-run,
- additional MetaWorld,
- StackCube/PegInsertion only after a practical learning budget is demonstrated.

---

## 14. Compute-saving rules

1. No full-grid-first strategy.
2. No >5M manipulation task in the core suite without a pilot justification.
3. Three seeds by default, not five.
4. Reuse RQ1/RQ3 HERP logs for RQ2.
5. Reuse RQ1/RQ3 checkpoints for RQ4 whenever protocol-compatible.
6. Keep DMC on CPU and run it concurrently with ManiSkill GPU jobs.
7. Do not port unsupported baselines solely for a decorative table cell.
8. Do not run StackCube/PegInsertion blindly.
9. Do not let evaluation dominate training wall time.
10. Optimize **aggregate useful experiments/hour**, not utilization of a single GPU.

---

## 15. Main-paper outputs under the 8-page limit

### Table 1 — RQ1

One table:

- row groups: PPO / SAC / MBRL,
- columns: promoted tasks,
- cells: mean +/- std,
- `N/A` where appropriate.

### Figure 1 — RQ1 learning curves (optional)

One manipulation curve and one DMC curve if space permits.

### Figure 2 — RQ2 allocation entropy

Prefer two compact panels:

- ManiSkill: PickCube + LiftPegUpright,
- DMC: WalkerRun + sparse Cartpole.

### Figure 3 — RQ3 component ablation

Three compact panels:

- PickCube `success_at_once`,
- PickCube `success_at_end`,
- sparse Cartpole episodic return.

If page space is insufficient, move the DMC curve to supplement but retain its endpoint in text/table.

### Figure 4 — RQ4 sigma estimator

Two compact scatter panels (PickCube and WalkerRun) if readable. Otherwise show one scatter and a small table of Spearman correlations for both domains.

---

## 16. Decision logic

Desired evidence:

- RQ1: HERP improves fixed-budget performance across multiple tasks/backbones.
- RQ2: allocation does not collapse or remain permanently uniform across both simulator families.
- RQ3: `p x sigma` improves over p-only/sigma-only on at least a consistent subset and does not systematically degrade both domains.
- RQ4: online sigma positively rank-correlates with the high-sample oracle in both manipulation and DMC.

If RQ3 fails across both domains, revisit the combination rule/normalization before paying for the full matrix.

If RQ4 fails but RQ3 succeeds, sigma may still be a useful heuristic but cannot be claimed as an accurate estimator of the proposed dispersion quantity.

If RQ1 is mixed, report where HERP helps rather than selecting only winning tasks.

---

## 17. Repository artifacts

```text
outputs/final_campaign/
  pilots/
  rq1/
  rq2/
  rq3/
    maniskill/
    dmc/
  rq4/
    maniskill/
    dmc/
  task_budgets.json
  manifest.json
  decision_log.md

analysis/
  plot_rq1.py
  plot_rq2_entropy.py
  plot_rq3_ablation.py
  sigma_diagnostics.py
  make_main_table.py
```

Every run directory should contain provenance/config, metrics, summary, checkpoints when required, and console logs.

---

## 18. Immediate next actions

1. Benchmark the rented node: one ManiSkill job + several DMC jobs concurrently; record aggregate throughput.
2. Fix/verify DMC HERP adapter, restore semantics, and raw-step accounting.
3. Run vanilla budget pilots for ManiSkill and DMC **in parallel**.
4. Freeze `task_budgets.json`.
5. Run RQ3 seed 0 simultaneously: PickCube on GPU and sparse Cartpole on CPU.
6. If RQ3 is viable, run seeds 1--2 and freeze HERP.
7. Run RQ4 on PickCube and WalkerRun; exploit CPU capacity for the WalkerRun oracle.
8. Produce RQ2 from PickCube/LiftPeg + WalkerRun/sparse-Cartpole HERP logs.
9. Run RQ1 seed-0 matrix with CPU/GPU lanes simultaneously.
10. After all cells are valid, run final seeds 1--2.
11. Only then run reserve/challenge tasks.

The guiding principle is simple: **while the RTX 3060 is busy with ManiSkill, the CPU pool should be producing DMC evidence for the same paper.**