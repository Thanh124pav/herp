# HERP Experimental Plan

**Date:** 2026-09-14  
**Scope:** final experiment campaign for the HERP paper  
**Primary constraint:** training is substantially slower than initially expected; some task/method pairs can remain at zero success for millions of interactions. The campaign therefore uses a staged **screen -> freeze -> promote -> full-seed** protocol instead of a full Cartesian grid from the start.

---

## 1. Research questions

The paper experiments are organized around four research questions. Every main-paper table/figure must answer one of them directly.

### RQ1 — Performance

**Does HERP improve downstream task performance under a fixed environment-interaction budget?**

HERP is an allocation framework rather than a policy backbone, so the main result must test it with multiple learner families:

- PPO backbone,
- SAC backbone,
- model-based RL (MBRL) backbone.

The claim is within-backbone: e.g. HERP-PPO vs PPO/RND/Disagreement, HERP-SAC vs SAC-based competitors, and HERP-TD-MPC2 vs TD-MPC2. The paper does **not** need to claim that one backbone is globally better than another.

### RQ2 — Non-trivial resource allocation

**Does HERP produce a non-trivial allocation of interaction budget across behavioral regions, rather than collapsing onto one region or behaving almost uniformly?**

Primary metric: normalized allocation entropy over training.

### RQ3 — Contribution of relevance and dispersion

**Does the gain come from jointly using relevance `p` and dispersion `sigma`, or is `p` alone sufficient?**

Primary experiment: controlled allocation ablation on one task with identical PPO learner, partitioning, budget, and evaluation protocol.

### RQ4 — Quality of the sigma estimator

**Does the inexpensive online `sigma` estimator used by HERP agree with an expensive high-sample Monte-Carlo oracle computed from additional frozen-policy rollouts?**

The oracle must estimate the **same fixed-window future-trajectory dispersion quantity** as HERP, not a different target such as full-network gradient variance.

---

## 2. Global experimental rules

These rules apply to every final comparison.

### 2.1 Count raw environment interactions

All training methods are compared under a fixed number of **raw environment transitions**.

For HERP, all acquisition/probing/reference interactions that affect the training procedure must be charged to the training interaction budget according to the implementation protocol. Evaluation episodes are diagnostic and are not part of the training budget, but they must be identical across methods.

For environments or baselines with action repeat, convert the native counter to raw environment transitions before comparing budgets. In particular, do not compare MaxInfoRL's internal update counter directly against another method's raw DMC step counter.

### 2.2 Equal-budget rule inside a comparison

Once a task and budget are frozen, every method in the same comparison receives exactly the same raw interaction budget. Do not early-stop a losing method in the final RQ1 table merely because it looks bad.

Early stopping is allowed only during **pilot/screening** runs used to determine whether a task is feasible.

### 2.3 Seeds

Use the following seed policy:

1. **Pilot/screening:** seed 0 only.
2. **Final promoted cells:** seeds 0, 1, 2.
3. If a promoted task has unusually high across-seed variance or the strongest comparison is statistically ambiguous, extend **all relevant methods on that task** to seeds 3 and 4 rather than selectively adding seeds to HERP.

For three-seed results, report mean ± standard deviation. Do not over-interpret a 95% CI from only three observations.

### 2.4 Evaluation cadence

Use a fixed evaluation cadence within each environment family. A reasonable default is every 50k raw interactions when this is not prohibitively expensive. Use enough evaluation episodes to reduce evaluation noise; 10 episodes is acceptable for screening, while 50–100 is preferable for final endpoint evaluation on manipulation tasks if wall time permits.

### 2.5 No task selection based on HERP winning

Task promotion must be based on **task informativeness**, not on HERP's relative rank.

A task is informative when the vanilla learner shows real learning but the task is not already saturated at the selected budget. Do not choose a cutoff after inspecting where HERP happens to look best.

---

## 3. Baseline matrix

The main performance table is grouped by learner family.

### 3.1 PPO group

Run:

- `PPO`
- `PPO + RND`
- `PPO + Disagreement`
- `HERP-PPO`

RND and Disagreement are important because they test whether HERP is doing more than adding a generic exploration/novelty signal.

### 3.2 SAC group

Run, where the native benchmark/task is supported:

- `SAC`
- `RFCL (SAC)`
- `MaxInfoRL`
- `BRO`
- `HERP-SAC`

Important interpretation:

- RFCL is a curriculum/allocation framework built around a SAC-style learner; it is **not** a fourth backbone.
- Do not create arbitrary RFCL-PPO or RFCL-TD-MPC2 variants and label them as literature baselines.
- BRO and MaxInfoRL should use their locked/native implementations whenever possible. Do not silently rewrite them into the HERP training code.

### 3.3 MBRL group

Run:

- `TD-MPC2`
- `HERP-TD-MPC2`

DreamerV3 is intentionally omitted from the final plan to reduce engineering and compute load.

### 3.4 N/A cells are acceptable

The main table does not need every baseline on every task. Some methods are only meaningful or natively supported on particular suites. Use `N/A` rather than porting a baseline solely to fill a rectangular table.

The strongest evidence is a faithful baseline on its native setting, not an unofficial reimplementation with uncertain tuning.

---

## 4. Environment and task suite

The campaign uses a small core suite plus reserve/challenge tasks.

### 4.1 MetaWorld

**Core task**

- `button-press-v3`

Reason: it has already shown relatively fast learning in local experiments, while MetaWorld itself is comparatively expensive. Keep only one MetaWorld task in the main campaign unless the pilot shows that another task is clearly cheaper/more informative.

Primary metric: task success rate. Episodic shaped reward may be logged as a learning diagnostic but is not the primary MetaWorld performance claim.

### 4.2 ManiSkill

**Core tasks**

1. `PickCube-v1`
   - reliable learning task,
   - useful at a reduced interaction budget before saturation,
   - main task for RQ3 and RQ4.

2. `LiftPegUpright-v1`
   - medium difficulty,
   - previous 5M-step seed-0 matrix showed strong method separation,
   - useful as the main discriminative manipulation task.

3. `PushCube-v1` — **low-budget candidate**
   - previous 5M runs saturated,
   - useful only if the budget is cut to the pre-saturation regime.

**Reserve task**

4. `PokeCube-v1`
   - previous 10M runs saturated,
   - may become informative at a substantially lower budget.

**Challenge task**

5. `PlaceSphere-v1`
   - strong exploration challenge in the old matrix,
   - keep out of the core table until the final HERP implementation is frozen because old results favored `p`-only and full HERP failed,
   - if used, run it as a clearly labeled challenge task with all promoted methods/seeds.

**Excluded from the low-budget main campaign**

- `StackCube-v1`: all methods were 0 at 5M in the previous matrix.
- `PegInsertionSide-v1`: too risky for the <5M compute regime.

These may be revisited only if later staged pilots establish a practical learning budget.

### 4.3 DeepMind Control Suite (DMC)

Use standard CPU `dm_control` for simplicity. Do not migrate to MJX/MuJoCo Warp for the current paper unless CPU simulation becomes the dominant wall-time bottleneck.

**Core candidates**

- `walker-run`
- `cartpole-swingup_sparse`

**Reserve candidate**

- `cheetah-run`

Rationale:

- DMC is relatively cheap state-based control,
- BRO/MaxInfoRL/TD-MPC2 are naturally comparable here,
- `cartpole-swingup_sparse` is particularly useful for an exploration/allocation story,
- DMC uses episodic return as the main performance metric; success probability is not defined for the standard tasks.

### 4.4 Fetch

Do **not** include Fetch in the first final campaign.

Reason: it adds engineering/domain breadth but has weaker baseline overlap than DMC/ManiSkill/MetaWorld. Revisit Fetch only if the core benchmark is too narrow after all promoted tasks are complete.

---

## 5. Task-budget calibration: the screening funnel

This stage is mandatory. It prevents the final matrix from spending millions of steps on all-zero or fully saturated tasks.

### 5.1 Goal

For each task, choose a fixed interaction budget `T_task` in the regime where learning is visible but not saturated.

Ideal endpoint regime for success-based manipulation tasks:

```text
0.2 <= vanilla success(T_task) <= 0.8
```

This is a heuristic, not a hard theorem. Values slightly outside this interval are acceptable if the learning curve is clearly informative.

### 5.2 Screening must not use HERP rank

Use vanilla backbones to determine whether a task/budget is feasible. The cutoff must be frozen before interpreting the new HERP-vs-baseline ranking.

### 5.3 Initial pilot ranges

These are **starting ranges**, not pre-declared final budgets. Select the actual cutoff from pilot curves.

| Task | Initial screening range | Intent |
|---|---:|---|
| PushCube-v1 | 0.5M–1.5M | find pre-saturation regime |
| PickCube-v1 | 0.5M–2M | retain learning signal without 5M saturation |
| LiftPegUpright-v1 | 2M–5M | medium/discriminative task |
| PokeCube-v1 | 2M–5M | replace old 10M saturated setting |
| PlaceSphere-v1 | up to 5M | challenge only |
| button-press-v3 | staged short pilot; stop before saturation | MetaWorld is wall-time heavy |
| walker-run | 0.1M then promote toward 0.5M if needed | cheap DMC |
| cartpole-swingup_sparse | 0.1M then promote toward 0.5M if needed | sparse exploration |
| cheetah-run | 0.1M then promote toward 0.5M if needed | reserve DMC |

### 5.4 Pilot kill criteria

A pilot can be killed after at least roughly half of its current screening cap if **both** are true:

1. task success remains exactly zero (or DMC return remains at the initial floor), and
2. the dense/episodic return shows no meaningful upward trend across the most recent evaluation windows.

Do **not** kill a sparse-success run when success is still zero but shaped return/task progress is clearly improving.

Also kill immediately for engineering failures such as NaNs, invalid state restore, broken action bounds, exploding values, or inconsistent step accounting.

### 5.5 Promotion criteria

Promote a task to final multi-seed experiments if:

- the vanilla learner shows reproducible learning signal,
- the task is not completely saturated at the chosen budget,
- evaluation is stable enough to compare methods,
- the environment/reset/restore mechanism required by HERP is correct,
- the required strong baseline can run faithfully on that task or an appropriate `N/A` policy is pre-declared.

### 5.6 Current preferred core after screening

Target approximately five main tasks rather than a large benchmark zoo:

1. MetaWorld `button-press-v3`
2. ManiSkill `PickCube-v1`
3. ManiSkill `LiftPegUpright-v1`
4. DMC `walker-run`
5. DMC `cartpole-swingup_sparse`

`PushCube-v1`, `PokeCube-v1`, `PlaceSphere-v1`, and `cheetah-run` are reserve/challenge tasks. Promote them only if a core task fails screening or if compute remains after the main matrix is complete.

---

## 6. RQ1 — Main performance experiment

### 6.1 Main table structure

Create one table with row groups:

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

Columns are the promoted tasks. Use `N/A` for unsupported baseline/task pairs rather than creating nonstandard ports.

### 6.2 Primary metrics

For manipulation tasks:

- primary endpoint metric: evaluation success at the frozen interaction budget,
- log both `success_at_once` and `success_at_end` where available,
- prefer the stricter/official task metric in the main table and retain the second success definition in supplementary diagnostics.

For DMC:

- primary endpoint metric: mean episodic return.

For all tasks:

- retain learning curves,
- compute return/success AUC offline as a supplementary sample-efficiency diagnostic; do not add another main-table metric unless endpoint performance is misleading.

### 6.3 Fairness

Within each task/backbone comparison:

- same raw environment interaction budget,
- same learner architecture and optimizer settings for HERP vs its vanilla backbone whenever HERP is integrated into that backbone,
- same evaluation episodes/seeds,
- same observation mode and reward mode,
- no method-specific budget extension after observing results.

### 6.4 Backbone-specific scope

The claim is modularity, not a fully dense benchmark grid.

A recommended efficient scope is:

- PPO group: all promoted core tasks.
- SAC group: ManiSkill tasks where SAC/RFCL are valid plus DMC tasks where SAC/BRO/MaxInfoRL are native.
- MBRL group: DMC plus any MetaWorld/ManiSkill task that the verified TD-MPC2 wrapper supports reliably.

### 6.5 Final-seed promotion

Run seed 0 for every frozen final cell first as a preflight. Only after all cells have valid logs and no engineering failure, launch seeds 1 and 2.

This seed-0 pass is **not** allowed to change the task/budget based on whether HERP wins; it only detects broken/infeasible cells.

---

## 7. RQ2 — Does allocation collapse?

RQ2 should stay intentionally narrow. It does **not** attempt to prove that allocation is semantically optimal; it only tests whether the learned resource distribution is non-trivial/non-collapsed.

### 7.1 Metric

At allocation decision `t`, let `q_t(v)` be HERP's probability mass assigned to eligible region `v`, and let `K_t` be the number of eligible regions.

Use normalized allocation entropy:

```text
H_norm(t) = - sum_v q_t(v) log q_t(v) / log(K_t)
```

Interpretation:

- `H_norm ~ 0`: allocation collapsed onto one/few regions.
- `H_norm ~ 1`: almost uniform allocation.
- intermediate, changing entropy: structured non-trivial preference without complete collapse.

### 7.2 Important implementation detail

Do not interpret the forced warmup/root-only phase as allocation collapse.

Start the RQ2 curve after HERP has enough non-root regions for the learned allocator to activate (`min_non_root_regions`), or visibly shade/exclude the warmup interval.

The root region should otherwise remain part of the eligible allocation distribution because the current HERP formulation treats root as an ordinary candidate.

### 7.3 Tasks

Use **three ManiSkill tasks** so the entire diagnostic shares one simulator and logging pipeline:

- `PushCube-v1` at its reduced budget,
- `PickCube-v1`,
- `LiftPegUpright-v1`.

If PushCube is removed during screening, replace it with `PokeCube-v1` only if PokeCube receives a valid low-budget cutoff.

### 7.4 Figure

One figure, one panel is sufficient:

- x-axis: raw training steps,
- y-axis: normalized allocation entropy,
- one line per task,
- mean over seeds 0–2,
- light uncertainty band across seeds,
- optional horizontal dotted line at 1.0 as the uniform reference.

Do not add heatmaps/effective-region-count/max-share to the main paper unless entropy alone becomes ambiguous. Those diagnostics can remain in supplementary logs.

---

## 8. RQ3 — Is `p × sigma` better than `p` alone?

### 8.1 Task

Use only:

- `PickCube-v1`

Use the reduced, pre-saturation budget chosen in the pilot. PickCube is intentionally used at a budget where it is learnable but not yet trivial.

### 8.2 Methods

Run four allocation variants with the same PPO backbone:

1. `HERP` — `p × sigma`
2. `herp_p` — `p` only
3. `herp_sigma` — `sigma` only
4. `uniform` — uniform allocation over eligible regions

Use `uniform` rather than vanilla PPO as the clean factor ablation because it preserves the HERP region/reset acquisition machinery while removing learned prioritization. Vanilla PPO already appears in RQ1.

### 8.3 Figure

One figure with two panels:

**(a) `success_at_once` vs raw training steps**  
**(b) `success_at_end` vs raw training steps**

Each panel contains the same four method curves. Plot mean over three seeds with a light standard-deviation band.

### 8.4 Claim discipline

RQ3 answers whether joint `p + sigma` allocation improves learning relative to its components. If full HERP does not beat `p`-only, do not hide the result; the method/story must be revised before submission.

Reward/return curves may be inspected while debugging, but the final RQ3 figure should use the two success metrics already agreed for PickCube.

---

## 9. RQ4 — Validate the sigma estimator against an oracle

### 9.1 What sigma is in HERP v3

The online estimator is based on dispersion of fixed-horizon **trajectory features**, not full-network gradient variance.

For each future fragment, the trajectory feature contains normalized next-state features plus weighted normalized action features over a fixed future horizon. The direct estimator computes fixed-window variance/equivalent pairwise squared distance and HERP converts the estimated variance quantity `q` into:

```text
sigma = sqrt(q + sigma_floor^2)
```

The full method uses shrinkage between direct measurement and a lightweight predictor.

### 9.2 Oracle definition

At a frozen training checkpoint:

1. freeze policy parameters,
2. freeze partition/regions,
3. freeze feature normalizer,
4. select a fixed set of eligible regions,
5. restore snapshots from each region,
6. collect many additional independent future fragments under the frozen policy,
7. compute **the same fixed-window trajectory-dispersion statistic** using the large pool.

Define:

```text
q_oracle(v)     = high-sample fixed-window future-trajectory variance
sigma_oracle(v) = sqrt(q_oracle(v) + sigma_floor^2)
```

These oracle rollouts are **diagnostic only** and must never be fed back into policy training or region updates.

### 9.3 Primary comparison should use the actual online estimator

For the final RQ4 claim, compare the actual checkpoint value used by HERP:

```text
sigma_online(v) = region.sigma_raw
```

against `sigma_oracle(v)` from additional frozen-policy rollouts.

This is preferable to inventing a different gradient oracle or evaluating only a re-fit estimator disconnected from what the allocator actually used.

### 9.4 Sampling protocol

Recommended default:

- task: `PickCube-v1`,
- backbone: PPO,
- oracle pool: `M = 128` future fragments per region,
- sampled regions: approximately 20–30 eligible regions per checkpoint, including root,
- checkpoints: 25%, 50%, and 75% of the frozen PickCube budget,
- seeds: run the oracle diagnostic for seeds 0–2 if affordable; at minimum run all three seeds at the middle checkpoint.

The current repository already contains `scripts/collect_sigma_oracle.py` and `analysis/sigma_diagnostics.py`; reuse them rather than creating an unrelated pipeline.

### 9.5 Required small analysis change

Extend the oracle collector/output to save, for each selected region:

- checkpoint `sigma_raw`,
- checkpoint `q_direct`,
- checkpoint `q_pred`,
- checkpoint `q_combined`,
- checkpoint direct sample count.

The main RQ4 scatter should use **checkpoint `sigma_raw`** vs high-sample oracle. The existing low-pool repeated-subsample/ridge analysis is useful as supplementary estimator diagnostics, but it should not replace validation of the actual online score.

### 9.6 Metric and figure

Primary metric: **Spearman rank correlation** `rho`.

Reason: HERP uses sigma primarily to rank/reweight regions; preserving the relative ordering is more important than reproducing the oracle magnitude exactly.

Main figure:

- x-axis: `sigma_oracle`,
- y-axis: `sigma_online`,
- each point: one region,
- annotate Spearman `rho`.

For multiple checkpoints/seeds, compute `rho` separately per checkpoint/seed and report mean ± std in the caption/text. Show one representative scatter in the 8-page main paper; retain additional scatters and Pearson/Kendall/NDCG/bias-variance analyses as supplementary material.

### 9.7 Censoring/early termination

Use the same valid-mask/common-window estimator as training. Log the fraction of censored fragments. Do not silently compare an oracle based only on fully surviving trajectories against an online estimator that includes common-window masked fragments.

---

## 10. Engineering prerequisites before launching the final matrix

Do these before spending compute on seeds 1–2.

### 10.1 PPO path

Already required/expected:

- PPO,
- RND,
- Disagreement,
- HERP-PPO,
- `herp_p`, `herp_sigma`, `uniform` for ablation.

Run a 50k–100k smoke test on every promoted environment and verify:

- environment steps increase correctly,
- evaluation executes,
- no NaNs,
- HERP can restore region snapshots,
- allocation probabilities sum to one,
- root/non-root region counts grow as expected.

### 10.2 SAC path

The repository has vanilla/native SAC and vectorized HERP-SAC paths. Before final experiments, verify that vanilla SAC and HERP-SAC use identical core SAC update/replay settings within a task comparison.

### 10.3 RFCL

Use an upstream/native RFCL implementation on tasks it genuinely supports and lock the upstream commit/version.

If a desired task is not natively supported and requires a major port or demonstration-generation change, mark the cell `N/A` rather than spending substantial engineering time just to fill the table.

### 10.4 BRO and MaxInfoRL

Use `scripts/run_external_native.py` with locked upstream checkouts. Verify raw DMC interaction accounting and export results to the same analysis schema as HERP.

### 10.5 TD-MPC2 / HERP-TD-MPC2

Vanilla TD-MPC2 wrapper exists. Before claiming MBRL-backbone generality, verify whether a real `HERP-TD-MPC2` collection/allocation integration is present.

If it is not yet implemented, this is a **blocking engineering task**:

- keep TD-MPC2's world model/planner/optimization unchanged,
- insert HERP only at the environment interaction/reset-region selection layer,
- count all HERP acquisition/probe interactions,
- preserve TD-MPC2 replay/training settings,
- validate equality with vanilla TD-MPC2 when allocation is forced to the ordinary-root behavior.

Do not label vanilla TD-MPC2 alone as evidence that HERP works with MBRL.

### 10.6 MetaWorld device check

The 2026-09-11 full-matrix report found a CPU-simulator / CUDA-action-bound mismatch. Before any MetaWorld campaign, run a short preflight that explicitly tests action clamping/device placement. Do not launch the full matrix until this passes.

### 10.7 DMC CPU path

Use CPU `dm_control`. GPU remains available to the learner network. Verify that external baselines and HERP report the same raw interaction definition.

---

## 11. Logging requirements

Every run should write enough information to reproduce the final table/figures without re-running training.

### 11.1 Common fields

Log at every evaluation:

- `method`
- `backbone`
- `task`
- `seed`
- `raw_env_steps`
- `eval_return`
- `success_at_once` if defined
- `success_at_end` if defined
- wall-clock time
- policy/update count
- provenance: git commit, external-baseline commit, command, config

### 11.2 HERP-specific fields

At allocation intervals log:

- number of active/eligible regions,
- allocation probability `q_t(v)` for every region,
- normalized allocation entropy,
- `p_raw`, `p_ema`,
- `q_direct`, `q_pred`, `q_combined`,
- `sigma_raw`,
- sigma direct sample count,
- root allocation share,
- number of acquisition/probe/reference interactions,
- region creation/merge events if applicable.

These logs make RQ2/RQ4 possible without repeating main training.

### 11.3 Checkpointing

For PickCube runs used by RQ4, save checkpoints at approximately:

- 25% of `T_PickCube`,
- 50%,
- 75%,
- final budget.

A checkpoint must include policy, archive/regions, normalizer, predictor, allocator statistics, policy version, and counters.

---

## 12. Execution order

Do not launch all experiments at once. Use the following order.

### Stage A — preflight

1. Unit tests.
2. 50k–100k smoke runs for each environment adapter.
3. Verify raw-step accounting.
4. Verify MetaWorld device fix.
5. Verify DMC CPU pipeline.
6. Verify external BRO/MaxInfoRL locked repositories.
7. Verify RFCL task compatibility.
8. Verify/implement HERP-TD-MPC2 before including MBRL claims.

**Gate:** no final experiment starts until all required promoted paths pass.

### Stage B — budget pilots

Run only vanilla backbones, seed 0, on candidate tasks. Determine and freeze `T_task` using the pre-saturation rule.

Output:

- `task_budgets.json`
- one pilot plot per task
- short decision log containing `promote / reserve / reject` and the reason.

**Do not use the new HERP relative result to choose the cutoff.**

### Stage C — RQ3 first

Before paying for the entire RQ1 matrix, run the cheap/high-information PickCube ablation:

- HERP,
- p-only,
- sigma-only,
- uniform.

Start with seed 0. If full `p × sigma` is clearly broken relative to `p`-only, stop and fix the method before running the expensive cross-backbone campaign.

If behavior is plausible, launch seeds 1–2 and freeze the method implementation.

This stage is the most important compute-saving gate.

### Stage D — RQ4 sigma validation

Using the frozen PickCube HERP implementation, collect oracle rollouts and validate that `sigma_online` has meaningful positive rank correlation with `sigma_oracle`.

If correlation is weak/negative, debug the estimator before paying for full RQ1.

### Stage E — RQ2 entropy diagnostic

Run/log the three ManiSkill task curves. This is cheap once the RQ1/RQ3 runs already log allocation distributions; reuse existing runs whenever protocol-identical.

### Stage F — RQ1 seed-0 matrix

Run all frozen RQ1 cells with seed 0. This checks interoperability of all baselines/backbones/tasks under the final protocol.

Cells can be removed only for objective engineering/incompatibility reasons declared before inspecting HERP's rank.

### Stage G — RQ1 final seeds

Launch seeds 1–2 only after Stage F is clean.

If a task has pathological variance, extend every relevant method on that task to seeds 3–4.

### Stage H — reserve/challenge tasks

Only after the core claims are complete, spend remaining compute on:

- PushCube/PokeCube if not already core,
- PlaceSphere challenge,
- cheetah-run,
- any additional MetaWorld task,
- StackCube/PegInsertion only if a practical learning budget has been demonstrated.

---

## 13. Compute-saving rules

1. **No full-grid-first strategy.** Pilot one seed, then promote.
2. **No >5M manipulation task in the core suite** unless a pilot proves the extra budget is scientifically necessary.
3. **No five seeds by default.** Start at three; extend only ambiguous promoted comparisons.
4. **Reuse runs across RQs.** RQ2 entropy should come from RQ1 HERP runs when the task/protocol matches. RQ4 uses checkpoints from the same PickCube HERP runs.
5. **Do not train an extra method just for a decorative table cell.** Use `N/A` for unsupported literature baselines.
6. **Keep DMC on CPU simulation.** Avoid MJX/Warp refactoring unless profiling proves simulator CPU time dominates.
7. **Do not run StackCube/PegInsertion blindly.** Old results already show the low-budget danger.
8. **Do not let evaluation dominate training wall time.** Keep frequent evaluation small during training; run a larger endpoint evaluation once per final checkpoint.

---

## 14. Main-paper outputs under an 8-page limit

The experiments section should be visually compact.

### Table 1 — RQ1 main performance

One main table:

- row groups: PPO / SAC / MBRL,
- columns: promoted tasks,
- cells: mean ± std across final seeds,
- bold best within the relevant comparison group,
- `N/A` for unsupported method-task combinations.

Do not add separate main tables for every backbone unless the table becomes unreadable.

### Figure 1 — learning curves / qualitative overview (optional)

If space permits, show one or two representative learning curves from RQ1. This figure is lower priority than RQ2–RQ4 because Table 1 already answers performance.

### Figure 2 — RQ2 allocation entropy

One plot:

- x: raw training steps,
- y: normalized allocation entropy,
- lines: ManiSkill tasks.

### Figure 3 — RQ3 component ablation

One figure, two panels:

- (a) PickCube `success_at_once`,
- (b) PickCube `success_at_end`,
- four lines in each panel: HERP / p-only / sigma-only / uniform.

### Figure 4 — RQ4 sigma estimator

One scatter:

- x: high-sample oracle sigma,
- y: online HERP sigma,
- annotate Spearman rho.

Additional oracle checkpoints, bias/variance sweeps, Pearson/Kendall/NDCG, entropy heatmaps, and reserve tasks go to supplementary material or the repository report.

---

## 15. Expected decision logic

The campaign should support the following honest conclusions depending on results.

### Desired outcome

- RQ1: HERP improves fixed-budget performance across multiple backbones/tasks.
- RQ2: allocation entropy stays away from both complete collapse and permanent uniformity.
- RQ3: full `p × sigma` improves over `p`-only and `sigma`-only.
- RQ4: online sigma has strong positive rank correlation with the high-sample oracle.

### If RQ3 fails

If `p`-only consistently matches or beats `p × sigma`, do **not** proceed as if sigma is validated. Revisit the combination rule/normalization or reduce the paper claim before spending the full RQ1 compute budget.

### If RQ4 fails but RQ3 succeeds

Then sigma may still be a useful heuristic but cannot be claimed as an accurate estimator of the proposed dispersion target. Either fix the estimator or weaken the interpretation.

### If RQ1 is mixed

A modular framework does not need to win every task, but the final paper needs a consistent story. Report failures and characterize when HERP helps rather than selecting only winning environments after the fact.

---

## 16. Concrete run checklist

Before starting a run batch, answer these four questions:

1. **Which RQ does this batch answer?**
2. **What is the cheapest experiment that can answer it?**
3. **What objective criterion promotes/kills the batch?**
4. **Will the result be reused by another RQ?**

If none of the four RQs needs the run, do not launch it.

---

## 17. Suggested repository artifacts

Create/maintain the following outputs:

```text
outputs/final_campaign/
  pilots/
  rq1/
  rq2/
  rq3/
  rq4/
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

Every run directory should contain:

```text
config/provenance
metrics.csv or metrics.jsonl
summary.json
checkpoint(s) when required
console.log
```

---

## 18. Immediate next actions

Execute in this order:

1. Confirm all environment adapters and raw-step accounting with short smoke tests.
2. Run vanilla budget pilots for the proposed core tasks.
3. Freeze `task_budgets.json`.
4. Run **RQ3 PickCube seed 0** (`HERP`, `p-only`, `sigma-only`, `uniform`).
5. If RQ3 is viable, run seeds 1–2 and freeze HERP.
6. Run **RQ4** using PickCube checkpoints and 128-rollout oracle pools.
7. Enable entropy logging and produce **RQ2** from promoted ManiSkill runs.
8. Run RQ1 seed-0 matrix.
9. After all cells are valid, run RQ1 seeds 1–2.
10. Only then run reserve/challenge tasks.

This ordering is deliberately designed so that a broken `p × sigma` mechanism or poor sigma estimator is discovered **before** the most expensive cross-backbone experiment campaign.
