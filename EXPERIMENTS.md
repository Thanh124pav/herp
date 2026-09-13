# HERP v3 — EXPERIMENTS.md

> **Target venue:** ICRA 2027  
> **Purpose:** executable experiment plan for HERP v3.  
> **Primary claim:** under a fixed online interaction budget, behavior-aware partitioning plus relevance–dispersion allocation \(n_v \propto p_v \sigma_v\) improves learning efficiency over undirected exploration, curriculum/revisit baselines, and uniform state revisitation.

---

# 0. Experimental questions

The paper should answer five questions.

**Q1 — Main performance.**  
Does HERP improve task success / return per environment interaction over PPO, exploration baselines, curriculum baselines, and state-revisit baselines?

**Q2 — Partition.**  
Does behavior-aware chain partition avoid region explosion while preserving decision-relevant boundaries?

**Q3 — Sigma.**  
Does low-budget predicted / estimated \(\hat\sigma_v\) rank regions similarly to a high-budget oracle \(\sigma_v^\star\), and does the linear predictor reduce finite-sample noise?

**Q4 — Relevance.**  
Does gradient-alignment \(p_v\) predict the actual downstream usefulness of additional PPO data from region \(v\)?

**Q5 — Allocation.**  
Is the product \(p_v\sigma_v\) better than either factor alone, uniform allocation, learning-progress curricula, and uncertainty-only exploration?

---

# 1. Baseline taxonomy

Do not force all related methods into one comparison table. HERP has three distinct baseline families:

1. **same-backbone causal baselines** — needed to prove which HERP component matters;
2. **closest curriculum / revisit baselines** — needed to establish novelty against methods that actively choose where training resumes;
3. **recent strong RL baselines** — needed to show competitive sample efficiency against modern continuous-control algorithms.

The paper must clearly distinguish an **exact published implementation** from a **HERP-setting adaptation**.

---

## 1.1 Tier A — same-backbone causal baselines

All Tier-A methods use the same PPO implementation, policy network, rewards, simulator wrappers, total interaction budget, and evaluation protocol.

### A0. PPO

Ordinary PPO from the environment reset distribution only.

### A1. PPO + RND

Random Network Distillation intrinsic reward.

Purpose: standard novelty-driven exploration.

### A2. PPO + Disagreement

Forward-model ensemble disagreement.

Purpose: uncertainty-driven exploration without explicit state revisitation.

### A3. PPO + ICM / Curiosity

Optional if compute permits.

Purpose: prediction-error exploration using the same PPO backbone.

### A4. Uniform Region Revisit

Use HERP's partition and snapshot infrastructure but allocate uniformly:

\[
n_v \propto 1.
\]

This is a mandatory baseline because it isolates whether HERP gains come merely from restoring previously visited states.

### A5. State-Radius Revisit

Use the old HERP-v2 normalized Euclidean state clustering, while keeping the v3 acquisition pipeline.

Run at least:

\[
\text{State-Radius + Uniform},
\qquad
\text{State-Radius + }p\sigma.
\]

This isolates the partition contribution.

### A6. PLR-Region — HERP-setting adaptation of PLR

PLR itself prioritizes **training levels** by estimated learning potential, with TD error as an effective priority signal.

For a same-backbone within-trajectory comparison, keep HERP's chain regions but replace the allocation score by regional TD error:

\[
w_v^{\mathrm{PLR}}
=
\operatorname{EMA}
\left[
\mathbb E_{(s,a,r,s')\in v}
|\delta_t|
\right].
\]

Then:

\[
n_v \propto w_v^{\mathrm{PLR}}.
\]

Call this method **PLR-Region**, never simply PLR, because the original ICML 2021 method samples procedural levels rather than robot state regions.

This is a high-priority baseline because it directly tests:

> Is HERP's \(p_v\sigma_v\) better than a generic learning-potential curriculum applied to the same regions?

### A7. SACL-style Revisit — HERP-setting adaptation

SACL resets agents to previously visited states and prioritizes subgames using a value-based learning signal in zero-sum MARL.

The original SACL score depends on its game-theoretic setting and should not be copied verbatim into single-agent robotics.

Implement a clearly labelled single-agent adaptation using the same HERP regions and snapshot mechanism, with a value-learning score such as:

\[
w_v^{\mathrm{SACL-style}}
=
\left|
V_t(v)-V_{t-\Delta}(v)
\right|
+
\lambda_u U_V(v),
\]

where \(U_V(v)\) is critic / ensemble uncertainty.

This tests a close alternative principle:

> revisit states whose value is changing or uncertain.

Do not claim that this adaptation is the original SACL algorithm.

### A8. Sigma-only HERP

\[
n_v \propto \sigma_v.
\]

### A9. Relevance-only HERP

\[
n_v \propto p_v.
\]

### A10. Full HERP

\[
\boxed{
n_v \propto p_v\sigma_v
}
\]

### A11. Oracle-\(\sigma\) HERP

Use estimated \(p_v\) but high-budget frozen-policy \(\sigma_v^\star\).

Diagnostic only; not deployable.

### A12. Oracle-\(p\) diagnostic

Use controlled PPO-delta as \(p_v^\star\) on frozen checkpoints.

Diagnostic only.

---

# 2. Closest published curriculum / state-revisit baselines

These methods are especially important for novelty positioning because they alter the training distribution, the initial-state distribution, or the location where new experience is collected.

---

## B1. PLR — ICML 2021

**Prioritized Level Replay.**

Original unit of allocation:

```text
procedurally generated training level
```

Original signal:

```text
estimated learning potential, with TD error as an effective priority
```

Relation to HERP:

```text
PLR:  choose which whole level to revisit
HERP: choose which behavioral region inside trajectories to revisit
```

Use the exact PLR paper as conceptual lineage, and use `PLR-Region` from Tier A for an apples-to-apples PPO ablation.

---

## B2. VaPRL — NeurIPS 2021

**Value-accelerated Persistent Reinforcement Learning / Autonomous RL via Subgoal Curricula.**

VaPRL constructs a curriculum of initial states using value information to reduce reliance on manual environment resets.

Relation to HERP:

```text
VaPRL: initial-state / subgoal curriculum for reset-free learning
HERP: budget allocation across discovered behavior regions under restorable simulation
```

Run exact VaPRL only on a compatible goal-conditioned or persistent-RL subset. Do not force it onto every ManiSkill/DMC task.

Use it primarily to answer whether HERP's region allocation offers advantages beyond value-based initial-state curriculum design.

---

## B3. SACL — AAAI 2024

**Subgame Automatic Curriculum Learning.**

SACL adaptively resets agents to previously visited states and prioritizes subgames in zero-sum MARL.

Relation to HERP is very close structurally:

```text
visited states -> select useful restart states -> collect more learning experience
```

but the original objective is game-theoretic and evaluated in multi-agent zero-sum settings.

Therefore:

- cite original SACL prominently as state-revisit lineage;
- use Tier-A `SACL-style Revisit` for the single-agent robotics causal comparison;
- do not present the adapted score as exact SACL.

---

## B4. RFCL — ICLR 2024

**Reverse Forward Curriculum Learning.**

RFCL is a high-priority robotics baseline because it explicitly exploits simulator state reset and constructs a reverse curriculum followed by a forward curriculum from demonstrations.

Its official repository supports ManiSkill and includes the following directly relevant manipulation tasks:

```text
PickCube
StackCube
PegInsertionSide
PlugCharger
```

This overlap should directly influence HERP's ManiSkill benchmark selection.

Important fairness distinction:

```text
RFCL receives demonstrations.
HERP does not require demonstrations.
```

Therefore RFCL is a strong external reference, but not a causal same-information baseline.

Where possible, run the official RFCL implementation and report demonstration count explicitly.

---

## B5. Active RL — AAAI 2025

**Active Reinforcement Learning Strategies for Offline Policy Improvement.**

This is one of the closest recent papers in problem formulation.

It assumes an existing offline dataset and actively collects a small number of additional online trajectories under an interaction budget.

Reported domains include:

```text
Gym-MuJoCo locomotion
Maze2d
AntMaze
CARLA
IsaacSimGo1
```

Relation to HERP:

```text
Active RL:
    offline dataset -> actively acquire additional trajectories

HERP:
    fully online learning -> continually partition -> score -> acquire -> update
```

Use it in a **secondary offline-to-online protocol** rather than pretending it has the same fully-online setting.

This baseline motivates adding a MuJoCo locomotion overlap block to the benchmark suite.

---

## B6. LP-ACRL — RA-L 2026

**Learning Progress-based Automatic Curriculum Reinforcement Learning.**

LP-ACRL estimates learning progress online and adapts the task/terrain sampling distribution for quadruped locomotion across stairs, slopes, gravel, low-friction terrain, and high-speed commands.

Relation:

```text
LP-ACRL: allocation across tasks / terrain parameters
HERP:    allocation across regions inside trajectories
```

This is a highly recent robotics curriculum reference but differs in granularity.

Use it exactly only if the paper includes a legged-locomotion / terrain benchmark. Otherwise cite it and include a learning-progress Tier-A adaptation if desired.

---

## B7. SR2 — IJCAI 2025

**State Revisit and Re-explore.**

Conceptually close because it revisits previously seen states and reinitializes the simulator for further exploration.

However, SR2 is an offline-and-online setting with an imperfect simulator, rather than HERP's fully online acquisition setting.

Use as:

1. close related work;
2. optional exact baseline in a dedicated offline-to-online experiment.

---

# 3. Recent strong algorithmic baselines

These methods establish that HERP is competitive against modern sample-efficient / exploration algorithms, but they do not isolate HERP components because their RL backbones differ.

## C1. BRO — NeurIPS 2024

Use official BRO where possible.

Why include:

- strong modern continuous-control baseline;
- optimistic exploration;
- broad evaluation across DMC, MetaWorld, and MyoSuite;
- convenient overlap with the proposed HERP benchmark suite.

## C2. MaxInfoRL — ICLR 2025

Use official state-based implementation on shared DMC / continuous-control tasks.

Why include:

- directly targets online exploration;
- uses information gain;
- provides a modern uncertainty/information-driven comparator.

If a PPO adaptation is implemented, call it `MaxInfo-style PPO`; do not label it as exact MaxInfoRL.

## C3. SOMBRL — NeurIPS 2025

Use on shared DMC tasks.

This provides a strong contrast between:

```text
HERP: model-free region-budget allocation
SOMBRL: optimistic model-based exploration
```

## C4. CP3ER — NeurIPS 2024, optional visual extension

Use only if HERP includes a visual-observation experiment.

Otherwise keep it in Related Work / appendix planning.

---

# 4. Recommended comparison tables

Do not force every baseline into one huge table.

## 4.1 Main same-backbone causal table

Run broadly across the HERP core suite:

```text
PPO
PPO + RND
PPO + Disagreement
Uniform Region Revisit
State-Radius Revisit
PLR-Region
SACL-style Revisit
HERP-σ
HERP-p
HERP-pσ
```

This is the most important table for claims about HERP.

## 4.2 Published close-baseline table / subset comparisons

Run methods only where their original assumptions make sense:

```text
RFCL       -> ManiSkill demonstration-based subset
VaPRL      -> persistent / goal-conditioned subset
Active RL  -> offline-to-online MuJoCo / navigation subset
LP-ACRL    -> optional terrain / locomotion subset
SR2        -> optional offline-to-online subset
```

Use `N/A` rather than inventing unnatural ports.

## 4.3 Strong modern RL comparison

```text
BRO
MaxInfoRL
SOMBRL (DMC subset)
```

Compare only on shared tasks and equal raw environment-step budgets.

---

# 5. Environment strategy

The benchmark suite must satisfy two requirements:

1. **robotics diversity** — manipulation, locomotion, contact-rich control;
2. **baseline overlap** — include tasks/suites used by the closest and strongest baselines.

HERP should not be evaluated only on ManiSkill.

---

# 6. ManiSkill — robotics manipulation core and RFCL overlap

ManiSkill is the primary robotics suite because it provides GPU-parallel simulation, restorable physical states, multi-stage manipulation, and direct overlap with RFCL.

## 6.1 Core ManiSkill tasks

Use the four RFCL-overlap tasks as the main block:

```text
PickCube-v1
StackCube-v1
PegInsertionSide-v1
PlugCharger-v1
```

Why these four:

- direct comparison / lineage with RFCL;
- progressively harder manipulation;
- clear behavioral stages for partition analysis;
- PegInsertionSide and PlugCharger provide high-precision contact-rich tasks where exploration budget matters.

## 6.2 Additional ManiSkill tasks

Add in appendix / extended suite:

```text
PushCube-v1
PushT-v1
OpenCabinetDrawer-v1
```

Roles:

- `PushCube`: easy sanity check;
- `PushT`: non-trivial planar pushing / contact dynamics;
- `OpenCabinetDrawer`: articulated-object manipulation.

---

# 7. Gym-MuJoCo locomotion — Active-RL overlap family

Add a locomotion block to prevent the paper from being manipulation-only and to overlap the environment family used by Active RL.

Recommended standard tasks:

```text
HalfCheetah
Hopper
Walker2d
Ant
```

Exact Gym/Gymnasium version must be fixed in the reproducibility table.

Use these for two settings:

### Setting A — fully online HERP

Run PPO and Tier-A HERP baselines from scratch.

### Setting B — offline-to-online active acquisition

Create or use a fixed offline dataset and compare against the exact Active-RL-style acquisition protocol on a subset of these tasks.

Do not mix Setting A and Setting B in the same aggregate score.

Also consider `Maze2d` and `AntMaze` only for the secondary Active-RL overlap experiment, because their offline/goal-navigation conventions differ from the main online PPO setting.

---

# 8. DeepMind Control Suite — strong-modern-baseline overlap

DMC gives direct overlap with BRO / MaxInfoRL / SOMBRL-style modern continuous-control evaluation and contains useful long-motion tasks for partition analysis.

Recommended subset:

```text
Acrobot-Swingup
Finger-Turn-Hard
Hopper-Hop
Walker-Run
Quadruped-Run
Humanoid-Walk
```

Special role of `Walker-Run`:

> long physical displacement should not automatically generate more HERP behavioral regions when the local behavioral regime remains unchanged.

This task is especially useful for the region-explosion analysis.

---

# 9. MetaWorld and optional MyoSuite

## 9.1 MetaWorld

MetaWorld gives diverse one-embodiment manipulation and overlaps strong recent continuous-control work.

Core subset:

```text
Assembly
Hand-Insert
Stick-Pull
Pick-Place
```

Appendix:

```text
Hammer
Button-Press
```

`Assembly` is particularly useful because it also appears in RFCL-style manipulation evaluation outside ManiSkill and in recent continuous-control comparisons.

## 9.2 MyoSuite — optional appendix

Use only after the core suite is stable:

```text
Key-Turn-Hard
Obj-Hold-Hard
Pen-Twirl-Hard
Pose-Hard
```

This adds high-dimensional musculoskeletal / dexterous control and overlaps BRO's benchmark family.

## 9.3 Optional legged terrain block for LP-ACRL

If compute and simulator support permit, add a small terrain-curriculum experiment inspired by LP-ACRL:

```text
stairs
slopes
gravel
low-friction flat terrain
```

This should be a separate locomotion experiment, not a blocking requirement for the main paper.

---

# 9A. Final recommended task suite

## 9A.1 Core paper suite — 18 tasks

### ManiSkill — 4

```text
PickCube-v1
StackCube-v1
PegInsertionSide-v1
PlugCharger-v1
```

### Gym-MuJoCo — 4

```text
HalfCheetah
Hopper
Walker2d
Ant
```

### DMC — 6

```text
Acrobot-Swingup
Finger-Turn-Hard
Hopper-Hop
Walker-Run
Quadruped-Run
Humanoid-Walk
```

### MetaWorld — 4

```text
Assembly
Hand-Insert
Stick-Pull
Pick-Place
```

This 18-task core suite covers:

- multi-stage robot manipulation;
- precision insertion;
- articulated / contact-rich behavior in extensions;
- standard MuJoCo locomotion;
- long state drift;
- high-dimensional control;
- multiple simulator families;
- direct overlap with RFCL, Active-RL environment families, and modern continuous-control baselines.

## 9A.2 Extended appendix suite

Add:

```text
ManiSkill:
    PushCube-v1
    PushT-v1
    OpenCabinetDrawer-v1

MetaWorld:
    Hammer
    Button-Press

MyoSuite:
    Key-Turn-Hard
    Obj-Hold-Hard
    Pen-Twirl-Hard
    Pose-Hard

Offline-to-online overlap:
    Maze2d
    AntMaze

Optional terrain block:
    stairs
    slopes
    gravel
    low-friction terrain
```

---

# 9B. Main training budgets

Do not force one raw budget across simulator families when standard learning scales differ.

## Gym-MuJoCo / DMC / MetaWorld / MyoSuite

Start with:

\[
1\text{M environment steps}
\]

for the broad comparison because this is a common sample-efficiency scale in modern continuous-control work.

Report AUC over the full 0–1M curve, not only final return.

If PPO demonstrably fails to enter a meaningful learning regime within 1M on a selected task, increase the budget **for all methods on that task** and document the change before final comparison.

## ManiSkill

Use task-appropriate budgets after a PPO calibration sweep.

Initial planning values:

```text
PickCube-v1          5M
StackCube-v1        10M
PegInsertionSide-v1 20M–50M
PlugCharger-v1      20M–50M
PushCube-v1          5M
PushT-v1            10M
OpenCabinetDrawer-v1 20M
```

The final budget must be fixed before comparing HERP against baselines.

---

# 10. Seeds and statistical reporting

Minimum:

\[
5 \text{ random seeds}
\]

for every main method-task cell.

Preferred for the most important comparisons:

\[
10 \text{ seeds}
\]

on:

```text
PickCube
PegInsertionSide
Assembly
Stick-Pull
Walker-Run
Humanoid-Walk
```

Report:

- mean ± 95% bootstrap CI;
- IQM across tasks;
- median;
- optimality gap when a normalized reference exists;
- probability of improvement over PPO;
- learning curves versus environment steps.

Use `rliable` style aggregate metrics for cross-task comparisons.

---

# 11. Main performance metrics

Primary:

\[
\text{success rate versus environment interactions}
\]

for manipulation tasks.

For DMC / MyoSuite:

\[
\text{episode return versus environment interactions}.
\]

Cross-task aggregate:

\[
\text{normalized score}
\]

with task-specific random / expert or known upper reference.

Also report:

### Sample efficiency

Steps to reach fixed performance thresholds.

Example:

```text
steps to 50% success
steps to 80% success
```

when achievable.

### AUC

Area under the learning curve within the fixed interaction budget.

### Wall-clock

Secondary only.

HERP's main claim is interaction efficiency, not necessarily computational speed.

---

# 12. Oracle definition for sigma experiments

The oracle experiment must freeze the object being estimated.

At checkpoint \(c\):

1. freeze policy parameters \(\theta_c\);
2. freeze region partition / region archive;
3. select a fixed set of regions;
4. from each region collect a large number \(K^\star\) of independent \(M\)-step continuations;
5. compute high-budget direct dispersion.

Define:

\[
q_v^\star
=
\frac{1}{K^\star(K^\star-1)}
\sum_{i<j}
d_M(\xi_i,\xi_j)
\]

and

\[
\sigma_v^\star=\sqrt{q_v^\star}.
\]

Recommended:

```text
K* = 128
```

for the final oracle.

Before trusting it, compare:

```text
K*=32
K*=64
K*=128
```

and verify oracle stabilization.

No PPO update is allowed while collecting these rollouts.

---

# 13. Sigma correlation experiment

This should be one of the main mechanism figures.

For each frozen checkpoint and region:

- collect oracle pool \(K^\star=128\);
- repeatedly subsample low-budget sets.

Use:

\[
K_{\mathrm{low}}
\in
\{2,4,8,16,32\}.
\]

For each \(K_{\mathrm{low}}\), compare:

### S0. Direct empirical estimator

\[
\hat\sigma_v^{\mathrm{direct}}.
\]

### S1. Linear predictor

\[
\hat\sigma_v^{f}.
\]

### S2. Shrinkage estimator

\[
\hat q_v
=
\lambda_v q_v^{\mathrm{direct}}
+
(1-\lambda_v)f(x_v).
\]

### S3. Old HERP-v2 estimator

Discounted / compressed future representation.

### S4. One-step dispersion

Only next-state dispersion.

### S5. End-of-chain dispersion

Compare chain endpoints.

### S6. Variable-chain-length dispersion

Use full chain.

This ablation should demonstrate why fixed \(M\) is needed.

---

# 14. Sigma correlation metrics

Do not report only Pearson correlation.

Allocation is largely ranking-based, so ranking quality is important.

Report:

### Pearson

\[
r(\hat\sigma,\sigma^\star).
\]

### Spearman

\[
\rho(\hat\sigma,\sigma^\star).
\]

This should be the primary correlation metric.

### Kendall-\(\tau\)

Useful when number of regions is moderate.

### Top-k overlap

For:

```text
k = 5
k = 10
top 20%
```

measure intersection between highest predicted-\(\sigma\) regions and oracle-\(\sigma\) regions.

### NDCG

Treat oracle \(\sigma^\star\) as relevance and predicted ranking as retrieval ranking.

This directly measures whether allocator prioritization is correct.

### Normalized MAE / RMSE

Measure value accuracy, not just ordering.

---

# 15. Sigma finite-sample variance experiment

This directly tests the motivation for learning \(f\).

For every region with oracle pool:

1. sample \(R\) independent low-budget subsets of size \(K\);
2. estimate \(\hat q\) from each subset;
3. compare estimator variance.

Recommended:

```text
R = 100 resamples
K = 4, 8, 16
```

Compute:

\[
\operatorname{Var}_{\mathrm{subsample}}
[
\hat q_v^{\mathrm{direct}}
]
\]

versus

\[
\operatorname{Var}_{\mathrm{subsample}}
[
\hat q_v^{\mathrm{shrinkage}}
].
\]

Report:

\[
\text{variance reduction ratio}
=
\frac{
\operatorname{Var}(\hat q^{\mathrm{direct}})
}{
\operatorname{Var}(\hat q^{\mathrm{shrinkage}})
}.
\]

The desired result is not merely higher correlation; it is lower finite-sample estimator variance at similar bias.

---

# 16. Sigma bias–variance decomposition

For each estimator \(e\):

\[
\operatorname{MSE}_e
=
\operatorname{Bias}_e^2
+
\operatorname{Variance}_e.
\]

Estimate across repeated low-budget resampling.

Expected behavior:

```text
direct:
    low bias
    high variance

linear f:
    potentially higher bias
    low variance

shrinkage:
    best finite-sample MSE
```

This is the cleanest empirical justification for \(f\).

---

# 17. Sigma checkpoint dependence

Run oracle correlation at:

```text
early
middle
late
```

training checkpoints.

Recommended relative positions:

\[
10\%,\quad 50\%,\quad 90\%
\]

of the baseline training budget.

Reason:

\[
\Xi_v
\]

changes as the policy changes.

A predictor that works only after convergence is not sufficient for HERP.

---

# 18. Sigma horizon ablation

Test:

\[
M\in\{4,8,16,32,64\}.
\]

Measure:

- oracle correlation;
- low-budget estimator variance;
- downstream HERP performance;
- compute cost.

The expected curve may be non-monotonic:

```text
too small M:
    misses meaningful future branching

too large M:
    rollout noise dominates / costs too much
```

Main \(M\) should be chosen on development tasks and then frozen.

---

# 19. Sigma representation ablation

Compare trajectory features:

### R0. State only

\[
\psi_t=\phi(s_t).
\]

### R1. Action only

\[
\psi_t=a_t.
\]

### R2. State + action

\[
\psi_t=[\phi(s_t),\lambda_a a_t].
\]

### R3. Endpoint only

\[
\psi=\phi(s_M).
\]

### R4. Mean feature

Compress all \(M\) steps to an average before pairwise distance.

The main hypothesis is:

```text
full fixed-M state+action sequence
```

should preserve more future branching information than endpoint-only or one-step variants.

---

# 20. Predictor-feature ablation

For the linear \(f\), use nested features.

### F0

```text
intercept only
```

### F1

```text
policy-distribution change
```

### F2

```text
policy change + state change
```

### F3

```text
policy change + state change + action entropy
```

### F4

```text
policy change + state change + action entropy + log visitation count
```

Report:

- held-out \(R^2\);
- Spearman with oracle;
- coefficients;
- coefficient confidence intervals.

If action entropy is state-independent because PPO uses a global log-std, state this explicitly and omit it from the final default model.

---

# 21. Predictor generalization protocol

Do not evaluate \(f\) only on the same regions used to fit it.

Use two protocols.

## 21.1 Region holdout

Randomly hold out 20% of regions.

Train on 80%, evaluate on 20%.

## 21.2 Temporal holdout

Train \(f\) on labels from earlier rounds.

Evaluate on newly created regions / later checkpoints before adding their labels.

Temporal holdout is more realistic for HERP.

---

# 22. Partition ablations

Compare:

### P0. Raw state radius

Current HERP-v2 style.

### P1. K-means / online centroid state clustering

No chain structure.

### P2. Fixed temporal windows

For example every 10 or 20 steps.

### P3. Action-distribution boundaries only

\[
b_t=D(\pi_{t-1},\pi_t).
\]

### P4. State-change boundaries only

\[
b_t=\|s_t-s_{t-1}\|.
\]

### P5. Weighted action + state boundary

\[
b_t
=
\lambda_\pi d^\pi_t
+
\lambda_s d^s_t.
\]

### P6. Full HERP chain + cross-trajectory clustering

Main method.

---

# 23. Partition-specific tests

## 23.1 Long-travel invariance

Construct / select tasks with long monotonic movement.

Compare the same behavior executed over distance scaling:

\[
1,\;10,\;100,\;1000.
\]

Measure number of regions / chains.

Desired:

```text
state clustering:
    region count grows with traveled distance

HERP chain partition:
    chain count remains roughly stable
```

DMC Walker-Run / locomotion tasks are natural real benchmarks for this property.

---

## 23.2 Boundary semantic alignment

For manipulation tasks with meaningful stages:

```text
reach
approach
grasp
lift
transport
insert / place
```

manually annotate a small diagnostic set or infer task-event markers from simulator state.

Measure distance from detected boundaries to task-event transitions.

This is only a diagnostic; the method remains unsupervised.

---

## 23.3 Region explosion

Plot:

\[
|\mathcal V_t|
\]

versus:

- total environment steps;
- episode horizon;
- cumulative path length.

HERP should grow with newly discovered behavioral contexts, not mechanically with physical path length.

---

# 24. p-v ablation

At frozen checkpoints select at least 30 regions.

Compute estimated:

```text
cosine gradient alignment
dot-product gradient alignment
occupancy
random score
```

Compute controlled PPO-delta oracle:

\[
\Delta_v
=
J_{\mathrm{ref}}(\theta+\text{base+region update})
-
J_{\mathrm{ref}}(\theta+\text{base update}).
\]

Report:

- Pearson;
- Spearman;
- sign accuracy:

\[
\Pr[
\operatorname{sign}(p_v)
=
\operatorname{sign}(\Delta_v)
].
\]

The main relevance estimator should be chosen before the final full training sweep.

---

# 25. Allocation ablation

Run the following under identical partition and PPO backbone:

```text
Uniform
σ only
p only
p * σ
oracle-σ with estimated p
```

If compute permits:

```text
oracle-p * oracle-σ
```

on short controlled runs.

This gives a clean ladder:

\[
\text{uniform}
\rightarrow
\text{one signal}
\rightarrow
\text{two signals}
\rightarrow
\text{oracle ceiling}.
\]

---

# 26. Root-region ablation

Compare:

### V0. Fixed PPO/root only

\[
n_0=B.
\]

### V1. Hard root lower bound

\[
n_0\ge\eta B.
\]

Use \(\eta\in\{0.1,0.25\}\).

### V2. Fixed root/local mixture

Example:

```text
70% root
30% regions
```

### V3. Root as ordinary HERP candidate

No hard lower bound.

This is the main method.

### V4. Root sigma definition

Compare:

```text
direct root rollout sigma
child total-variance sigma
```

The intended HERP-v3 main formulation is:

\[
\sigma_0^2
=
\sum_c w_c\sigma_c^2
+
\sum_c w_c\|\mu_c-\mu_0\|^2
\]

with a common \(\varepsilon_\sigma\) floor.

Direct root rollouts serve as an empirical validation / oracle for this recursive construction.

Report correlation and absolute error between the two.

---

# 27. Degenerate-case experiment

Explicitly test all-success and all-fail regimes.

Create controlled checkpoints / tasks where observed outcomes are nearly identical.

Set:

\[
\sigma_v=\varepsilon_\sigma
\]

when empirical variation vanishes.

Verify experimentally:

```text
no NaN
no zero-probability dead regions
allocation becomes approximately uniform when all p and σ signals are indistinguishable
root remains eligible
```

This should be a unit test and a small experiment figure/table.

---

# 28. Stochasticity stress test

HERP's benefit should depend on meaningful branching.

Create controlled environment variants with:

### Action noise

\[
a_t^{env}=a_t+\epsilon_t.
\]

### State / dynamics noise

Perturb object friction, mass, or transition noise.

Levels:

```text
0
low
medium
high
```

Measure:

- oracle \(\sigma^\star\);
- predicted \(\hat\sigma\);
- HERP improvement over PPO;
- allocation entropy.

Expected trend:

```text
higher meaningful branching
→ larger variation in σ across regions
→ more opportunity for HERP to improve allocation.
```

This directly tests the motivation of HERP.

---

---

# 29. Fairness rules for external baselines

External baselines are split into exact published methods and HERP-setting adaptations.

## 29.1 Exact published methods

For RFCL, VaPRL, Active RL, LP-ACRL, BRO, MaxInfoRL, SOMBRL, and SR2 when run exactly:

1. use official code when practical;
2. keep their original required information sources explicit (e.g. RFCL demonstrations, Active RL offline data);
3. use shared environment/task definitions where possible;
4. match raw online environment-interaction budget for the online portion;
5. report the native RL backbone;
6. never treat cross-backbone differences as component-level proof.

## 29.2 HERP-setting adaptations

For `PLR-Region`, `SACL-style Revisit`, or any future adapted curriculum:

1. use the identical PPO backbone as HERP;
2. use identical HERP regions / snapshots unless partition itself is being ablated;
3. change only the allocation score;
4. label the method with `-Region`, `-style`, or `adapted` in every table and plot;
5. never cite the adaptation's numbers as if they were official numbers from the original paper.

Main causal conclusions must come from Tier-A same-backbone comparisons.

---

# 30. Experiment phases

Do not launch the complete matrix immediately.

## Phase 1 — mechanism validation

Use four deliberately different tasks:

```text
PickCube-v1
PegInsertionSide-v1
Walker-Run
Assembly
```

Run:

- partition diagnostics;
- \(\sigma\)-oracle correlation;
- repeated-subsample bias–variance analysis;
- linear-\(f\) generalization;
- \(p\) controlled PPO-delta.

No full external baseline suite yet.

## Phase 2 — same-backbone algorithm pilot

Same four tasks, 3 seeds first.

Run:

```text
PPO
RND
Disagreement
Uniform Revisit
State-Radius Revisit
PLR-Region
SACL-style Revisit
HERP-σ
HERP-p
HERP-pσ
```

Proceed only if full HERP gives a consistent interaction-efficiency signal.

## Phase 3 — closest published robotics / acquisition baselines

Integrate:

```text
RFCL on ManiSkill overlap tasks
Active RL on the offline-to-online MuJoCo subset
VaPRL on a compatible persistent / goal-conditioned subset if practical
```

LP-ACRL and SR2 are optional exact integrations depending on available simulator/settings.

## Phase 4 — strong recent algorithmic baselines

Integrate:

```text
BRO
MaxInfoRL
SOMBRL on DMC
```

## Phase 5 — full core suite

Run the 18-task core suite with at least 5 seeds for Tier-A methods.

External baselines run only on their meaningful overlap subsets.

## Phase 6 — key-task confirmation

Use 10 seeds on:

```text
PegInsertionSide-v1
PlugCharger-v1
Walker2d
Walker-Run
Assembly
Hand-Insert
```

## Phase 7 — appendix generalization

Add:

```text
PushT
OpenCabinetDrawer
Hammer
Button-Press
MyoSuite
Maze2d / AntMaze offline-to-online
optional terrain curriculum
```

---

# 31. Minimum figures for the paper

## Figure 1 — Method

Trajectory → behavior-consistent chains → regions → \(p,\sigma\) → allocation.

## Figure 2 — Main learning curves

Representative tasks:

```text
PegInsertionSide-v1
PlugCharger-v1
Walker2d
Walker-Run
Assembly
```

## Figure 3 — Aggregate same-backbone performance

IQM normalized performance across the 18-task core suite for Tier-A methods.

Do not mix demonstration-assisted RFCL or offline-to-online Active RL into the same aggregate IQM.

## Figure 4 — Sigma oracle correlation

X-axis:

\[
K_{\mathrm{low}}
\]

Y-axis:

```text
Spearman(sigma_hat, sigma_oracle)
```

Curves:

```text
direct
linear f
shrinkage
old-v2
one-step
endpoint
```

## Figure 5 — Sigma bias–variance

Direct vs linear \(f\) vs shrinkage across \(K\in\{4,8,16\}\).

## Figure 6 — Partition behavior

Region count vs traveled distance / episode horizon, plus detected boundaries on manipulation trajectories.

## Figure 7 — Allocation-signal comparison

```text
Uniform
PLR-Region
SACL-style Revisit
σ only
p only
pσ
oracle-σ
```

## Figure 8 — Closest published baseline subsets

Separate panels:

```text
ManiSkill: HERP vs RFCL
Offline-to-online MuJoCo: HERP acquisition vs Active RL
```

Keep assumptions visible in the caption.

---

# 32. Minimum tables

## Table 1 — Core task suite

Columns:

```text
Suite
Task
Observation dim
Action dim
Horizon
Reward type
Reset / snapshot support
Training budget
Baseline overlap
```

`Baseline overlap` examples:

```text
RFCL
Active-RL family
BRO / MaxInfoRL / SOMBRL
```

## Table 2 — Same-backbone main results

Rows:

```text
PPO
RND
Disagreement
Uniform Revisit
State-Radius Revisit
PLR-Region
SACL-style Revisit
HERP-σ
HERP-p
HERP
```

Columns grouped by suite plus overall IQM.

## Table 3 — Published external-baseline subset results

Rows may include:

```text
RFCL
VaPRL
Active RL
BRO
MaxInfoRL
SOMBRL
HERP
```

Use `—` for incompatible settings/tasks instead of forced adaptations.

## Table 4 — Sigma estimator

Rows:

```text
direct K=4
direct K=8
linear f
shrinkage
old v2
one-step
endpoint
variable-length
```

Columns:

```text
Pearson
Spearman
Kendall tau
Top-20% overlap
NDCG
normalized RMSE
bias
variance
```

## Table 5 — Component ablations

```text
partition
allocation signal
root handling
p estimator
M horizon
predictor features
```

---

# 33. Compute-saving rule

If the matrix becomes too large, prioritize:

1. same-backbone causal baselines;
2. \(\sigma\)-oracle validation and bias–variance experiments;
3. `PLR-Region` and `SACL-style Revisit` as the closest allocation-signal alternatives;
4. full \(p\sigma\) factor ablation;
5. exact RFCL on the ManiSkill overlap block;
6. exact Active RL on a secondary offline-to-online subset;
7. BRO / MaxInfoRL shared-task comparisons;
8. task diversity across ManiSkill + MuJoCo + DMC + MetaWorld;
9. SOMBRL;
10. VaPRL exact subset;
11. MyoSuite / LP-ACRL terrain extension;
12. visual CP3ER extension.

Do not sacrifice mechanism validation merely to maximize the number of named baselines.

---

# 34. Recommended first implementation queue

Implement next in this order:

```text
1. Fixed-M oracle collector
2. Sigma correlation + repeated-subsample bias/variance analysis
3. Uniform Region Revisit
4. State-Radius Revisit
5. PLR-Region allocator
6. SACL-style value-change/uncertainty allocator
7. Full p / sigma / p*sigma switches
8. RFCL-compatible ManiSkill task wrappers: PickCube, StackCube, PegInsertionSide, PlugCharger
9. Gym-MuJoCo adapter: HalfCheetah, Hopper, Walker2d, Ant
10. DMC adapter
11. MetaWorld task expansion
12. RFCL runner / official-env bridge
13. Active-RL secondary offline-to-online runner
14. BRO runner
15. MaxInfoRL runner
16. SOMBRL runner
17. optional VaPRL / MyoSuite / terrain extensions
```

Items 1–7 are required before deciding whether HERP v3 itself works.

---

# 35. Baseline decision summary

For the ICRA paper, treat the following as **must-have** unless implementation proves infeasible:

```text
PPO
RND
Disagreement
Uniform Region Revisit
State-Radius Revisit
PLR-Region
SACL-style Revisit
HERP-σ
HERP-p
HERP-pσ
RFCL on ManiSkill overlap
Active RL on a secondary offline-to-online overlap
BRO or MaxInfoRL on shared continuous-control tasks
```

The conceptual novelty comparison should read:

```text
PLR / LP-ACRL:
    allocate across levels / tasks using learning progress

VaPRL / RFCL:
    curriculum over initial states, often with reset-free or demonstration assumptions

SACL / SR2:
    revisit previously seen states in different problem settings

Active RL:
    actively request additional trajectories from offline-to-online data

HERP:
    discover behavior-consistent trajectory regions online,
    estimate their future dispersion and downstream relevance,
    and allocate a unified interaction budget by p_v * sigma_v.
```
