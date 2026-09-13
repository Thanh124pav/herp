# HERP — Updated Positioning, Baselines, and Experiment Plan

## 0. Repository Audit and Required New Implementation

This revision is grounded in the current `Thanh124pav/herp` repository.

## 0.1 What exists now

HERP v3 is still PPO-centric.

Current `train.py` methods:

```text
ppo, herp, herp_sigma, herp_p, rnd, disagreement, go_explore, plr
```

The v3 campaign additionally contains PPO-family variants such as:

```text
uniform, state_radius_uniform, state_radius_psigma,
plr_region, sacl_style, herp_sigma, herp_p, herp
```

Current native HERP baselines:

```text
src/herp/baselines/rnd.py
src/herp/baselines/disagreement.py
```

There is old SAC code at:

```text
src/experience_routing/rl/sac.py
```

but it belongs to the older experience-routing project. It is **not** a HERP-v3 SAC backbone. It can only be used as implementation reference.

The repository already contains locked external checkouts for:

```text
RFCL
ActiveRL
BRO
MaxInfoRL
```

with SHAs recorded in:

```text
docs/v3/external_baselines.lock.json
```

Missing work:

1. add a clean official SAC baseline;
2. integrate HERP with SAC;
3. download/build a model-based baseline;
4. execute BRO and MaxInfoRL from the locked repositories;
5. standardize outputs across PPO/SAC/MBRL.

---

## 0.2 Do not rewrite the whole PPO path

The existing HERP-PPO pipeline is the stable core. Do not refactor it into a universal trainer before submission.

Instead add:

```text
src/herp/learners/
    base.py
    ppo.py
    sac.py

scripts/
    sac_official.py
    train_herp_sac.py
    merge_cross_backbone_results.py
```

Minimal learner interface:

```python
class Learner:
    def act(self, obs, deterministic=False): ...
    def observe(self, transition): ...
    def update(self): ...
    def save(self, path): ...
    def load(self, path): ...
    def policy_parameters(self): ...
    def gradient_signature(self, batch): ...
```

HERP partitioning, archive, probing, and allocation should depend on this small interface rather than PPO internals.

For PPO, `gradient_signature()` uses the PPO policy objective.

For SAC, `gradient_signature()` should use the **actor objective**, not the Q-network objective, so the meaning of \(p_v\) remains:

```text
downstream policy-learning relevance
```

---

## 0.3 Expand the existing ManiSkill upstream checkout

The official ManiSkill repo already provides maintained PPO, SAC, and TD-MPC2 baselines.

If `third_party/ManiSkill` already exists as the current sparse checkout:

```bash
git -C third_party/ManiSkill sparse-checkout add \
  examples/baselines/sac \
  examples/baselines/tdmpc2
```

If it is absent:

```bash
mkdir -p third_party

git clone --filter=blob:none --sparse \
  https://github.com/mani-skill/ManiSkill.git \
  third_party/ManiSkill

git -C third_party/ManiSkill sparse-checkout set \
  examples/baselines/ppo \
  examples/baselines/sac \
  examples/baselines/tdmpc2
```

Lock the exact commit:

```bash
git -C third_party/ManiSkill rev-parse HEAD
```

Relevant paths:

```text
third_party/ManiSkill/examples/baselines/sac/sac.py
third_party/ManiSkill/examples/baselines/tdmpc2/
```

---

## 0.4 Build vanilla SAC first

Create:

```text
scripts/sac_official.py
```

Prefer a thin wrapper around ManiSkill's official SAC baseline. A lightly patched copy, analogous to the existing `scripts/ppo_official.py`, is also acceptable.

Only patch infrastructure:

```text
seed
output/checkpoint path
evaluation frequency
logging schema
sim backend
raw environment-step counting
```

Do not silently modify:

```text
SAC objective
actor/critic architecture
entropy tuning
replay logic
UTD
published/task-specific hyperparameters
```

Smoke-test:

```text
PushCube-v1
PickCube-v1
```

Gate:

```text
vanilla SAC reaches non-trivial success/return
AND
raw environment-step accounting is correct
```

Only then build HERP-SAC.

---

## 0.5 Implement HERP-SAC

Add:

```text
src/herp/learners/sac.py
scripts/train_herp_sac.py
```

Start from the official ManiSkill SAC actor, twin critics, targets, entropy tuning, and replay buffer.

HERP should change **where new data is acquired**, not SAC's update rule:

\[
\text{HERP allocation}
\rightarrow
\text{restored-region interactions}
\rightarrow
\text{SAC replay buffer}
\rightarrow
\text{native SAC updates}.
\]

Minimum publishable pair:

```text
sac
sac_herp
```

Optional:

```text
sac_herp_p
sac_herp_sigma
```

### Replay fairness

For the main causal comparison keep replay sampling **uniform**.

Do not add prioritized replay at the same time, because that would change both:

```text
acquisition distribution
and
utilization distribution
```

Log:

```text
new transitions per region
buffer occupancy per region
replay samples per region
actor-gradient signature
return / success
```

### SAC relevance estimator

Use the SAC actor objective:

\[
J_{\text{actor}}
=
\mathbb{E}[
\alpha\log\pi(a|s)-Q(s,a)
].
\]

For region \(v\):

\[
g_v^{\mathrm{SAC}}
=
\nabla_\theta J_{\text{actor}}(D_v).
\]

Construct a reference gradient from ordinary-reset/reference data:

\[
g_{\text{ref}}^{\mathrm{SAC}}.
\]

Then use the same alignment family as PPO, e.g.

\[
p_v =
\max(0,\cos(g_v^{\mathrm{SAC}},g_{\text{ref}}^{\mathrm{SAC}})).
\]

Thus the semantics of \(p_v\) remain unchanged across backbones.

---

## 0.6 Add model-based RL as a native comparison group

For the first submission, MBRL does **not** need to be HERP-integrated.

Use TD-MPC2 first.

ManiSkill already maintains a TD-MPC2 baseline, with commands under:

```text
third_party/ManiSkill/examples/baselines/tdmpc2/baselines.sh
```

Useful overlap tasks:

```text
PushCube-v1
PickCube-v1
StackCube-v1
PegInsertionSide-v1
PushT-v1
```

Keep its native trainer. Normalize only:

```text
seed
raw environment-step budget
evaluation protocol
output format
```

Imagined/model rollouts do **not** count as raw environment interactions.

### Optional official TD-MPC2 checkout

For DMC / MetaWorld:

```bash
git clone https://github.com/nicklashansen/tdmpc2.git \
  third_party/tdmpc2

git -C third_party/tdmpc2 rev-parse HEAD
```

Build TD-MPC2 in its own environment using its provided conda or Docker setup.

Do not force TD-MPC2 dependencies into the main HERP environment.

### Optional SOMBRL

If a second modern MBRL/exploration method is needed:

```bash
git clone https://github.com/lasgroup/ombrl.git \
  third_party/ombrl

git -C third_party/ombrl rev-parse HEAD
```

Example isolated setup:

```bash
conda create -n herp-ombrl python=3.11 -y
conda activate herp-ombrl
pip install -U "jax[cuda12]"
pip install -e third_party/ombrl
```

TD-MPC2 has higher priority because its ManiSkill overlap is cleaner.

---

## 0.7 Existing external baselines

Do not re-download the already locked baselines unless intentionally updating them.

Current locked revisions:

```text
RFCL      7bd660f05dbda1ec8881bcfb250a8d1881532dbb
ActiveRL  cb367f31da874b50388faa930af09587d3a93887
BRO       5870a13d4bf4f66eeddfc291a59df6d2009de600
MaxInfoRL 8da195780d1de6d2100598ce8aa62c1ded9ae19c
```

Use isolated environments when needed:

```text
herp-main
herp-bro
herp-maxinforl
herp-tdmpc2
herp-ombrl      # optional
herp-rfcl       # optional
```

Do not spend time forcing PyTorch HERP, JAX BRO, and MBRL stacks into one environment.

---

## 0.8 Where-to-Learn

Where-to-Learn is a valuable 2026 comparator, but do not assume a full drop-in implementation is publicly available.

Policy:

```text
if official runnable code exists:
    clone + lock SHA + run exact implementation
else:
    cite prominently
    optionally implement a clearly labelled reproduction
```

A from-scratch reproduction should not block:

```text
HERP-PPO
HERP-SAC
BRO
MaxInfoRL
TD-MPC2
```

If reimplemented, label it:

```text
Where-to-Learn-style reimplementation
```

unless official author code is used.

---

## 0.9 Standardize the outer protocol, not the inner algorithms

Keep native trainers for PPO, SAC, BRO, MaxInfoRL, and TD-MPC2.

Standardize only:

```text
task
seed
raw environment-step budget
evaluation interval
number of evaluation episodes
return / success metric
checkpoint naming
result schema
```

Add:

```text
scripts/merge_cross_backbone_results.py
```

Recommended result record:

```json
{
  "method": "sac_herp",
  "family": "off_policy",
  "task": "PickCube-v1",
  "seed": 0,
  "env_steps": 1000000,
  "eval_success": 0.0,
  "eval_return": 0.0,
  "source": "herp_native"
}
```

Merge:

```text
outputs/v3_campaign/
outputs/sac_campaign/
outputs/external/BRO/
outputs/external/MaxInfoRL/
outputs/external/TD-MPC2/
```

Main fairness quantity:

\[
\boxed{\text{raw environment interactions}}
\]

not optimizer steps, replay samples, imagined rollouts, or wall-clock time.

---

## 0.10 Implementation order

### Phase 0 — freeze current HERP-PPO

Archive:

```text
current git commit
package/environment lock
working checkpoints
current result CSVs
```

### Phase 1 — native strong baselines

1. official SAC on 2 ManiSkill tasks;
2. TD-MPC2 on 2 ManiSkill tasks;
3. BRO on shared/native tasks;
4. MaxInfoRL on shared/native tasks;
5. normalize outputs.

### Phase 2 — HERP-SAC

Start with:

```text
PickCube-v1
PushCube-v1 or StackCube-v1
```

Run 3 seeds:

```text
SAC
HERP-SAC
```

If the gain is stable, expand to the full ManiSkill subset.

### Phase 3 — remaining recent methods

Only after the above:

```text
RFCL
SOMBRL
Where-to-Learn reproduction
```

if they strengthen the final story.

### Phase 4 — final ablations

Mandatory:

```text
HERP-p
HERP-sigma
HERP-p*sigma
Uniform Revisit
partition ablation
```

High-value:

```text
PLR-Region
SACL-style
interaction-budget sensitivity
```

RND and Disagreement remain classic diagnostic controls, not the main modern-SOTA claim.

---

## 0.11 Definition of done

```text
[ ] current HERP-PPO remains reproducible
[ ] official SAC learns the selected ManiSkill tasks
[ ] HERP-SAC uses the same SAC update as vanilla SAC
[ ] HERP-SAC main experiment uses uniform replay
[ ] TD-MPC2 executes on selected shared tasks
[ ] BRO runs from its locked SHA
[ ] MaxInfoRL runs from its locked SHA
[ ] all methods report raw environment steps consistently
[ ] evaluation protocol is matched within each task
[ ] all external SHAs are saved
[ ] all package/env locks are saved
[ ] one merger produces the PPO/SAC/MBRL main table
[ ] ablations are generated separately
```

Not required for the first submission:

```text
HERP-TD-MPC2
HERP-Dreamer
one universal trainer
one Python environment containing every baseline
```

# 1. Algorithm Positioning

### 1.1 Core positioning

HERP should be positioned as:

> **HERP is a behavior-aware interaction allocation framework for online reinforcement learning.**

HERP does **not** replace PPO, SAC, or a model-based RL learner. Instead, it operates one level above the underlying RL optimizer and answers:

> **Given a finite environment-interaction budget, where should additional experience be collected?**

The conceptual decomposition is:

\[
\boxed{\text{HERP: where to collect experience}}
\qquad
\text{vs.}
\qquad
\boxed{\text{RL backbone: how to learn from experience}}
\]

HERP therefore consists of three main ideas:

1. **Behavior-aware partitioning** of previously encountered trajectories into meaningful regions \(v\).
2. **Evaluation relevance** \(p_v\), estimating how useful region \(v\) is for downstream performance.
3. **Future uncertainty / dispersion** \(\sigma_v\), estimating how much additional information or learning signal may still be obtained from region \(v\).

The interaction budget is then allocated according to

\[
\boxed{
n_v \propto p_v \sigma_v
}
\]

where \(n_v\) denotes the number of additional environment interactions allocated to region \(v\).

---

### 1.2 HERP is a framework, not a PPO-specific algorithm

The intended abstraction is

\[
\text{HERP allocation}
\rightarrow
\text{experience collection}
\rightarrow
\text{RL learner}.
\]

Possible instantiations include

\[
\text{HERP-PPO},
\qquad
\text{HERP-SAC},
\qquad
\text{potentially HERP-MBRL}.
\]

The primary implementation may still use PPO because the on-policy setting gives a particularly clean correspondence

\[
\text{allocated interaction}
\rightarrow
\text{new rollout data}
\rightarrow
\text{policy update}.
\]

This makes PPO convenient for controlled causal analysis of the allocation mechanism.

However, HERP should not be described as merely a PPO submodule. The intended contribution is the **interaction-allocation principle**, while PPO is one realization of the downstream learner.

---

### 1.3 PPO, SAC, and model-based RL

The main learning paradigms considered are:

| Paradigm | Representative learner | Key property |
|---|---|---|
| On-policy | PPO | Fresh rollout data directly drives the next policy updates |
| Off-policy | SAC | New data enters a replay buffer and can be reused many times |
| Model-based RL | TD-MPC2 / Dreamer-style | Real data is used to learn a world model, which supports planning or imagined learning |

HERP is naturally compatible with PPO.

For SAC, HERP allocates **data acquisition**, while the replay buffer determines **data utilization**. Therefore the effect of a region depends on both

\[
P(\text{region collected})
\quad\text{and}\quad
P(\text{region sampled from replay}).
\]

This makes HERP-SAC an important generality test.

For model-based RL, the allocation question remains valid because the learned world model still depends on finite real-environment data. A natural extension is

\[
\text{HERP}
\rightarrow
D_{\text{real}}
\rightarrow
\text{world model}
\rightarrow
\text{planner / policy}.
\]

The model-based extension is not required for the first HERP paper unless the implementation is stable, but at least one strong model-based baseline should be included.

---

### 1.4 Relation to tree search / MCTS

HERP should not be positioned as trajectory tree search.

The distinction is:

| | MCTS / tree search | HERP |
|---|---|---|
| Unit | Action/tree node | Behavioral region |
| Budget | Search / simulation rollouts | Environment interactions |
| Main stage | Decision-time planning | Training-time data acquisition |
| Output | Action / plan | Interaction allocation distribution |
| Objective | Decide what action to take | Decide where more learning data should be collected |

A concise distinction is:

> **MCTS allocates computation to decide what to do; HERP allocates environment interaction to decide where to learn.**

Therefore MCTS is related conceptually but is not a mandatory direct baseline unless HERP is explicitly extended into planning/search.

---

# 2. Baseline Strategy

The comparison suite should be divided into three roles rather than placing all methods into one undifferentiated list.

---

## 2.1 Group A — Controlled same-backbone baselines

These are necessary to isolate **why HERP works**.

All methods in this group should use the same PPO implementation, policy network, environment wrappers, reward, interaction budget, and evaluation protocol.

### Mandatory controlled baselines

| Method | Purpose |
|---|---|
| **PPO** | Vanilla on-policy learner |
| **PPO + RND** | Classic novelty-driven exploration |
| **PPO + Disagreement** | Uncertainty-driven exploration |
| **Uniform Region Revisit** | Tests whether simply resetting/revisiting old states is enough |
| **PLR-Region** | Learning-potential allocation on the same HERP regions |
| **HERP-\(p\)** | Relevance-only allocation |
| **HERP-\(\sigma\)** | Uncertainty/dispersion-only allocation |
| **HERP-\(p\sigma\)** | Full method |

### High-value optional controlled baseline

**SACL-style Revisit**

Use the HERP region/snapshot infrastructure but replace the HERP score with a value-change/value-uncertainty score.

This tests the alternative principle

> revisit states whose value is changing or uncertain.

Because the original SACL setting is different, this must be labeled as an adaptation rather than exact SACL.

---

## 2.2 Group B — Recent strong online RL / exploration baselines

These are necessary to show that HERP is competitive with **current methods**, not only older exploration techniques.

### High-priority recent methods

#### Where-to-Learn — RA-L 2026

Role:

- recent directed-exploration method;
- especially relevant because it addresses where an on-policy robot should explore;
- potentially the closest recent competitor to HERP's high-level motivation.

Priority: **very high**.

---

#### MaxInfoRL — ICLR 2025

Role:

- recent information-gain exploration;
- strong comparison against uncertainty/information-driven acquisition;
- especially useful for off-policy comparison.

Priority: **very high**.

---

#### BRO — NeurIPS 2024

Role:

- strong modern continuous-control RL algorithm;
- useful as a competitive sample-efficiency baseline;
- represents a strong off-policy learner rather than a controlled allocation ablation.

Priority: **high**.

---

### Optional recent methods

#### SOMBRL — NeurIPS 2025

Role:

- modern model-based exploration;
- useful if environment overlap and implementation stability are sufficient.

Priority: medium/high if code integration is practical.

#### LP-ACRL — RA-L 2026

Role:

- very recent learning-progress curriculum;
- conceptually relevant but allocates at task/terrain level rather than within-trajectory region level.

Priority: primarily related work or adapted learning-progress baseline unless a terrain benchmark is added.

---

## 2.3 Group C — Closest curriculum / state-revisit robotics baselines

These establish that HERP is not merely another restart curriculum.

### RFCL — ICLR 2024

Important because it:

- uses simulator state resets;
- constructs a curriculum over restart states;
- has direct structural similarity to HERP;
- overlaps well with ManiSkill manipulation tasks.

Key fairness distinction:

> RFCL receives demonstrations; HERP does not require demonstrations.

Use exact RFCL on a compatible ManiSkill subset if feasible.

---

### PLR — ICML 2021

Use primarily as conceptual lineage.

The exact published PLR allocates across training levels rather than behavioral trajectory regions.

For the controlled main experiment, use **PLR-Region** rather than presenting it as exact PLR.

---

### SACL — AAAI 2024

Use prominently in related work.

For the main causal table, use a clearly labeled **SACL-style** single-agent adaptation if included.

---

### Active RL — AAAI 2025

Relevant but its primary setting is offline-to-online active acquisition.

Use only in a secondary offline-to-online experiment if such a protocol is added.

Do not mix it into the main fully-online HERP table.

---

# 3. Main Experimental Design

The main experimental table should be organized by **learning paradigm**.

A recommended structure is:

| Paradigm | Method | ManiSkill 1 | ManiSkill 2 | ManiSkill 3 | DMC / MetaWorld 1 | DMC / MetaWorld 2 | Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| **On-policy** | PPO | | | | | | |
|  | PPO + HERP | | | | | | |
|  | Where-to-Learn | | | | | | |
| **Off-policy** | SAC | | | | | | |
|  | SAC + HERP | | | | | | |
|  | MaxInfoRL | | | | | | |
|  | BRO | | | | | | |
| **Model-based** | TD-MPC2 / selected MBRL baseline | | | | | | |
|  | SOMBRL, if feasible | | | | | | |

The exact cells do not need to be filled for every method if a benchmark is incompatible.

Use **N/A** rather than forcing unnatural ports.

---

# 4. Environment Strategy

The environment suite should satisfy two goals:

1. **Robotics relevance**
2. **Overlap with strong published baselines**

---

## 4.1 ManiSkill — primary robotics block

ManiSkill should be the main robotics benchmark.

Recommended representative tasks:

- PickCube
- StackCube
- PegInsertionSide
- PushT
- OpenCabinetDrawer
- PlugCharger

A final paper does not need all of them. A balanced subset of approximately 4 tasks is sufficient if they cover:

- easy manipulation;
- multi-stage manipulation;
- contact-rich control;
- sparse or difficult exploration.

ManiSkill is the most important place to test:

\[
\text{PPO},
\text{HERP-PPO},
\text{SAC},
\text{HERP-SAC},
\]

plus recent methods that can be ported fairly.

---

## 4.2 DMC / MetaWorld — shared benchmark block

A smaller DMC or MetaWorld subset should remain in the paper because recent baselines such as BRO and MaxInfoRL were developed and tuned on standard continuous-control benchmarks.

Purpose:

> show that HERP remains competitive in the native benchmark regime of strong modern RL algorithms.

This protects against the criticism that a baseline underperformed only because it was ported into an unfamiliar robotics environment.

---

# 5. Generality Experiment: PPO vs SAC

This should be treated as a particularly important experiment.

Use approximately 2–3 representative environments.

Compare:

\[
\text{PPO}
\quad\text{vs}\quad
\text{HERP-PPO},
\]

and

\[
\text{SAC}
\quad\text{vs}\quad
\text{HERP-SAC}.
\]

If both improve consistently, HERP can be positioned much more convincingly as a framework orthogonal to the underlying optimizer.

A compact result table can be:

| Backbone | Vanilla | + HERP | Relative gain |
|---|---:|---:|---:|
| PPO | | | |
| SAC | | | |

If HERP-SAC does not produce a stable gain, the paper should simply narrow its claim to the on-policy setting rather than overclaiming backbone independence.

---

# 6. Ablation Studies

The ablations should answer a small number of direct scientific questions.

---

## 6.1 Allocation score ablation

Compare

\[
n_v \propto 1,
\]

\[
n_v \propto p_v,
\]

\[
n_v \propto \sigma_v,
\]

and

\[
\boxed{
n_v \propto p_v \sigma_v
}.
\]

Question:

> Is the product \(p_v\sigma_v\) actually better than either factor alone?

This is mandatory.

---

## 6.2 Partition ablation

Compare:

1. no explicit partition / ordinary PPO;
2. state-radius clustering;
3. HERP behavior-aware partition.

Potential configurations:

\[
\text{State-Radius + Uniform},
\]

\[
\text{State-Radius + }p\sigma,
\]

\[
\text{HERP Partition + }p\sigma.
\]

Question:

> Are the gains caused by the allocation rule alone, or does the behavior-aware partition matter?

This is mandatory if partitioning is claimed as a core novelty.

---

## 6.3 Revisit mechanism ablation

Compare:

\[
\text{ordinary PPO reset distribution}
\]

against

\[
\text{uniform region revisit}
\]

and

\[
\text{HERP region revisit}.
\]

Question:

> Is HERP better merely because it can restart from previously seen states?

This is highly recommended.

---

## 6.4 Interaction-budget sensitivity

Evaluate HERP under several acquisition budgets, for example low / medium / high.

Question:

> Does HERP remain useful when the interaction budget is tight, and does its advantage diminish as the budget becomes large?

This is useful if space permits.

---

## 6.5 Estimator quality diagnostics

If \(p_v\) and \(\sigma_v\) are estimated rather than directly observed, include compact diagnostics such as:

- estimated \(\hat p_v\) vs controlled downstream utility;
- estimated \(\hat\sigma_v\) vs a high-budget oracle;
- ranking correlation rather than only raw prediction error.

These can appear as one compact figure rather than a full table.

---

# 7. Recommended Figures

Because the paper is visually constrained, use figures to communicate multiple ideas at once.

## Figure 1 — HERP overview

Illustrate

\[
\text{trajectory}
\rightarrow
\text{behavior-aware partition}
\rightarrow
(p_v,\sigma_v)
\rightarrow
n_v \propto p_v\sigma_v
\rightarrow
\text{new interactions}
\rightarrow
\text{RL learner}.
\]

The figure should make it obvious that PPO/SAC/MBRL sit **after** HERP rather than being replaced by it.

---

## Figure 2 — Learning curves

Use 2–3 representative tasks.

Show:

- environment steps on the x-axis;
- success/return on the y-axis;
- HERP against the strongest relevant competitors.

This communicates sample efficiency more clearly than final-score tables alone.

---

## Figure 3 — Partition / allocation visualization

Example visualization:

- trajectory colored by discovered behavioral region;
- region size or opacity representing \(p_v\);
- another visual encoding for \(\sigma_v\);
- highlighted regions receiving higher interaction budgets.

This figure is useful both scientifically and aesthetically.

---

# 8. Recommended Paper-Level Comparison Structure

The paper should avoid presenting all baselines as if they answer the same question.

A clean structure is:

### Main Table — Modern algorithm comparison

Grouped by

- on-policy;
- off-policy;
- model-based.

Focus on current methods such as:

- Where-to-Learn;
- MaxInfoRL;
- BRO;
- one model-based baseline;
- HERP-PPO;
- HERP-SAC.

---

### Ablation Table — Why HERP works

Include:

- Uniform;
- PLR-Region;
- HERP-\(p\);
- HERP-\(\sigma\);
- HERP-\(p\sigma\);
- partition ablations.

Older baselines such as RND and Disagreement belong here as scientific controls, not as the main claim of state-of-the-art comparison.

---

# 9. Minimum Recommended Experiment Set

If time becomes constrained, prioritize in this order:

### Tier 1 — mandatory

- PPO
- HERP-PPO
- Uniform Region Revisit
- HERP-\(p\)
- HERP-\(\sigma\)
- HERP-\(p\sigma\)
- partition ablation

### Tier 2 — recent competitors

- Where-to-Learn
- MaxInfoRL
- BRO

### Tier 3 — framework generality

- SAC
- HERP-SAC

### Tier 4 — model-based coverage

- TD-MPC2 or another strong MBRL baseline
- SOMBRL if integration is practical

### Tier 5 — additional lineage baselines

- RFCL
- SACL-style
- PLR-Region
- RND
- Disagreement

---

# 10. Final Intended Claim

If the experiments support it, the final positioning should be approximately:

> **HERP is a behavior-aware interaction allocation framework for online reinforcement learning that partitions encountered trajectories into meaningful regions and allocates additional environment interactions according to downstream relevance and future uncertainty. The allocation principle improves sample efficiency across multiple robotic control domains and can be instantiated with different underlying RL learners.**

The strongest version of the claim should only be used if both PPO and SAC experiments support it.

If only PPO is consistently improved, narrow the final claim to:

> **HERP is an interaction allocation framework for on-policy online RL.**

The paper should prioritize a precise, defensible claim over an overly broad one.
