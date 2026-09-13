# HERP — Updated Positioning, Baselines, and Experiment Plan

## 1. Algorithm Positioning

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
