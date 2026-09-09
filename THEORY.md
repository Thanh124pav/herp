# HERP Theory: Performance-Aware Exploration of Trajectory Space

**Target:** ICRA 2027  
**Status:** implementation-facing theory draft; freeze the core formulation first, then only add theory that is experimentally testable.

## 1. Core research question

In online reinforcement learning, a fixed interaction budget should not be spent uniformly over trajectory space. Different state regions have different roles:

1. **Exploration need:** some states are branching points, where nearby actions or stochastic dynamics lead to substantially different future trajectories.
2. **Performance relevance:** some states produce policy updates that are much more aligned with the optimization direction that improves performance under the target/deployment distribution.

HERP allocates extra rollout budget to regions that are simultaneously **branching** and **performance-relevant**.

The core question is:

> Given an interaction budget \(B\), from which previously reached state regions should the current policy collect additional on-policy rollouts to maximize downstream evaluation performance?

The working allocation rule is

\[
n_v^\star \propto p_v \sigma_v,
\]

where \(v\) indexes a region of trajectory space, \(p_v\) measures performance relevance, and \(\sigma_v\) measures future-trajectory branching / uncertainty.

---

## 2. Problem setup

Consider an MDP

\[
\mathcal M=(\mathcal S,\mathcal A,P,r,\gamma,\rho_0)
\]

with policy \(\pi_\theta(a\mid s)\). Training proceeds online. The simulator permits saving and restoring environment states that have already been reached by the agent.

Let

\[
\mathcal V_t=\{v_1,\ldots,v_{K_t}\}
\]

be an adaptive partition of the currently discovered trajectory space. A region \(v\) contains a set of related states and an archive of simulator snapshots

\[
\mathcal A_v=\{x_{v,1},\ldots,x_{v,M_v}\}.
\]

A snapshot contains enough simulator state to restore the environment exactly enough for short future rollouts.

At allocation round \(t\), HERP has an additional rollout budget \(B_t\). It selects

\[
n_v \in \mathbb N_0,\qquad \sum_{v\in\mathcal V_t}n_v\le B_t.
\]

Each selected rollout is initialized from a stored state in region \(v\), then executed using the **current** policy \(\pi_{\theta_t}\). Therefore, the rollout is on-policy with respect to the action-generating policy, although its initial-state distribution is deliberately changed.

---

## 3. Region representation

For the ICRA implementation, use a simple representation first.

For each state \(s\), define

\[
z(s)=\phi(s)\in\mathbb R^d,
\]

where \(\phi\) is one of:

1. normalized privileged state features from ManiSkill `state` / `state_dict`;
2. task-relevant state features plus robot proprioception;
3. later, a learned encoder if needed.

A state belongs to region \(v\) if its representation is close to the region centroid \(c_v\). An online radius-based or k-center clustering rule is sufficient for the first version.

The theoretical formulation does **not** require a particular clustering algorithm. It only requires that each region be sufficiently local that its future-trajectory statistics are meaningful.

---

# Part I. Exploration need: \(\sigma_v\)

## 4. What \(\sigma_v\) should mean

The desired quantity is not merely state novelty.

We want \(\sigma_v\) to answer:

> If the agent reaches region \(v\), how many qualitatively different futures are accessible from this region under small changes in action or environment stochasticity?

A high-\(\sigma_v\) region is a **branching region**. Additional rollouts there are useful because one trajectory is an insufficient description of what can happen next.

A low-\(\sigma_v\) region is locally predictable. Repeating many rollouts from it provides redundant information.

---

## 5. Main estimator: future-trajectory dispersion

From a stored state \(x_v\), run \(K\) short probe rollouts of horizon \(H\):

\[
\tau^{+}_{v,k}
=
(s_{0}^{(k)},a_{0}^{(k)},s_{1}^{(k)},\ldots,s_H^{(k)}),
\qquad k=1,\ldots,K,
\]

with \(s_0^{(k)}=x_v\).

Compress each future into a trajectory feature

\[
Z_{v,k}
=
\Psi(\tau^{+}_{v,k}).
\]

The simplest implementation is a discounted mean of normalized state features:

\[
Z_{v,k}
=
\frac{
\sum_{h=1}^{H}\gamma_b^{h-1}\phi(s_h^{(k)})
}{
\sum_{h=1}^{H}\gamma_b^{h-1}
}.
\]

Then define pairwise future dispersion

\[
\hat\sigma_{v,\text{pair}}^2
=
\frac{1}{2K(K-1)}
\sum_{i\ne j}
\|Z_{v,i}-Z_{v,j}\|_2^2.
\]

Using the variance identity,

\[
\frac12\mathbb E\|Z-Z'\|_2^2
=
\operatorname{tr}\operatorname{Cov}(Z),
\]

this is an empirical estimate of future-trajectory variance.

### Interpretation

- If all probe rollouts stay close, \(\hat\sigma_v\) is small.
- If the futures diverge into different modes, \(\hat\sigma_v\) is large.
- The score is explicitly multi-step, unlike one-step prediction error.

This should be the **first estimator implemented**.

---

## 6. Preferred refinement: separate controllable branching from stochastic noise

A weakness of naive curiosity is the "noisy-TV" problem: unpredictable but uncontrollable randomness can receive a high exploration score.

HERP can explicitly separate two sources of future variance.

Let \(q_\delta(a_{0:H-1}\mid x_v)\) be a local action-probe distribution around the current policy. Draw \(A\) action probes. For each action probe, if the environment is stochastic, repeat it \(M\) times with different environment random seeds.

Let

\[
Z_{v,a,m}
=
\Psi(\tau^+_{v,a,m}).
\]

Define

\[
\bar Z_{v,a}
=
\frac1M\sum_{m=1}^M Z_{v,a,m},
\qquad
\bar Z_v
=
\frac1A\sum_{a=1}^A \bar Z_{v,a}.
\]

### Environment / aleatoric variance

\[
\hat\sigma^2_{\text{dyn},v}
=
\frac1A
\sum_{a=1}^A
\frac1M
\sum_{m=1}^M
\|Z_{v,a,m}-\bar Z_{v,a}\|_2^2.
\]

### Action-sensitive / controllable branching variance

\[
\hat\sigma^2_{\text{branch},v}
=
\frac1A
\sum_{a=1}^A
\|\bar Z_{v,a}-\bar Z_v\|_2^2.
\]

This is a law-of-total-variance decomposition:

\[
\operatorname{Var}(Z\mid v)
=
\mathbb E_A[
\operatorname{Var}(Z\mid v,A)
]
+
\operatorname{Var}_A(
\mathbb E[Z\mid v,A]
).
\]

For HERP, the **main exploration score should be**

\[
\sigma_v
=
\sqrt{
\hat\sigma^2_{\text{branch},v}
+
\lambda_{\text{dyn}}\hat\sigma^2_{\text{dyn},v}
},
\]

with \(\lambda_{\text{dyn}}\in[0,1]\). Start with

\[
\lambda_{\text{dyn}}=0
\]

in deterministic simulators. This prioritizes states where different controllable choices genuinely create different futures, instead of rewarding irreducible noise.

---

## 7. Alternative \(\sigma_v\) estimators to ablate

### 7.1 Return dispersion

\[
\sigma^2_{R,v}
=
\operatorname{Var}_{k}
\left[
\sum_{h=0}^{H-1}
\gamma^h r_{h}^{(k)}
+
\gamma^H V_\theta(s_H^{(k)})
\right].
\]

Pros:
- directly task-aware;
- cheap.

Cons:
- can miss geometrically different futures with currently similar value.

### 7.2 Per-trajectory gradient variance

Let

\[
G_{v,k}
=
\nabla_\theta \ell(\theta;\tau^+_{v,k}).
\]

Define

\[
\sigma^2_{g,v}
=
\frac1K
\sum_k
\|G_{v,k}-\bar G_v\|_2^2.
\]

This is theoretically the cleanest choice for the Neyman-allocation result below, but it is more expensive and less directly connected to the "branching state" story.

Use it as an **oracle / diagnostic** or compute it on the policy head only.

### 7.3 Dynamics-ensemble disagreement

Train an ensemble

\[
f_j(s,a)\approx \phi(s')
\]

and define

\[
\sigma^2_{\text{ens},v}
=
\mathbb E_{(s,a)\sim v}
\operatorname{Var}_{j}
[f_j(s,a)].
\]

This directly connects to Self-Supervised Exploration via Disagreement. It is a strong baseline and an alternative HERP estimator.

### 7.4 Distributional trajectory discrepancy

Replace Euclidean distance by MMD, energy distance, Wasserstein approximation, or Jensen-Shannon divergence after discretization / density modeling:

\[
\sigma_v^{\text{dist}}
=
D(
P(\Psi(\tau^+)\mid v),
\text{reference distribution}
).
\]

Do not make this the first implementation unless Euclidean trajectory features fail.

---

# Part II. Connection to the Simulation Lemma

## 8. Lobel-Parr tight simulation-lemma bound

Lobel and Parr (RLC 2024, arXiv:2406.16249) consider two MDPs whose transition kernels differ by at most

\[
\epsilon_T
\]

in \(L_1\), with reward error \(\epsilon_R\). They derive the tight discounted value-error bound

\[
|V^\pi(s)-\hat V^\pi(s)|
\le
\frac{1}{1-\gamma}
-
\frac{1-\epsilon_R}
{1-\gamma(1-\epsilon_T/2)}.
\]

When \(\epsilon_R=0\),

\[
B_\gamma(\epsilon_T)
=
\frac{1}{1-\gamma}
-
\frac{1}
{1-\gamma(1-\epsilon_T/2)}.
\]

This quantity is monotone in transition mismatch.

### How HERP can use this correctly

The paper does **not** directly prove that pairwise future-trajectory distance is equal to \(\epsilon_T\). Therefore, HERP should not claim that the pairwise trajectory score itself is the Simulation Lemma error.

Instead define a local set of plausible transition models

\[
\mathcal P_v
\]

from a dynamics ensemble or empirical confidence set. Let

\[
\hat\epsilon_{T,v}
=
\sup_{P_i,P_j\in\mathcal P_v}
\|P_i(\cdot\mid v)-P_j(\cdot\mid v)\|_1.
\]

Then an optional bound-shaped exploration score is

\[
\sigma^{\text{SL}}_v
=
B_\gamma(\hat\epsilon_{T,v}).
\]

This gives a principled mapping from local transition-model uncertainty to worst-case policy-value uncertainty.

### Recommended use in ICRA

Use the Lobel-Parr result as:

1. theoretical motivation that transition uncertainty is not equally consequential for value;
2. an optional ensemble-based \(\sigma_v^{\text{SL}}\) ablation;
3. a reason to use a nonlinear value-sensitivity mapping rather than only raw transition error.

Do **not** make the entire paper depend on proving that trajectory dispersion exactly estimates the Simulation-Lemma \(\epsilon_T\).

---

# Part III. Performance relevance: \(p_v\)

## 9. Training-time reference objective

Do not use the final held-out benchmark test set during training.

Instead maintain a small **training-time reference distribution** \(\mathcal D_{\mathrm{ref}}\), collected from normal environment resets and standard current-policy rollouts under the target task distribution.

The final paper evaluation must use disjoint seeds / task instances.

Let the policy loss on the reference batch be

\[
L_{\mathrm{ref}}(\theta).
\]

Define

\[
g_{\mathrm{ref}}
=
\nabla_\theta L_{\mathrm{ref}}(\theta).
\]

For region \(v\), collect a small on-policy batch \(\mathcal D_v\) and compute

\[
g_v
=
\nabla_\theta L_v(\theta).
\]

---

## 10. First-order gradient-alignment score

Suppose a small update using region \(v\) is

\[
\theta^+
=
\theta-\eta g_v.
\]

A Taylor expansion gives

\[
L_{\mathrm{ref}}(\theta^+)
\approx
L_{\mathrm{ref}}(\theta)
-
\eta
g_{\mathrm{ref}}^\top g_v.
\]

Therefore

\[
u_v^{\mathrm{FO}}
=
g_{\mathrm{ref}}^\top g_v
\]

is a first-order estimate of how helpful an update on region \(v\) is for the reference objective.

A scale-insensitive alternative is cosine alignment:

\[
c_v
=
\frac{
g_{\mathrm{ref}}^\top g_v
}{
\|g_{\mathrm{ref}}\|_2\|g_v\|_2+\epsilon
}.
\]

For the main implementation define

\[
\tilde p_v
=
[\operatorname{EMA}(c_v)]_+ + \epsilon_p,
\]

and normalize

\[
p_v
=
\frac{\tilde p_v^\alpha}
{\sum_u \tilde p_u^\alpha}.
\]

This makes \(p_v\) a probability-like performance-relevance distribution.

### Important distinction

Gradient alignment is a **first-order influence proxy**, not a full influence function.

---

## 11. Influence-function version

For a training objective with Hessian

\[
H_\theta
=
\nabla_\theta^2 L_{\mathrm{train}}(\theta),
\]

the classical upweighting influence of region \(v\) on the reference loss has the form

\[
\mathcal I_v
=
-
g_{\mathrm{ref}}^\top
H_\theta^{-1}
g_v.
\]

With the sign convention that positive \(p_v\) should mean "helpful", define

\[
u_v^{\mathrm{IF}}
=
g_{\mathrm{ref}}^\top
(H_\theta+\lambda I)^{-1}
g_v.
\]

Because policy-gradient Hessians are non-convex and can be indefinite, a more stable RL approximation is to replace \(H\) with a Fisher / Gauss-Newton matrix:

\[
u_v^{\mathrm{Fisher}}
=
g_{\mathrm{ref}}^\top
(F_\theta+\lambda I)^{-1}
g_v.
\]

Implementation options:

1. **FO cosine alignment** — main MVP.
2. **FO dot product** — ablation.
3. **Diagonal empirical Fisher** — recommended second estimator.
4. **Conjugate-gradient inverse-Fisher-vector product** — stronger but slower.
5. **LiSSA / IHVP** — only if time permits.

For ICRA, the safest story is:

> HERP uses first-order performance alignment in the main algorithm and evaluates second-order/Fisher influence as an estimator ablation.

---

## 12. Additional \(p_v\) options

### 12.1 Occupancy relevance

If \(d_{\mathrm{ref}}(v)\) is the frequency with which region \(v\) appears under normal reference rollouts,

\[
p_v^{\mathrm{occ}}
=
\frac{d_{\mathrm{ref}}(v)}
{\sum_u d_{\mathrm{ref}}(u)}.
\]

This recovers the original intuition that states frequently encountered at evaluation matter more.

### 12.2 Hybrid occupancy + gradient relevance

\[
\tilde p_v
=
(d_{\mathrm{ref}}(v)+\epsilon)^\eta
([c_v]_+ + \epsilon)^{1-\eta}.
\]

This is useful if pure gradient alignment is noisy.

### 12.3 Actual short-horizon improvement oracle

For analysis only, update a copied policy on region \(v\) for one tiny optimization step and measure

\[
\Delta J_{\mathrm{ref},v}
=
J_{\mathrm{ref}}(\theta_v^+)-J_{\mathrm{ref}}(\theta).
\]

This is too expensive for the main algorithm but is an excellent ground-truth diagnostic for \(p_v\).

---

# Part IV. Why \(n_v \propto p_v\sigma_v\)

## 13. Stratified gradient-estimation view

Let each region produce a stochastic gradient contribution

\[
G_v(\tau),
\qquad
\mu_v=\mathbb E[G_v(\tau)\mid v].
\]

Assume a performance-relevant target gradient has the form

\[
g^\star
=
\sum_v p_v \mu_v.
\]

With \(n_v\) independent rollouts from region \(v\),

\[
\hat g
=
\sum_v
p_v
\left(
\frac1{n_v}
\sum_{j=1}^{n_v}
G_{v,j}
\right).
\]

Let

\[
\sigma_{g,v}^2
=
\operatorname{tr}
\operatorname{Cov}[G_v(\tau)\mid v].
\]

Then

\[
\mathbb E\|\hat g-g^\star\|_2^2
=
\sum_v
\frac{p_v^2\sigma_{g,v}^2}{n_v},
\]

ignoring cross-region covariance.

The budget-allocation problem is

\[
\min_{n_v>0}
\sum_v
\frac{p_v^2\sigma_{g,v}^2}{n_v}
\quad
\text{s.t.}
\quad
\sum_v n_v=B.
\]

The Lagrangian is

\[
\mathcal L
=
\sum_v
\frac{p_v^2\sigma_{g,v}^2}{n_v}
+
\lambda
\left(
\sum_vn_v-B
\right).
\]

Stationarity gives

\[
-\frac{p_v^2\sigma_{g,v}^2}{n_v^2}
+\lambda
=0,
\]

hence

\[
n_v^\star
=
B
\frac{p_v\sigma_{g,v}}
{\sum_u p_u\sigma_{g,u}}.
\]

This is the classical Neyman allocation structure.

---

## 14. Connecting trajectory branching to gradient variance

HERP does not want to compute full gradient variance for every region. It uses future-trajectory dispersion as a proxy.

Assume the per-trajectory gradient signature can be written as

\[
G_v(\tau)=h_\theta(Z_v)
\]

for a future-trajectory representation \(Z_v=\Psi(\tau^+)\), and that \(h_\theta\) is locally \(L_g\)-Lipschitz:

\[
\|h_\theta(z)-h_\theta(z')\|_2
\le
L_g\|z-z'\|_2.
\]

Then

\[
\frac12
\mathbb E
\|G-G'\|_2^2
\le
L_g^2
\frac12
\mathbb E
\|Z-Z'\|_2^2.
\]

Since

\[
\frac12
\mathbb E\|G-G'\|_2^2
=
\operatorname{tr}\operatorname{Cov}(G),
\]

we obtain

\[
\sigma_{g,v}^2
\le
L_g^2
\sigma_{\text{traj},v}^2.
\]

Thus trajectory dispersion can serve as an upper-bound proxy for gradient uncertainty under a local Lipschitz assumption.

This is the cleanest bridge between the desired "branching future trajectories" story and the allocation theory.

---

# Part V. Practical allocator

## 15. Robust finite-budget score

Directly using \(p_v\sigma_v\) can starve regions because both estimates are noisy early in training.

Use

\[
q_v
=
\frac{
(\tilde p_v+\epsilon_p)^{\alpha}
(\tilde\sigma_v+\epsilon_\sigma)^{\beta}
}{
\sum_u
(\tilde p_u+\epsilon_p)^{\alpha}
(\tilde\sigma_u+\epsilon_\sigma)^{\beta}
}.
\]

Then

\[
n_v
=
n_{\min}
+
\operatorname{AllocateMultinomial}
(B-Kn_{\min},q).
\]

Default:

\[
\alpha=1,\qquad
\beta=1,\qquad
n_{\min}=0
\]

with an \(\epsilon\)-floor or 5--10% uniform exploration mass.

A useful staleness term is

\[
q_v
\leftarrow
(1-\rho)q_v
+
\rho
\frac{a_v}{\sum_u a_u},
\]

where \(a_v\) increases with time since region \(v\) was last probed. This mirrors the anti-starvation principle in Prioritized Level Replay.

---

# Part VI. Full HERP algorithm

## 16. Outer-loop algorithm

At each allocation round:

1. Collect ordinary current-policy trajectories from the original initial-state distribution.
2. Add encountered simulator states to the region archive.
3. Build / update \(\mathcal D_{\mathrm{ref}}\) from ordinary target-distribution rollouts.
4. Compute the reference gradient signature \(g_{\mathrm{ref}}\).
5. For candidate regions:
   - restore archived state;
   - run \(K\) short probe futures;
   - estimate \(\sigma_v\);
   - compute region policy-gradient signature \(g_v\);
   - estimate \(p_v\).
6. Compute
   \[
   q_v\propto p_v\sigma_v.
   \]
7. Allocate the extra interaction budget across regions.
8. Restore sampled archive states and collect on-policy rollout fragments.
9. Train PPO on ordinary + allocated fragments.
10. Repeat with the updated policy.

The ordinary rollouts are never removed. HERP reallocates an **additional or fixed split of the same total interaction budget**, so all methods must be compared under identical environment-step budgets.

---

# Part VII. Paper-facing theoretical claims

## 17. Claims that are currently defensible

### Proposition A — Optimal stratified allocation

Under independent per-region gradient sampling and fixed \(p_v\), the allocation minimizing the trace variance of the weighted gradient estimator is

\[
n_v^\star\propto p_v\sigma_{g,v}.
\]

This can be formally proved exactly.

### Proposition B — First-order performance relevance

For sufficiently small \(\eta\),

\[
L_{\mathrm{ref}}(\theta-\eta g_v)
=
L_{\mathrm{ref}}(\theta)
-
\eta g_{\mathrm{ref}}^\top g_v
+
O(\eta^2).
\]

Thus gradient alignment is a local first-order proxy for improvement of the reference objective.

This can be formally proved by Taylor expansion.

### Proposition C — Trajectory dispersion controls gradient dispersion

Under the local Lipschitz assumption above,

\[
\sigma_{g,v}
\le
L_g\sigma_{\text{traj},v}.
\]

This justifies using future-trajectory branching as a computational proxy.

### Proposition D — Transition disagreement can be mapped to value uncertainty

If \(\hat\epsilon_{T,v}\) is a valid bound on local transition-model mismatch, the Lobel-Parr tight Simulation-Lemma expression yields a monotone bound-shaped value-sensitivity score.

Do not claim that raw trajectory Euclidean distance is itself \(\epsilon_T\).

---

# Part VIII. What the experiments must validate

## 18. Mechanism validation

The paper must separately demonstrate that both learned quantities mean what the method claims.

### \(\sigma_v\) validation

For a sampled set of archived states:

1. estimate \(\hat\sigma_v\) using only \(K\in\{2,4,8\}\) probe rollouts;
2. obtain an expensive oracle using 64--128 futures;
3. measure Spearman correlation between estimated and oracle branching;
4. visualize top and bottom branching states.

Additional test:

- compare future-trajectory dispersion with policy-gradient variance;
- report correlation
  \[
  \rho(\sigma_{\text{traj},v},\sigma_{g,v}).
  \]

### \(p_v\) validation

For sampled regions:

1. compute FO alignment, Fisher alignment, occupancy relevance;
2. on a copied policy, make one small update using data from region \(v\);
3. measure actual reference-performance change \(\Delta J_{\mathrm{ref},v}\);
4. report Spearman correlation between predicted \(p_v\) and actual improvement.

This experiment is essential because it demonstrates that "performance-aware" is not just terminology.

---

## 19. Essential ablations

At minimum:

\[
\text{Uniform},\quad
\sigma\text{-only},\quad
p\text{-only},\quad
p\times\sigma.
\]

Estimator ablations:

- trajectory pairwise \(\sigma\);
- branch/decomposition \(\sigma\);
- ensemble disagreement \(\sigma\);
- gradient variance oracle \(\sigma\);
- occupancy \(p\);
- FO cosine \(p\);
- FO dot-product \(p\);
- diagonal-Fisher \(p\).

Only a subset must appear in the main paper; the rest can be used to decide the final method.

---

# Part IX. Relationship to baselines

## 20. Conceptual comparison

### RND / ICM

They prioritize novelty or prediction error. They do not explicitly ask whether the resulting gradient helps the target performance objective.

### Exploration by disagreement

Very close to the \(\sigma_v\) axis: model disagreement estimates uncertainty in dynamics. HERP differs by:
- using multi-step future branching;
- optionally separating controllable branching from environment noise;
- multiplying exploration need by performance relevance.

### Go-Explore

Very close operationally because it stores promising states and later returns to them before exploring. HERP differs in the criterion for which stored states deserve additional rollouts: a performance-aware \(p_v\sigma_v\) allocation rather than archive heuristics.

### Prioritized Level Replay

PLR prioritizes revisiting training levels based on estimated learning potential and staleness. HERP moves the sampling unit from environment level to **state region inside trajectory space**, and decomposes priority into branching need and performance relevance.

### Gradient-aligned / influence-based online selection

GradAlign and InfOES show that reference-gradient alignment / influence can guide online RL data selection in LLM RL. HERP transfers the optimization principle to robot trajectory-space exploration and combines it with dynamics-dependent branching.

---

# Part X. References

- Lobel, S. and Parr, R. **An Optimal Tightness Bound for the Simulation Lemma.** RLC 2024. https://arxiv.org/abs/2406.16249
- Pathak, D. et al. **Curiosity-driven Exploration by Self-supervised Prediction.** ICML 2017. https://proceedings.mlr.press/v70/pathak17a.html
- Burda, Y. et al. **Exploration by Random Network Distillation.** ICLR 2019. https://arxiv.org/abs/1810.12894
- Pathak, D., Gandhi, D., Gupta, A. **Self-Supervised Exploration via Disagreement.** ICML 2019. https://proceedings.mlr.press/v97/pathak19a.html
- Ecoffet, A. et al. **First return, then explore.** Nature 2021. https://www.nature.com/articles/s41586-020-03157-9
- Jiang, M., Grefenstette, E., Rocktäschel, T. **Prioritized Level Replay.** ICLR 2021. https://arxiv.org/abs/2010.03934
- Yang, N. et al. **GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning.** 2026. https://arxiv.org/abs/2602.21492
- Gong, Y. et al. **Influence-based Online Experience Selection for Effective RLHF.** ACL 2026. https://aclanthology.org/2026.acl-long.2206/
