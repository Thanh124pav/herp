# HERP THEORY.md

## 0. Scope

**Target venue:** ICRA 2027  
**Paper scope:** online robot reinforcement learning with a fixed interaction budget.

HERP addresses the question:

> Given a limited online interaction budget, which previously reached regions of trajectory space should receive additional rollouts so that downstream evaluation performance improves most?

The central hypothesis is that different trajectory-space regions should not receive equal interaction budget.

Some regions should be explored more because they are **branching states**: small action changes lead to substantially different futures.

Other regions are more important because updates collected there are more relevant to the policy behavior required under the target evaluation distribution.

HERP therefore assigns rollout budget using two online quantities:

\[
\sigma_v
\]

for **future branching / exploration need**, and

\[
p_v
\]

for **performance relevance**.

The final allocation rule is

\[
\boxed{
n_v^\star \propto p_v \sigma_v
}
\]

subject to a fixed interaction budget.

---

# 1. Problem formulation

Consider an MDP

\[
\mathcal M=(\mathcal S,\mathcal A,P,r,\gamma,\rho_0)
\]

with stochastic policy

\[
\pi_\theta(a\mid s).
\]

Training is online. The agent starts ordinary episodes from the environment initial-state distribution \(\rho_0\), but it may also restore previously visited simulator states.

Let the set of discovered trajectory-space regions at training round \(t\) be

\[
\mathcal V_t=\{v_1,\dots,v_{K_t}\}.
\]

Each region \(v\) stores an archive

\[
\mathcal A_v=\{x_{v,1},\dots,x_{v,M_v}\},
\]

where each \(x_{v,i}\) is an environment snapshot corresponding to a state actually reached by the policy during online interaction.

At a reallocation round, HERP selects

\[
n_v \ge 0
\]

additional rollout steps or fragments for each region, subject to

\[
\sum_{v\in\mathcal V_t} n_v \le B_t.
\]

All probe rollouts, reference rollouts, and allocated rollouts count toward the same online interaction budget.

---

# 2. Region representation

A benchmark-specific adapter exposes a task-state feature map

\[
\phi:\mathcal S\rightarrow\mathbb R^d.
\]

For a state \(s\),

\[
z(s)=\phi(s).
\]

The online regionizer uses normalized features and radius-based assignment. If

\[
\min_v \|z(s)-c_v\|_2 > r
\]

and the maximum number of regions has not been reached, a new region is created. Otherwise, the state is assigned to the nearest region.

For the ICRA version, no learned region encoder is required.

---

# 3. Exploration need: \(\sigma_v\)

## 3.1 Desired meaning

HERP defines \(\sigma_v\) to capture:

> how strongly the future trajectory distribution changes when the agent makes different local action choices from region \(v\).

This is deliberately different from pure state novelty.

A region should have high \(\sigma_v\) when it behaves like a decision point or branch point. A region should have low \(\sigma_v\) when repeated local rollouts produce nearly redundant futures.

---

# 4. Multi-step future trajectory representation

From an archived snapshot \(x_v\), collect short probe futures

\[
\tau_{v,k}^{+}=(s_1^{(k)},\dots,s_H^{(k)}).
\]

Compress each future into

\[
Z_{v,k}=\Psi(\tau_{v,k}^{+}),
\]

where

\[
\Psi(\tau_{v,k}^{+})
=
\frac{
\sum_{h=1}^{H}
\gamma_b^{h-1}\phi(s_h^{(k)})
}{
\sum_{h=1}^{H}
\gamma_b^{h-1}
}.
\]

Here \(\gamma_b\in(0,1]\) controls how strongly nearby future states are weighted.

---

# 5. Main \(\sigma_v\) estimator

The default estimator is pairwise future dispersion:

\[
\hat\sigma_{v,\mathrm{pair}}^2
=
\frac{1}{2K(K-1)}
\sum_{i\neq j}
\|Z_{v,i}-Z_{v,j}\|_2^2.
\]

Equivalently,

\[
\hat\sigma_{v,\mathrm{pair}}
=
\sqrt{
\frac{1}{2K(K-1)}
\sum_{i\neq j}
\|Z_{v,i}-Z_{v,j}\|_2^2
}.
\]

Using

\[
\frac12\mathbb E\|Z-Z'\|_2^2
=
\operatorname{tr}\operatorname{Cov}(Z),
\]

this is a multi-step future-dispersion statistic.

---

# 6. Controllable branching decomposition

A refined version separates:

1. action-sensitive branching,
2. uncontrollable environment randomness.

Let \(a\in\{1,\dots,A\}\) index local action probes and \(m\in\{1,\dots,M\}\) repeated environment trials.

Let

\[
Z_{v,a,m}=\Psi(\tau_{v,a,m}^{+}).
\]

Define

\[
\bar Z_{v,a}
=
\frac1M
\sum_{m=1}^M Z_{v,a,m},
\qquad
\bar Z_v
=
\frac1A
\sum_{a=1}^A
\bar Z_{v,a}.
\]

The controllable branching component is

\[
\hat\sigma_{\mathrm{branch},v}^2
=
\frac1A
\sum_{a=1}^A
\|\bar Z_{v,a}-\bar Z_v\|_2^2.
\]

The environment-noise component is

\[
\hat\sigma_{\mathrm{dyn},v}^2
=
\frac1A
\sum_{a=1}^A
\frac1M
\sum_{m=1}^M
\|Z_{v,a,m}-\bar Z_{v,a}\|_2^2.
\]

Thus,

\[
\operatorname{Var}(Z\mid v)
=
\mathbb E_A[\operatorname{Var}(Z\mid v,A)]
+
\operatorname{Var}_A[\mathbb E(Z\mid v,A)].
\]

HERP may use

\[
\boxed{
\sigma_v
=
\sqrt{
\hat\sigma_{\mathrm{branch},v}^2
+
\lambda_{\mathrm{dyn}}
\hat\sigma_{\mathrm{dyn},v}^2
}
}
\]

with \(\lambda_{\mathrm{dyn}}\in[0,1]\). For deterministic simulation benchmarks, start with \(\lambda_{\mathrm{dyn}}=0\).

---

# 7. Probe design

For a stored state \(x_v\), let \(\mu_\theta(s)\) and \(\sigma_\theta(s)\) denote Gaussian PPO policy parameters.

HERP perturbs the first action:

\[
a_0^{(k)}
=
\mu_\theta(s_0)
+
c\sigma_\theta(s_0)\epsilon_k,
\qquad
\epsilon_k\sim\mathcal N(0,I).
\]

After the first action, the probe follows the current policy.

For branch decomposition, continuation policy noise may be shared across action groups so that between-group differences primarily reflect the first local action choice.

Probe rollouts count toward the global environment-step budget.

---

# 8. Alternative \(\sigma_v\) estimators

## 8.1 Return dispersion

\[
\sigma_{R,v}^2
=
\operatorname{Var}
\left[
\sum_{h=0}^{H-1}
\gamma^h r_h
+
\gamma^H V_\theta(s_H)
\right].
\]

## 8.2 Dynamics-ensemble disagreement

With forward models \(f_j(s,a)\),

\[
\sigma_{\mathrm{ens},v}^2
=
\mathbb E_{(s,a)\sim v}
\operatorname{Var}_j[f_j(s,a)].
\]

## 8.3 Gradient variance oracle

For mechanism validation only,

\[
G_{v,k}
=
\nabla_{\theta_\pi}
\ell_\pi(\theta;\tau_{v,k}),
\]

and

\[
\sigma_{g,v}^2
=
\operatorname{tr}
\operatorname{Cov}[G_v].
\]

This is diagnostic, not part of the main allocator.

---

# 9. Connection to the Simulation Lemma

Lobel and Parr derive a tight discounted Simulation-Lemma bound. Let \(\epsilon_T\) be an \(L_1\) transition-kernel mismatch and \(\epsilon_R\) reward mismatch. Their bound can be written as

\[
|V^\pi(s)-\hat V^\pi(s)|
\le
\frac{1}{1-\gamma}
-
\frac{1-\epsilon_R}
{1-\gamma(1-\epsilon_T/2)}.
\]

For \(\epsilon_R=0\),

\[
B_\gamma(\epsilon_T)
=
\frac{1}{1-\gamma}
-
\frac{1}
{1-\gamma(1-\epsilon_T/2)}.
\]

HERP does not claim that Euclidean trajectory dispersion is itself \(\epsilon_T\). Instead, this motivates the principle that local transition uncertainty can induce non-uniform value uncertainty.

If an ensemble provides a local transition-confidence quantity \(\hat\epsilon_{T,v}\), an optional score is

\[
\sigma_v^{\mathrm{SL}}
=
B_\gamma(\hat\epsilon_{T,v}).
\]

This is an ablation/theory-supporting option only.

---

# 10. Performance relevance: \(p_v\)

## 10.1 Desired meaning

HERP defines \(p_v\) as:

> how useful policy updates induced by data from region \(v\) are for improving policy behavior under the target training-time reference distribution.

The held-out test set is never used to compute \(p_v\).

---

# 11. Training-time reference distribution

Maintain a small reference batch

\[
\mathcal D_{\mathrm{ref}}
\]

collected from ordinary environment resets under the current policy.

The reference distribution uses separate training-time seeds, no archived-state reset, and no HERP prioritization.

Define

\[
g_{\mathrm{ref}}
=
\nabla_{\theta_\pi}
L_{\mathrm{ref}}^\pi(\theta).
\]

Only policy parameters are used for \(p_v\); the critic is excluded.

---

# 12. PPO-consistent regional gradient

For region \(v\), collect a current-policy rollout batch \(\mathcal D_v\) and define

\[
g_v
=
\nabla_{\theta_\pi}
L_v^\pi(\theta).
\]

The implementation must use the full actor parameter set, not only the actor head.

Advantage normalization must be consistent between regional and reference gradient computation. Do not independently zero-mean every tiny region batch if that changes the update direction relative to actual PPO.

Use

\[
L_{\mathrm{sig}}(D)
=
-
\mathbb E[
\log\pi_\theta(a\mid s)\hat A
].
\]

Advantages should use a shared running or scoring-round normalization scale.

---

# 13. First-order performance relevance

For a small update

\[
\theta^+
=
\theta-\eta g_v,
\]

Taylor expansion gives

\[
L_{\mathrm{ref}}^\pi(\theta^+)
=
L_{\mathrm{ref}}^\pi(\theta)
-
\eta
g_{\mathrm{ref}}^\top g_v
+
O(\eta^2).
\]

Thus

\[
u_v^{\mathrm{dot}}
=
g_{\mathrm{ref}}^\top g_v.
\]

A scale-normalized alternative is

\[
u_v^{\mathrm{cos}}
=
\frac{
g_{\mathrm{ref}}^\top g_v
}{
\|g_{\mathrm{ref}}\|_2
\|g_v\|_2
+
\epsilon
}.
\]

Use

\[
\tilde p_v
=
[\operatorname{EMA}(u_v)]_+
+
\epsilon_p.
\]

Then normalize

\[
p_v
=
\frac{
\tilde p_v^\alpha
}{
\sum_u
\tilde p_u^\alpha
}.
\]

---

# 14. Occupancy relevance

Let

\[
d_{\mathrm{ref}}(v)
\]

be the fraction of reference states assigned to region \(v\). Then

\[
p_v^{\mathrm{occ}}
=
\frac{
d_{\mathrm{ref}}(v)+\epsilon
}{
\sum_u d_{\mathrm{ref}}(u)+K\epsilon
}.
\]

Because pilot evidence may favor occupancy over gradient alignment, occupancy remains a first-class estimator.

---

# 15. Hybrid relevance

Combine occupancy and gradient relevance:

\[
\boxed{
\tilde p_v
=
(p_v^{\mathrm{occ}}+\epsilon)^\eta
(p_v^{\mathrm{grad}}+\epsilon)^{1-\eta}
}
\]

with \(\eta\in[0,1]\), then normalize over regions.

Ablate

\[
\eta\in\{0,0.5,1\}.
\]

---

# 16. Influence-function option

A second-order variant is

\[
u_v^{\mathrm{IF}}
=
g_{\mathrm{ref}}^\top
(H+\lambda I)^{-1}
g_v.
\]

For PPO, use a diagonal empirical Fisher approximation:

\[
u_v^{\mathrm{Fisher}}
=
g_{\mathrm{ref}}^\top
\frac{
g_v
}{
\operatorname{diag}(F)+\lambda
}.
\]

Full IHVP is not required for ICRA.

---

# 17. Why \(n_v\propto p_v\sigma_v\)

Suppose each region produces stochastic policy-gradient samples \(G_v(\tau)\) with mean

\[
\mu_v
=
\mathbb E[G_v(\tau)\mid v].
\]

Assume the target performance-relevant gradient has form

\[
g^\star
=
\sum_v p_v\mu_v.
\]

Using \(n_v\) independent rollouts from region \(v\),

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

Ignoring cross-region covariance,

\[
\mathbb E\|\hat g-g^\star\|_2^2
=
\sum_v
\frac{
p_v^2\sigma_{g,v}^2
}{
n_v
}.
\]

Solve

\[
\min_{\{n_v\}}
\sum_v
\frac{
p_v^2\sigma_{g,v}^2
}{
n_v
}
\]

subject to

\[
\sum_v n_v=B.
\]

The stationary solution is

\[
\boxed{
n_v^\star
=
B
\frac{
p_v\sigma_{g,v}
}{
\sum_u p_u\sigma_{g,u}
}
}.
\]

This is the Neyman-allocation structure.

---

# 18. Connecting future branching to gradient variance

Assume

\[
G_v(\tau)=h_\theta(Z_v),
\qquad
Z_v=\Psi(\tau^+),
\]

and local Lipschitzness:

\[
\|h_\theta(z)-h_\theta(z')\|_2
\le
L_g\|z-z'\|_2.
\]

Then

\[
\frac12
\mathbb E\|G-G'\|_2^2
\le
L_g^2
\frac12
\mathbb E\|Z-Z'\|_2^2.
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
\boxed{
\sigma_{g,v}
\le
L_g\sigma_{\mathrm{traj},v}
}.
\]

Thus future-trajectory branching is a proxy for regional gradient uncertainty.

---

# 19. Practical priority

Use

\[
q_v
=
\frac{
(p_v+\epsilon_p)^\alpha
(\sigma_v+\epsilon_\sigma)^\beta
}{
\sum_u
(p_u+\epsilon_p)^\alpha
(\sigma_u+\epsilon_\sigma)^\beta
}.
\]

Add uniform mixing

\[
q_v
\leftarrow
(1-\rho)q_v
+
\frac{\rho}{K}.
\]

Optionally add staleness.

Recommended defaults:

\[
\alpha=\beta=1,
\qquad
\rho=0.1.
\]

---

# 20. HERP algorithm

At each training round:

1. Collect ordinary PPO rollouts.
2. Add visited states to the archive.
3. Periodically collect a reference batch.
4. Select candidate regions.
5. For each candidate:
   - restore an archived state,
   - collect short future probes,
   - estimate \(\sigma_v\),
   - collect a signature batch,
   - estimate \(p_v\).
6. Compute \(q_v\propto p_v\sigma_v\).
7. Spend remaining interaction budget on restored-state rollouts sampled from \(q_v\).
8. Train PPO on ordinary + allocated rollouts.
9. Repeat.

Reference rollouts are excluded from PPO optimization. Probe rollouts are counted in the interaction budget.

---

# 21. ICRA benchmark scope

Use three benchmark groups.

## ManiSkill

- PushCube-v1
- PickCube-v1
- StackCube-v1
- PegInsertionSide-v1

## Meta-World

- button-press-v3
- drawer-open-v3
- pick-place-v3
- peg-insert-side-v3

## Gymnasium Robotics / Fetch

- FetchPush-v4
- FetchPickAndPlace-v4

This gives 10 tasks across two simulator families and multiple manipulation settings without overengineering the codebase.

---

# 22. Main baselines

Main table:

1. PPO
2. RND
3. Disagreement
4. HERP-\(\sigma\)
5. HERP-\(p\)
6. HERP

Optional if time allows:

7. Go-Explore-style region sampling
8. PLR-style region replay

All methods use the same PPO backbone and equal environment-step budget.

---

# 23. Mechanism validation

## 23.1 \(\sigma_v\)

Compare cheap \(K=4\) estimates to independent \(K=64\) estimates over archived regions and report Spearman correlation.

Optionally test

\[
\rho(
\sigma_{\mathrm{traj},v},
\sigma_{g,v}
).
\]

## 23.2 \(p_v\)

Use a controlled PPO-delta experiment. For each region \(v\):

1. clone the checkpoint twice,
2. update copy A with a matched base PPO batch,
3. update copy B with base + region data,
4. evaluate both on identical reference seeds.

Define

\[
\Delta_v
=
J_{\mathrm{ref}}(\theta_{B})
-
J_{\mathrm{ref}}(\theta_{A}).
\]

Report Spearman correlation between \(\Delta_v\) and occupancy, cosine, dot, Fisher, and hybrid relevance.

---

# 24. Required ablations

Core factorization:

\[
\text{PPO}
\quad
\text{vs}
\quad
\text{HERP-}\sigma
\quad
\text{vs}
\quad
\text{HERP-}p
\quad
\text{vs}
\quad
\text{HERP}.
\]

Estimator ablations only on representative tasks.

\(\sigma_v\):

- pairwise future dispersion,
- branch decomposition,
- return variance,
- disagreement.

\(p_v\):

- occupancy,
- cosine,
- dot,
- diagonal Fisher,
- hybrid.

Hyperparameter ablations only on one or two tasks:

- \(H\in\{4,8,16\}\),
- \(K\in\{2,4,8\}\),
- allocated fraction \(\in\{0.1,0.25,0.5\}\).

---

# 25. Claims not to make

Do not claim:

- raw trajectory distance equals transition-kernel TV distance,
- gradient alignment is a full influence function,
- the 32k-step pilot establishes final method ranking,
- probes are free,
- HERP is universally optimal.

---

# 26. References

- Lobel, S. and Parr, R. An Optimal Tightness Bound for the Simulation Lemma. RLC 2024.
- Pathak, D. et al. Curiosity-driven Exploration by Self-supervised Prediction. ICML 2017.
- Burda, Y. et al. Exploration by Random Network Distillation. ICLR 2019.
- Pathak, D., Gandhi, D., Gupta, A. Self-Supervised Exploration via Disagreement. ICML 2019.
- Ecoffet, A. et al. First return, then explore. Nature 2021.
- Jiang, M., Grefenstette, E., Rocktäschel, T. Prioritized Level Replay. ICLR 2021.
