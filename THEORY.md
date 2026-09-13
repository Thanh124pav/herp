# HERP v3 — THEORY.md

> **Version:** September 2026  
> **Target:** ICRA 2027  
> **Scope:** online robotic reinforcement learning under a fixed environment-interaction budget.

---

# 0. Core idea

HERP allocates a finite online interaction budget across behaviorally meaningful trajectory regions according to

\[
\boxed{n_v^\star \propto p_v\,\sigma_v}
\]

where:

- \(p_v\) measures how useful a policy update from region \(v\) is for the ordinary-reset PPO objective;
- \(\sigma_v\) measures how diverse the next fixed \(M\) steps can be when rollout starts from region \(v\);
- regions are defined from **behavior-consistent trajectory chains**, not from arbitrary geometric state clusters.

The environment initial-state distribution is represented explicitly by a root region \(v_0\). Hence HERP decides, in one allocator, whether the next interaction should begin from a normal environment reset or from a previously reached region.

---

# 1. Research gap

Standard online PPO repeatedly samples

\[
s_0\sim\rho_0
\]

and spends interaction budget according to the occupancy induced by the current policy. It does not explicitly ask which already reached decision contexts deserve additional interaction.

A pure state-space partition is problematic in robotics. A robot may execute the same qualitative behavior for a long time while absolute state variables change continuously. Moving in the same direction for \(1\) m or \(1000\) m should not automatically create many exploration regions.

HERP separates three questions:

1. **Partition:** where does a behavioral regime begin and end?
2. **Exploration need:** how diverse are future continuations from that region?
3. **Performance relevance:** would learning from that region help downstream PPO performance?

These correspond to

\[
\mathcal R,\qquad \sigma_v,\qquad p_v.
\]

---

# 2. Problem setting

Consider an MDP

\[
\mathcal M=(\mathcal S,\mathcal A,P,r,\gamma,\rho_0)
\]

with stochastic policy

\[
\pi_\theta(a\mid s).
\]

A trajectory is

\[
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_T).
\]

At acquisition round \(t\), HERP maintains

\[
\mathcal V_t=\{v_0,v_1,\ldots,v_{K_t}\}.
\]

The distinguished root region \(v_0\) represents ordinary reset:

\[
s_0\sim\rho_0.
\]

For \(v>0\), the region stores restorable simulator snapshots observed during previous online interaction.

Let \(M\) be a fixed continuation length. One allocation unit is one \(M\)-step rollout fragment. If a round has budget \(B_t\) transitions, define

\[
N_t=\left\lfloor\frac{B_t}{M}\right\rfloor,
\qquad
\sum_{v\in\mathcal V_t} n_v=N_t.
\]

All training-time interactions used by HERP, including reference rollouts for \(p_v\), count toward the reported interaction budget.

---

# 3. HERP v3 loop

\[
\boxed{
\text{rollout}
\rightarrow
\text{behavioral segmentation}
\rightarrow
\text{cross-trajectory chain clustering}
\rightarrow
(\hat p_v,\hat\sigma_v)
\rightarrow
\text{allocation}
\rightarrow
\text{new rollout fragments}
\rightarrow
\text{PPO update}
}
\]

HERP does not replace PPO. PPO remains the policy optimizer. HERP changes where interaction is acquired.

---

# 4. Behavior-aware partition

## 4.1 Why chains rather than individual states?

Let

\[
\tau=(s_0,a_0,s_1,a_1,\ldots,s_T).
\]

Suppose a robot stays in the same behavioral mode for a long interval. Its absolute state may move far in Euclidean space, yet the local policy regime remains essentially unchanged.

State-radius clustering can therefore create many regions simply because the trajectory is long. HERP instead keeps one chain until a behavioral boundary is detected.

The intended invariance is:

> A long but behaviorally consistent motion should not create more regions merely because it spans a larger geometric distance.

---

# 5. Stage I — split each trajectory into behavior-consistent chains

Let

\[
z_t=\phi(s_t)
\]

be normalized state features.

For continuous-control PPO, assume

\[
\pi_\theta(\cdot\mid s_t)
=
\mathcal N(\mu_t,\Sigma_t).
\]

Define policy-distribution change

\[
d_t^\pi
=
D_\pi\!\left(
\pi_\theta(\cdot\mid s_{t-1}),
\pi_\theta(\cdot\mid s_t)
\right).
\]

For diagonal Gaussian PPO, HERP v3 uses symmetric KL by default.

Define local normalized state displacement

\[
d_t^s=\|z_t-z_{t-1}\|_2.
\]

The boundary score is

\[
b_t
=
\lambda_\pi\widetilde d_t^\pi
+
\lambda_s\widetilde d_t^s,
\qquad
\lambda_\pi+\lambda_s=1,
\]

where each scalar channel is normalized by running statistics.

A temporal boundary is declared when

\[
b_t>Q_q(\{b_j\}),
\]

with percentile \(q\).

The default main method uses

\[
\lambda_\pi=1,\qquad \lambda_s=0,
\]

while action+state scoring remains an ablation.

This yields chains

\[
C_m=(s_{\ell_m},a_{\ell_m},\ldots,s_{r_m}).
\]

Chain duration is determined by behavioral consistency, not path length.

---

# 6. Stage II — cluster chains across trajectories

Temporal segmentation only divides one trajectory. HERP then groups equivalent chains across different trajectories.

For a chain starting at state \(s_i\), define the entry context

\[
h(C)=
\begin{bmatrix}
z_{i-1}\\z_i\end{bmatrix}.
\]

For

\[
C_a=s^{(1)}_{i:j},
\qquad
C_b=s^{(2)}_{k:l},
\]

define

\[
\boxed{
D_{\mathrm{chain}}(C_a,C_b)
=
\frac12
\left[
d(z_i,z_k)+d(z_{i-1},z_{k-1})
\right]
}
\]

with normalized Euclidean distance \(d\).

Only the entry context is used. The distance therefore does not grow simply because one chain lasts longer than another.

Online nearest-centroid clustering in \(h(C)\)-space creates regions \(v\).

A HERP region means:

> a collection of behaviorally coherent trajectory chains entered from similar local state transitions.

---

# 7. Region graph

Segmented trajectories induce a directed graph

\[
\mathcal G=(\mathcal V,\mathcal E).
\]

If one chain-region is followed by another in a trajectory, add the corresponding directed edge. Consecutive duplicate region IDs are collapsed.

The root region \(v_0\) connects to first discovered chain-regions after ordinary reset.

The graph can contain shared descendants and cycles. It is used for transition statistics and root uncertainty diagnostics, not to constrain policy execution.

---

# 8. Why fixed \(M\) for \(\sigma_v\)?

If one region is compared over \(K\) future steps and another over \(L>K\), the latter can appear more diverse simply because trajectories have had more time to separate.

HERP therefore compares every region using the same future context window \(M\).

This removes a systematic preference for long chains and makes region uncertainty scores comparable.

---

# 9. Fixed-\(M\) continuation representation

A rollout fragment from region \(v\) is

\[
\xi=(s_0,a_0,s_1,a_1,\ldots,s_M).
\]

Define per-step trajectory feature

\[
y_m=\psi(s_m,a_m).
\]

The default is

\[
\psi(s_m,a_m)
=
\begin{bmatrix}
\phi(s_m)\\
\lambda_a\widetilde a_m
\end{bmatrix},
\]

where \(\widetilde a_m\) is the normalized action and \(\lambda_a\) is an ablation coefficient.

For two continuations \(\xi,\xi'\), define

\[
\boxed{
d_M(\xi,\xi')
=
\frac1M
\sum_{m=1}^{M}
\|y_m-y_m'\|_2^2.
}
\]

---

# 10. Oracle future-dispersion quantity

Let \(\Xi_v\) denote the distribution of \(M\)-step continuations generated by restoring a snapshot from region \(v\) and following the current policy.

Define

\[
\boxed{
q_v
=
\frac12
\mathbb E_{\xi,\xi'\stackrel{iid}{\sim}\Xi_v}
[d_M(\xi,\xi')]
}
\]

and

\[
\boxed{\sigma_v=\sqrt{q_v}.}
\]

Under finite second moments, \(q_v\) is the trace of the covariance of the normalized fixed-\(M\) future representation.

The key point is that this is the target quantity itself, not an upper-bound proxy.

---

# 11. Direct empirical estimator

Suppose region \(v\) has \(K_v\ge2\) continuation fragments

\[
\xi_{v,1},\ldots,\xi_{v,K_v}.
\]

Use

\[
\boxed{
\hat q_v
=
\frac{1}{K_v(K_v-1)}
\sum_{1\le i<j\le K_v}
 d_M(\xi_{v,i},\xi_{v,j}).
}
\]

Because

\[
q_v=\frac12\mathbb E[d_M(\xi,\xi')],
\]

this U-statistic is unbiased:

\[
\boxed{\mathbb E[\hat q_v]=q_v.}
\]

The square-root estimator \(\sqrt{\hat q_v}\) has the ordinary finite-sample square-root bias. Therefore HERP performs regression and convergence analysis on

\[
q_v=\sigma_v^2
\]

rather than directly on \(\sigma_v\).

---

# 12. Variance floor

With few samples, empirical dispersion can be exactly zero. Define

\[
\varepsilon_\sigma>0
\]

and use for allocation

\[
\boxed{
\sigma_v^{\mathrm{alloc}}
=
\sqrt{\tilde q_v+\varepsilon_\sigma^2}
}
\]

where \(\tilde q_v\) is the direct or shrinkage estimate defined later.

The floor is not interpreted as observed stochasticity. It prevents finite data from making a region permanently dead.

If every region has zero empirical dispersion, then all receive the same common \(\varepsilon_\sigma\) before relevance is applied.

---

# 13. Root region \(v_0\)

The initial-state distribution itself is an allocation candidate.

Selecting \(v_0\) means:

1. reset normally;
2. sample \(s_0\sim\rho_0\);
3. follow the current policy for \(M\) transitions;
4. partition and archive any newly observed behavior chains.

Thus root allocation generates new trajectories and can expand the support of discovered regions.

No permanent hard constraint

\[
n_0\ge k
\]

is required.

---

# 14. Root uncertainty and total variance

HERP v3 defines root uncertainty from the child-region decomposition itself. Direct root continuations are still collected, but their direct dispersion is used as an oracle-like diagnostic to validate the recursive estimate rather than as the main root score.

Let root children be \(c\in\mathcal C(0)\), with empirical probabilities

\[
w_{c\mid0},
\qquad
\sum_c w_{c\mid0}=1.
\]

Let a child future representation have mean \(\mu_c\) and covariance trace \(q_c\). Then

\[
\mu_0=\sum_c w_{c\mid0}\mu_c
\]

and the law of total variance gives

\[
\boxed{
q_0
=
\sum_c w_{c\mid0}q_c
+
\sum_c w_{c\mid0}\|\mu_c-\mu_0\|_2^2.
}
\]

The first term is within-child uncertainty. The second term is between-child branching.

If all children are identical and each empirical variance is zero, adding the same floor \(\varepsilon_\sigma^2\) yields

\[
q_0^{\mathrm{alloc}}=\varepsilon_\sigma^2.
\]

Hence the root remains allocatable without receiving an artificial bonus.

The main allocator uses the total-variance root estimate above. Direct root dispersion is logged in parallel as a validation signal. If the root has no observed children yet, set its empirical component to zero so that the common variance floor gives \(\sigma_0=\varepsilon_\sigma\). For child within-region terms, use each child's current combined estimate \(\tilde q_c\); for the between-child term, use recent empirical child future-feature means.

---

# 15. Learned variance predictor \(f\)

## 15.1 Motivation

The direct estimator is unbiased for \(q_v\), but its finite-sample variance can be large when only a few rollout fragments are available.

HERP therefore optionally learns

\[
f_\omega(x_v)\approx q_v
\]

to share statistical information across similar regions.

This is an estimator inside the \(\sigma\)-module, not an independent fourth contribution.

---

# 16. Predictor input

The predictor may only use information available before new rollouts are allocated.

Use the interpretable vector

\[
\boxed{
x_v=
[
1,
\overline d_v^\pi,
\overline H_v^\pi,
\overline d_v^s,
\log(1+N_v)
]^\top.
}
\]

Here:

- \(\overline d_v^\pi\): mean policy-distribution change around entries into region \(v\);
- \(\overline H_v^\pi\): mean action entropy or Gaussian log-scale statistic;
- \(\overline d_v^s\): mean normalized state displacement at the boundary;
- \(N_v\): number of observed entries into the region.

Do not include chain length in the default predictor.

The default model is linear ridge regression:

\[
\boxed{f_\omega(x)=\omega^\top x.}
\]

At inference, negative predictions are clipped to zero before the variance floor is added.

---

# 17. Predictor labels

Whenever HERP allocates fresh rollout fragments to region \(v\), compute a fresh direct estimator

\[
\hat q_{v,t}.
\]

Append

\[
(x_{v,t},\hat q_{v,t})
\]

to the regression dataset.

Fit

\[
\boxed{
\hat\omega
=
\arg\min_\omega
\sum_j w_j
(\omega^\top x_j-\hat q_j)^2
+
\lambda_f\|\omega\|_2^2.
}
\]

A simple confidence weight is

\[
w_j=K_j,
\]

where \(K_j\) is the number of fresh continuations used to form the label.

---

# 18. Why noisy labels can still converge

Assume

\[
\mathbb E[\hat q\mid x]=q(x).
\]

Under squared loss, the Bayes-optimal predictor is

\[
f^\star(x)=\mathbb E[\hat q\mid x]=q(x).
\]

Thus unbiased label noise increases finite-sample estimation variance but does not move the population optimum.

For the linear class

\[
\mathcal F_{\rm lin}=\{x\mapsto\omega^\top x\},
\]

standard regression assumptions imply convergence to the best linear projection

\[
f^\star_{\rm lin}
=
\arg\min_{f\in\mathcal F_{\rm lin}}
\mathbb E[(f(x)-q(x))^2].
\]

Under linear realizability

\[
q(x)=\omega^{\star\top}x,
\]

persistent excitation, finite moments, and vanishing regularization,

\[
\hat\omega_N\rightarrow\omega^\star.
\]

Without realizability, HERP must only claim convergence to the best linear approximation.

---

# 19. Shrinkage combination

The learned predictor should not immediately replace direct evidence.

Let \(K_v\) be the number of recent direct continuation samples. Define

\[
\lambda_v=\frac{K_v}{K_v+\kappa}.
\]

Use

\[
\boxed{
\tilde q_v
=
\lambda_v\hat q_v
+
(1-\lambda_v)\hat q_v^{f},
}
\]

where

\[
\hat q_v^f=\max(0,f_\omega(x_v)).
\]

Then

\[
\boxed{
\sigma_v^{\mathrm{HERP}}
=
\sqrt{\max(0,\tilde q_v)+\varepsilon_\sigma^2}.
}
\]

Interpretation:

- few direct samples \(\Rightarrow\) rely more on shared predictor;
- many direct samples \(\Rightarrow\) approach direct empirical variance.

Main ablations are direct only, linear predictor only, and shrinkage.

---

# 20. Performance relevance \(p_v\)

HERP v3 retains gradient alignment as the main relevance estimator.

Maintain a training-time reference batch

\[
\mathcal D_{\rm ref}
\]

collected only from ordinary resets under the current policy.

Compute PPO-consistent actor gradient signature

\[
g_{\rm ref}
=
\nabla_{\theta_\pi}L_{\rm sig}(\mathcal D_{\rm ref}).
\]

For region \(v\), using transitions acquired from that source region,

\[
g_v
=
\nabla_{\theta_\pi}L_{\rm sig}(\mathcal D_v).
\]

The default relevance score is cosine alignment:

\[
\boxed{
u_v
=
\frac{g_{\rm ref}^\top g_v}
{\|g_{\rm ref}\|_2\|g_v\|_2+\varepsilon_g}.
}
\]

Maintain a positive EMA and floor:

\[
\boxed{
p_v
=
\left([
\operatorname{EMA}(u_v)
]_+
+\varepsilon_p\right)^{\alpha_p}.
}
\]

No separate normalization is required because the final allocator normalizes \(p_v\sigma_v\).

The root region is scored in exactly the same way using ordinary-reset fragments.

---

# 21. Why gradient alignment is relevant

For a small update from region \(v\),

\[
\theta^+=\theta-\eta g_v.
\]

First-order expansion gives

\[
L_{\rm ref}(\theta^+)
=
L_{\rm ref}(\theta)
-
\eta g_{\rm ref}^\top g_v
+
O(\eta^2).
\]

Thus positive gradient alignment means that data from the region points in a locally useful direction for the reference PPO objective.

Cosine alignment is used in the main method because it removes gradient magnitude as a confounder.

---

# 22. Deriving \(n_v\propto p_v\sigma_v\)

Assume a performance-relevant target gradient can be decomposed as

\[
g^\star=\sum_v p_v\mu_v,
\qquad
\mu_v=\mathbb E[G_v(\xi)\mid v].
\]

With \(n_v\) independent rollout fragments from each region,

\[
\hat g
=
\sum_v p_v
\left(
\frac1{n_v}
\sum_{j=1}^{n_v}G_v(\xi_{v,j})
\right).
\]

Ignoring cross-region covariance,

\[
\mathbb E\|\hat g-g^\star\|_2^2
=
\sum_v
\frac{p_v^2\sigma_{g,v}^2}{n_v},
\]

where

\[
\sigma_{g,v}^2
=
\operatorname{tr}\operatorname{Cov}[G_v(\xi)\mid v].
\]

If \(\sigma_{g,v}\) were known exactly, Neyman allocation gives

\[
n_v^\star\propto p_v\sigma_{g,v}.
\]

HERP substitutes fixed-window trajectory dispersion for the expensive gradient covariance through the following assumption.

---

# 23. Connecting future dispersion to gradient variance

Assume the per-fragment policy-gradient map is locally Lipschitz in the fixed-\(M\) trajectory metric:

\[
\|G_v(\xi)-G_v(\xi')\|_2
\le
L_G\sqrt{d_M(\xi,\xi')}.
\]

Using

\[
\operatorname{tr}\operatorname{Cov}(G)
=
\frac12\mathbb E\|G-G'\|_2^2,
\]

we obtain

\[
\sigma_{g,v}^2
\le
L_G^2
\frac12\mathbb E[d_M(\xi,\xi')]
=
L_G^2 q_v.
\]

Therefore

\[
\boxed{\sigma_{g,v}\le L_G\sigma_v.}
\]

Minimizing the resulting MSE upper bound gives

\[
\boxed{n_v^\star\propto p_v\sigma_v.}
\]

The common constant \(L_G\) cancels.

---

# 24. Integer allocation and the uninformative limit

Define

\[
w_v=p_v\sigma_v^{\rm HERP}.
\]

Then

\[
q_v^{\rm alloc}
=
\frac{w_v}{\sum_u w_u}.
\]

Allocate the integer rollout units by multinomial sampling or largest-remainder rounding.

HERP does not need a permanent root lower bound, uniform-mix heuristic, or staleness bonus in the main method.

If all raw relevance signals are non-positive and all empirical dispersions are zero, then

\[
p_v=\varepsilon_p,
\qquad
\sigma_v=\varepsilon_\sigma
\]

for every active region, including the root. Hence

\[
\boxed{
q_v^{\rm alloc}=\frac1{|\mathcal V|}.
}
\]

Therefore

\[
\boxed{
\text{no discriminative evidence}
\Longrightarrow
\text{uniform allocation}
}
\]

emerges from the score itself rather than an exception branch.

---

# 25. New-region discovery

HERP does not only branch from old states.

Support can expand through:

\[
\boxed{
\text{root exploration}
+
\text{descendant branching}.
}
\]

A root rollout can discover completely new chains. A continuation from an existing region can discover unseen descendants. Both are generated by the same allocator.

---

# 26. PPO compatibility

HERP rollout fragments can be truncated after \(M\) steps and bootstrapped with the critic:

\[
\hat R_t
=
r_t+\gamma r_{t+1}+\cdots+
\gamma^{M-t}V(s_M).
\]

GAE is computed within each fragment. True environment termination sets bootstrap value to zero; artificial HERP truncation does not.

Restored-state rollouts intentionally change the acquisition distribution relative to ordinary PPO occupancy. HERP therefore distinguishes:

- **acquisition distribution:** selected by HERP;
- **reference distribution:** ordinary resets;
- **optimizer:** PPO.

The role of \(p_v\) is precisely to prefer acquired data whose induced policy update is aligned with the ordinary-reset reference objective.

---

# 27. Main propositions for the paper

## Proposition 1 — unbiased fixed-window variance estimator

Under i.i.d. continuation sampling,

\[
\mathbb E[\hat q_v]=q_v.
\]

## Proposition 2 — horizon fairness

For fixed \(M\), two regions with identical \(M\)-step future distributions have the same \(q_v\), regardless of how long their underlying behavior chains continue after the window.

## Proposition 3 — degenerate-limit stability

If all raw dispersion and relevance signals vanish, positive common floors imply uniform allocation over all active regions including \(v_0\).

## Proposition 4 — predictor consistency

Under unbiased labels, finite moments, persistent excitation, and linear realizability,

\[
\hat\omega_N\to\omega^\star.
\]

Without realizability, convergence is only to the best linear \(L_2\) projection.

## Proposition 5 — trajectory-dispersion allocation bound

Under the local Lipschitz-gradient assumption,

\[
\sigma_{g,v}\le L_G\sigma_v,
\]

and minimization of the corresponding variance upper bound yields

\[
n_v^\star\propto p_v\sigma_v.
\]

---

# 28. Required ablations

## Partition

- raw state radius clustering;
- fixed-length temporal windows;
- policy-distribution boundary only;
- policy + state boundary;
- HERP chain segmentation + entry-context clustering.

## \(\sigma\)

- one-step state dispersion;
- end-of-chain dispersion;
- variable-length chain dispersion;
- fixed \(M\)-step direct dispersion;
- linear predictor only;
- direct + linear shrinkage predictor.

## \(p\)

- uniform;
- occupancy;
- gradient dot product;
- gradient cosine alignment;
- controlled PPO-delta oracle for mechanism analysis only.

## Allocation

- uniform;
- \(\sigma\) only;
- \(p\) only;
- \(p\sigma\).

## Root handling

- fixed global/local mixture;
- hard root lower bound;
- root as an ordinary allocation candidate.

---

# 29. Evaluation quantities

Primary result:

- task success / return versus **total environment interactions**.

Mechanism diagnostics:

- low-sample versus high-sample \(q_v\) correlation;
- linear predictor versus high-sample \(q_v\) correlation;
- \(p_v\) versus controlled PPO-delta correlation;
- number of behavior chains;
- number of regions;
- chain-length distribution;
- root allocation fraction;
- allocation entropy;
- newly discovered regions per million environment steps.

---

# 30. Contribution hierarchy

HERP v3 should be presented as three contributions only.

### Contribution 1 — behavior-aware trajectory partition

Partition trajectories by changes in the policy regime and cluster chain-entry contexts across trajectories. This prevents geometric region explosion during long but behaviorally consistent motion.

### Contribution 2 — statistically efficient future-dispersion estimation

Define \(\sigma_v\) through fixed-\(M\) future trajectory dispersion, estimate \(q_v=\sigma_v^2\) directly with an unbiased U-statistic, and optionally denoise finite-sample estimates using a lightweight interpretable linear predictor.

### Contribution 3 — performance-aware interaction allocation

Use PPO gradient alignment as relevance and derive

\[
n_v^\star\propto p_v\sigma_v.
\]

The linear predictor \(f\) is part of Contribution 2, not an independent fourth principle.

---

# 31. Default implementation values

Use as coding defaults, not immutable paper constants:

\[
M=32,
\qquad
q_{\rm boundary}=0.90,
\]

\[
\lambda_\pi=1,
\qquad
\lambda_s=0,
\]

\[
\varepsilon_\sigma=10^{-3},
\qquad
\varepsilon_p=10^{-3},
\]

\[
\alpha_p=1,
\qquad
\kappa=8.
\]

Tune the chain-clustering radius on development tasks and then hold it fixed for main experiments.

---

# 32. Non-claims

HERP v3 does not claim:

- Euclidean state distance alone captures semantic skill identity;
- high policy entropy always implies useful exploration;
- the linear predictor universally represents the true variance function;
- restored-state rollouts follow ordinary PPO occupancy;
- future trajectory dispersion equals gradient standard deviation exactly;
- the root must receive a manually fixed fraction of budget.

---

# 33. Compact algorithm

```text
Initialize PPO policy πθ
Initialize root region v0
Initialize chain region archive
Initialize linear variance predictor fω

Warm up from root until enough chains/regions exist.

repeat until interaction budget is exhausted:

    1. Build ordinary-reset reference batch and g_ref.

    2. For each active region v:
       - for v > 0: estimate recent direct q_hat_v from fixed-M fragments
       - compute predictor input x_v and predict q_f_v
       - shrink direct and predicted variance
       - for v = 0: aggregate child means and child q values by total variance
       - obtain sigma_v with the same variance floor
       - compute gradient-alignment relevance p_v

    3. Allocate rollout units:
           score_v = p_v * sigma_v
           n_v proportional to score_v
       with root v0 included in the same normalization.

    4. Acquire fragments:
       - v0: ordinary reset, rollout M steps
       - v>0: restore chain-entry snapshot, rollout M steps

    5. Partition new experience:
       - detect temporal behavioral boundaries
       - form chains
       - cluster chain-entry contexts
       - update region graph and snapshot archive

    6. PPO update on valid acquired transitions using fragment-local GAE.

    7. New fixed-M fragment groups provide fresh q_hat labels for fω.
       Refit/update the linear predictor.
```

---

# 34. Core intuition

> **Partition by what the robot is doing, estimate how many genuinely different futures can emerge from each decision context, measure whether learning there helps the target PPO behavior, and spend the next interaction where both signals are high.**
