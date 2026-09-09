# HERP implementation and experiment protocol (9 September 2026)

This implementation follows the operational allocation model in THEORY.md and IMPLEMENTATION.md. It is a new state-based PPO implementation, not a claim of reproducing the tuned official ManiSkill PPO benchmark. The two intrinsic-reward baselines use the same PPO network, optimizer and rollout implementation. They are state-MLP adaptations, not literal ports of Atari RND or the original TensorFlow Disagreement implementation.

## Changes addressing the supplied review

- Region gradients use actual current-policy observations, sampled actions and GAE advantages from short restored-state fragments. These fragments are reused for PPO and charged as allocated interactions. The previous artificial zero-action/unit-advantage construction is gone.
- A separate environment collects ordinary-reset reference data every `relevance_interval` updates. Its gradient is computed before updating the policy. Reference transitions are excluded from PPO, but charged to the budget. Cached reference gradients and region scores can remain stale until their refresh interval; this is deliberate and does not pretend old actions were newly sampled.
- Candidate sigma values receive tie-aware rank normalization. Allocation retains both a uniform floor and an age-based mixture. The recent/stale/random candidate selector reserves slots for each component.
- Cosine, dot-product, diagonal empirical Fisher, and occupancy relevance are implemented. The Fisher averages squared individual **score** gradients, without multiplying by advantages. Dot/Fisher positive scores use a robust common scale across candidates.
- RND-PPO and ensemble-Disagreement-PPO are implemented. Five independently initialized forward models use bootstrap masks. Prediction error / ensemble variance supplies a normalized intrinsic bonus; the RND target is frozen. The intrinsic coefficient is fixed at 0.01 in the pilot, not separately tuned by task.
- Optional archive controls are explicitly named Go-Explore-style and PLR-style region replay. They are adaptations, not reproductions of those original algorithms.
- Gaussian sampled actions are stored unchanged for PPO log likelihoods; only the action passed to the environment is clipped. Truncations bootstrap from the final state, true terminals do not, and GAE never leaks across an episode/reset boundary.
- Snapshots are deep-copied and restore physical state, controller state and episode clock. Region centroids are transformed when the running normalizer changes coordinates. The first 31 normalizer samples use unit scale to avoid unstable startup variance.
- Metrics average over minibatches. CSV declares evaluation fields from the first row, so later evaluation results are preserved. Per-source cumulative step counts, score summaries, priority entropy, gradient norms, allocations, wall time, per-region scores, and checkpoints are saved.
- YAML loading is explicit with repeatable `--config` arguments; later files override earlier files and command-line values override files. Unknown fields and method names fail immediately.

`*_ema_tau` is the weight on the **old** estimate. It does not denote the new-sample weight.

## Budget and evaluation

Each run ends at exactly its configured training interaction budget:

```
training interactions = normal + probe + allocated + reference
```

`allocated_frac` is the fraction reserved for all additional HERP interactions in a rollout block. Reference collection, sigma probing and gradient fragments consume this pool before further allocated rollouts. Unused budget returns to ordinary rollouts if there are no candidates. Baselines use the entire training budget for ordinary interaction. No evaluation steps are used for training or allocation; their count is logged separately. All methods evaluate at exactly the same interaction checkpoints.

Evaluation is deterministic, with ordinary reset seeds `2,000,000 ... 2,000,049`. The reported success rate is the fraction of episodes that achieve success **at any step**. Final-step success is also logged. Reference collection uses a separate seed range starting at 1,000,000. Mechanism analysis uses ranges starting at 3,000,000. Return is the actual task reward sum; there are no invented normalization constants.

The pilot uses 32,768 steps per run, three training seeds, and 50 evaluation episodes at 0, 8,192, 16,384, 24,576 and 32,768 interactions. This is an exploratory small-budget experiment, not evidence of convergence. The aggregate reports means, sample standard deviations and Student-t intervals over training seeds. Three-seed confidence intervals are imprecise. Success AUC includes the evaluation at step zero. Steps-to-threshold remain missing if a run never reaches the threshold, and are not replaced by the maximum budget.

## Commands

Use the requested environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deeplearning
bash scripts/smoke.sh
python train.py --config configs/base.yaml --config configs/pickcube.yaml --method herp
python scripts/run_herp_suite.py --tasks PushCube-v1 PickCube-v1 \
  --steps 32768 --eval-interval 8192 --eval-episodes 50 --workers 2 \
  --output-dir outputs/herp_pilot_20260909
```

The suite restarts failed/incomplete cells and skips completed ones. It does not resume optimizer/environment state mid-run. Use a new output directory for a changed protocol. Checkpoints contain policy, optimizer, archive, regionizer and RNG state for diagnostics; they are not a promise of exact training resume (intrinsic-model and environment runtime state are not serialized).

The older `scripts/run_all_experiments.sh` belongs to the separate experience-routing project and is **not** the HERP experiment runner.

## Mechanism experiments

```bash
python analysis/mechanisms.py --checkpoint PATH/checkpoint_32768.pt \
  --kind sigma --regions 50 --small-probes 4 --oracle-probes 64 \
  --output-dir outputs/mechanism_sigma
python analysis/mechanisms.py --checkpoint PATH/checkpoint_32768.pt \
  --kind p --regions 50 --eval-episodes 50 --step-size 0.001 \
  --output-dir outputs/mechanism_p
```

Sigma uses independent low-K and K=64 futures from the same saved state. It saves raw per-region estimates and actual future trajectories, with a scatter plot and low/high-dispersion path plots in shared state PCA coordinates. The primary estimator is discounted normalized-state dispersion. Branch decomposition fixes continuation policy-noise tapes so that variation across first-action groups measures controllable sensitivity; environment RNG is not rewound between repetitions. In the deterministic simulator, within-group variance should be near zero. Perturbed sigma probes are not used as if they were current-policy PPO data.

For p, a fresh ordinary-reset batch defines alignment and Fisher scores; a second independent batch defines the held-out surrogate. For each region, a fresh on-policy batch produces a gradient on the actor-head-plus-log-standard-deviation signature parameters. A clone receives one explicit SGD step on exactly those parameters. The clone and original policy are evaluated with the same 50 ordinary reset seeds. The diagnostic reports correlations against both actual paired return change and an independent importance-ratio policy surrogate change; the latter is not labeled environment return. Occupancy, cosine, dot and Fisher predictions all predict the same actual SGD update. Undefined correlations (constant values) are stored as null, never as a fabricated zero correlation.

This diagnostic measures one tiny signature-subspace update, not a full PPO optimizer update or a second-order causal proof. Correlation can be weak or negative. All diagnostic interaction counts are separate from training counts.

## Results and scope

`outputs/herp_pilot_20260909/analysis/` is generated automatically as runs complete. It contains `RESULTS.md`, `main_table.csv`, `main_table.tex`, and PDF/PNG learning curves. Only completed runs enter tables. Each run includes its exact config and SHA-256 hashes of implementation files.

Wall times include initialization and evaluation while other local jobs run concurrently. They are descriptive runtime measurements, not a controlled single-job compute-overhead comparison.

A small-budget pilot cannot substantiate an ICRA claim of improving final benchmark performance. In particular, if every method has zero success, changes in dense return are evidence of early learning only. StackCube and PegInsertion, larger interaction budgets, five seeds, hyperparameter robustness, and multiple mechanism checkpoints remain additional experiments, not results implied by the runner existing.

## Baseline sources

- RND reference implementation: https://github.com/openai/random-network-distillation
- CleanRL PPO/RND documentation: https://github.com/vwxyzjn/cleanrl/blob/master/docs/rl-algorithms/ppo-rnd.md
- Disagreement reference implementation: https://github.com/pathak22/exploration-by-disagreement
- ManiSkill PPO reference: https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo

Implementation verification on this host: 65 project tests passed. Restoration of 20 archived states per task on PushCube and PickCube had maximum observation error 1.19e-7, identical next observations under the same action, and correct episode clocks. An isolated 1,537-step smoke run verified odd budgets, evaluation boundaries, and a single candidate. This does not verify every stochastic task or GPU backend.

The exact pilot implementation is saved in `outputs/herp_pilot_20260909/frozen_source/`. After all 36 runs had loaded that implementation, the delivery code received a candidate-cap boundary fix and additional input validation, already tested in an isolated copy. The pilot uses four candidates and valid default inputs, so those boundary changes do not alter its execution path. All 36 run provenance files have identical implementation hashes.
