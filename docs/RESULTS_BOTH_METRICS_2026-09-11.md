# HERP results — success_once vs success_at_end (2026-09-11)

Re-evaluated all learnable checkpoints with **two metrics** (fixed eval):

- **success_once** — reached success at any step during the episode.
- **success_at_end** — success at the terminal step (does the policy *hold* it?).

`success_at_end` reads `info["final_info"]` (the vec env auto-resets on
truncation, so `info["success"]` at the done step is the *next* episode's).
See commit `c1c46ac`. **W&B:** https://wandb.ai/hust_edu_vn/herp ; videos in run
`showcase-all-methods` (`i969p92b`).

## ManiSkill — once / at_end (100 eval episodes, seed 0)

| Task | ppo | rnd | disagreement | herp_sigma | herp_p | herp |
|---|---|---|---|---|---|---|
| PushCube | 1.00/0.52 | 1.00/0.96 | 1.00/0.97 | 0.95/0.87 | 0.87/0.87 | 0.91/0.87 |
| PickCube | 1.00/0.99 | 0.98/0.97 | 1.00/1.00 | 1.00/0.95 | 0.99/0.98 | 1.00/1.00 |
| LiftPegUpright | 0.00/0.00 | 0.70/0.15 | 0.98/**0.00** | 0.59/0.10 | 0.92/**0.00** | 0.11/0.03 |
| PlaceSphere | 0.03/0.01 | 0.00/0.00 | 0.03/0.01 | 0.01/0.00 | **0.81/0.33** | 0.00/0.00 |
| PokeCube | 1.00/1.00 | 0.98/0.77 | 1.00/1.00 | 1.00/0.99 | 1.00/0.88 | 0.62/0.62 |
| StackCube | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

## Meta-World / Fetch — once / at_end (30 eval episodes, seed 0, 1M budget)

| Task | ppo | rnd | disagreement | herp_sigma | herp_p | herp |
|---|---|---|---|---|---|---|
| button-press-v3 | 0.50/0.23 | 0.50/0.13 | 0.87/0.10 | **1.00/1.00** | 1.00/0.27 | 1.00/0.27 |
| drawer-open-v3 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| pick-place-v3 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| peg-insert-side-v3 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| FetchPush-v4 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| FetchPickAndPlace-v4 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

(All-zero Meta-World/Fetch cells were recorded as 0/0 without a rollout. Fetch
had a ~0.10–0.14 `once` in the training logs but re-eval gives 0 — that was
eval-noise/floor, not learning.)

## Reading (once vs at_end changes the story)

- **Genuinely solved & held** (at_end high): **PickCube**, **PokeCube** (most
  methods ~0.9–1.0). Best demo material.
- **PushCube**: everyone reaches the goal; **ppo drifts out** (at_end 0.52) while
  rnd/disagreement/herp hold (~0.87–0.97).
- **LiftPegUpright — mirage**: high `once` (disag 0.98, herp_p 0.92) collapses to
  **at_end ≈ 0 for all** — the peg is stood up momentarily then topples. Do **not**
  present this as a success.
- **PlaceSphere — herp_p is the real headline**: the only method that does
  anything (0.81 once, **0.33 at_end**); every other method ≈ 0 on both metrics.
- **button-press (Meta-World) — herp_sigma is the standout**: uniquely presses
  **and holds** (1.00/1.00); herp_p/herp press then release (1.00/0.27);
  ppo/rnd/disagreement weak.
- **Full `herp` (p·σ) is generally the weakest HERP variant** — the ablations
  `herp_p` and `herp_sigma` each win different tasks, but combining them (`herp`)
  underperforms. Worth investigating for the paper.
- **PegInsertionSide**: staged PPO 5M→75M, **0.0 throughout** — PPO alone does
  not solve it. **StackCube**: 0 at 5M for all — needs a larger budget.

## Which ablation helps where
- `herp_p` (relevance-weighted): decisive on **PlaceSphere** (hard grasp+place).
- `herp_sigma` (variance/exploration): decisive on **button-press** (hold).
- Neither dominates everywhere, and the full method does not combine them well
  yet — a concrete direction for the next iteration.

## Caveats
- **Single seed (0).** Fair (same backbone/budget/seed) but not statistically
  conclusive — confirm the discriminative results (PlaceSphere herp_p,
  button-press herp_sigma) with seeds 1–2.
- Meta-World/Fetch re-eval used 30 episodes (CPU, 500-step Meta-World episodes);
  ManiSkill used 100. `once` may differ by ~0.02 from the training logs due to
  fewer episodes + fresh task randomization.

## Reproduce
```bash
python scripts/reeval_checkpoints.py --root outputs/full_matrix_1seed \
  --tasks PushCube-v1 PickCube-v1 LiftPegUpright-v1 PlaceSphere-v1 PokeCube-v1 StackCube-v1 \
  --eval-envs 32 --episodes 100 --out outputs/reeval_both_metrics.json
python scripts/reeval_checkpoints.py --root outputs/cpu_mw_fetch_1seed \
  --tasks button-press-v3 drawer-open-v3 pick-place-v3 peg-insert-side-v3 FetchPush-v4 FetchPickAndPlace-v4 \
  --eval-envs 8 --episodes 30 --skip-below 0.05 --out outputs/reeval_mw_fetch.json
```
