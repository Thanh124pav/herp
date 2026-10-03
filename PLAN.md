# PLAN.md — MIRA Training-Event Video Capture for ICRA Supplementary Video

## 0. Objective

Add a **diagnostic video-capture pipeline** to the current MIRA training code so that training can automatically record and surface representative restart/allocation events.

The final goal is **not** to produce a polished submission video automatically. The goal is to produce a small set of scientifically faithful candidate clips on Weights & Biases (W&B), from which the author will manually select one for the final ICRA supplementary video.

Each candidate event should capture, as continuously as practical:

1. the ordinary training trajectory before the restart event,
2. the moment MIRA selects a previously visited region,
3. the restart from that region snapshot,
4. the rollout fragments collected according to MIRA's allocated budget,
5. the different outcomes produced by those restarted rollouts.

The capture window should target:

- **10 seconds before the event**
- **20 seconds after the event**

The system should then automatically rank candidate events by **outcome diversity** and upload the best ones to W&B.

---

## 1. Scientific constraints

The video is intended as supplementary scientific evidence.

Therefore:

- Do **not** add cinematic effects.
- Do **not** add transitions beyond hard cuts when unavoidable.
- Do **not** add music.
- Do **not** add voice-over.
- Do **not** add synthetic or staged behavior.
- Do **not** alter MIRA's policy, allocation rule, environment dynamics, or training data in order to obtain a nicer clip.
- Do **not** choose events based only on task success.
- Diagnostic text rendered by the code is allowed and encouraged.
- Every uploaded candidate must correspond to a real training event.
- Event ranking must use a pre-defined quantitative rule, not manual cherry-picking.

The renderer must be treated as a diagnostic observer and must not affect the training algorithm.

---

## 2. Current repository assumptions

The current implementation already contains most required information.

Relevant files:

- `scripts/train_v3.py`
  - main MIRA/HERP-v3 training loop
  - computes `p_ema`
  - computes `sigma_raw`
  - computes allocation probabilities
  - computes actual allocated fragments
  - writes `regions.jsonl`
  - supports W&B

- `src/herp/vector_acquisition.py`
  - vectorized allocation/restart logic
  - `VectorFragmentCollector.collect_round(...)`
  - restores archived snapshots for non-root region jobs
  - stores rollout fragments with:
    - `source_region_id`
    - states
    - actions
    - next states
    - rewards
    - chain region IDs
    - category
    - slot ID

- `src/herp/v3_observer.py`
- `src/herp/vector_acquisition.py::VectorPartitionObserver`
  - partition events
  - region assignments
  - boundaries

- `src/herp/archive.py`
  - region archive
  - representative snapshots
  - `p_ema`
  - `sigma_raw`
  - region metadata

- `src/herp/envs/maniskill.py`
  - supports ManiSkill `rgb_array`
  - supports snapshot save/restore

- `scripts/make_showcase_videos.py`
  - existing example of:
    - ManiSkill rendering
    - MP4 creation
    - `wandb.Video(...)`

Reuse existing infrastructure wherever possible.

---

# 3. Design principle

Do **not** continuously encode the entire training run into video.

Instead use a two-stage architecture:

```text
training
  |
  +-- maintain a lightweight rolling diagnostic buffer
  |
  +-- allocation/restart event occurs
  |
  +-- save event package:
  |      pre-event frames
  |      event metadata
  |      restart snapshots
  |      rollout information
  |      post-event frames
  |
  +-- score event diversity
  |
  +-- keep only top candidate events
  |
  +-- encode/upload selected candidates to W&B
```

The system may use either:

### Preferred mode

A dedicated **diagnostic render environment** with `num_envs=1` that mirrors selected training events.

This is preferred if rendering the main vectorized training environment would interfere with throughput or simulation behavior.

### Acceptable alternative

Render a selected training slot directly if this can be proven not to change training behavior or simulator state.

Default to the dedicated diagnostic render path unless direct training-slot rendering is clearly safe.

---

# 4. New CLI options

Add video-related CLI flags to `scripts/train_v3.py`.

Suggested arguments:

```bash
--capture-video-events
--video-fps 30
--video-pre-seconds 10
--video-post-seconds 20
--video-max-candidates 20
--video-upload-top-k 5
--video-min-region-prob 0.05
--video-min-allocated-fragments 2
--video-min-outcome-diversity 0.0
--video-resolution 720
--video-event-dir <output_dir>/video_events
--video-wandb-prefix mira_events
```

Behavior:

- video capture is **off by default**
- experiments without `--capture-video-events` must behave exactly as before

---

# 5. Event definition

A video event begins when all of the following are true:

1. MIRA is active, not in initial warmup.
2. At least one non-root region receives allocated rollout budget.
3. The chosen region has a valid archived snapshot.
4. The allocation round produces at least:
   - one actual restart from the region, and
   - at least `video_min_allocated_fragments` fragments from that region.

Prefer candidate regions with larger allocation probability, but do not restrict events only to the highest-probability region.

For every allocation round, identify one or more candidate regions:

```python
candidate_regions = [
    r for r in regions
    if r.region_id > 0
    and allocations.get(r.region_id, 0) >= min_allocated_fragments
    and probs[r.region_id] >= min_region_prob
    and len(r.snapshots) > 0
]
```

For each candidate event, store the values that existed **at allocation time**:

```text
training step
policy version
region id
p_raw
p_ema
sigma_raw
q_direct
q_pred
q_combined
allocation probability
allocated fragment count
root fraction
allocation entropy
snapshot id / snapshot metadata
```

Do not recompute these values later for display.

---

# 6. What one event clip must show

Each event clip should represent approximately:

```text
[-10 s] ---------------- EVENT ---------------- [+20 s]
```

The clip should make the following sequence visible:

### A. Pre-event training

Show the ordinary training trajectory leading into the allocation event.

Goal:

- make clear that the agent is interacting normally with the environment
- establish the state/task context before restart

### B. Allocation/restart moment

When MIRA chooses a non-root region:

- show the selected source region
- show that the environment is restored to an archived snapshot
- show the associated diagnostic values

### C. Allocated rollout fragments

Show the restarted rollouts generated from the selected region.

The event should make it possible to see:

```text
same / similar restart region
        |
        +--> continuation 1
        +--> continuation 2
        +--> continuation 3
        ...
```

The most valuable clips are those in which these continuations visibly diverge.

### D. Outcome summary

At the end of the event window, optionally display a static diagnostic line for ~1 second:

```text
region=R7 | allocated=8 | distinct_outcomes=4 | diversity=0.73
```

This is diagnostic information, not a cinematic effect.

---

# 7. Diagnostic overlay

Use a small fixed HUD in a corner.

Do not animate it.

Suggested fields:

```text
step: 421888
policy version: 37
source: R7
p_v: 0.812
sigma_v: 0.637
allocation prob: 0.274
allocated fragments: 8
restart rollout: 3 / 8
```

When showing the pre-event part:

```text
mode: ordinary training
current region: R3
```

When showing restart:

```text
mode: region restart
source region: R7
```

Do not show more information than fits cleanly.

Implementation may use OpenCV or PIL.

---

# 8. Rolling pre-event buffer

To capture 10 seconds before an event, maintain a rolling frame buffer.

For `fps = 30`:

```python
pre_frames = video_pre_seconds * video_fps
# default: 300 frames
```

Use:

```python
collections.deque(maxlen=pre_frames)
```

Each stored frame should already contain the diagnostic HUD corresponding to that frame.

Avoid storing raw simulator tensors if this creates excessive memory usage.

If 720p frames are too expensive to keep uncompressed:

- reduce diagnostic capture FPS internally, or
- keep JPEG-compressed frames in memory, or
- use a lightweight temporary rolling video segment

but the final candidate output should still satisfy the intended video quality.

The default implementation should prioritize robustness over perfect efficiency.

---

# 9. Post-event capture

After an allocation/restart event is triggered, continue recording for:

```text
20 seconds
```

or until all allocated restarted fragments associated with the selected event have completed, whichever gives a more semantically complete event.

Prefer capturing the full restart allocation if it exceeds 20 seconds slightly.

Do not truncate a rollout in the middle solely to hit exactly 20 seconds.

Hard maximum per candidate clip:

```text
40 seconds
```

unless explicitly overridden.

---

# 10. Mapping vectorized training to a readable video

The main training may use many parallel environments.

Do **not** attempt to tile hundreds of training slots.

For each candidate event:

1. choose one source region `v`,
2. choose the training fragments from that region,
3. replay or render them sequentially in a single diagnostic environment.

Recommended presentation:

```text
pre-event trajectory
restart from R_v
rollout 1
restart from R_v
rollout 2
restart from R_v
rollout 3
...
```

This remains faithful to the actual training event as long as:

- the policy checkpoint is the one used in that event,
- the same archived restart snapshot is used,
- the same actions are replayed when exact replay is possible,
- or the exact stored fragment states/actions are used to reconstruct the rollout.

Prefer exact replay using recorded actions.

Do not resample actions merely for video generation if the goal is to visualize the actual training event.

---

# 11. Event package format

Create a new module, for example:

```text
src/herp/video_events.py
```

Define an event package containing at least:

```python
@dataclass
class VideoEvent:
    event_id: str
    training_step: int
    policy_version: int

    region_id: int

    p_raw: float
    p_ema: float
    sigma_raw: float
    q_direct: float
    q_pred: float
    q_combined: float

    allocation_prob: float
    allocated_fragments: int

    pre_frames: list
    post_frames: list

    restart_snapshot: object
    rollout_fragments: list

    outcome_features: object | None
    diversity_score: float | None
    distinct_outcome_count: int | None
```

If storing full Python objects makes serialization fragile, split into:

```text
event_<id>.json
event_<id>.pt
event_<id>.mp4
```

Example:

```text
outputs/.../video_events/
    event_000421888_r7.json
    event_000421888_r7.pt
    event_000421888_r7.mp4
```

The JSON should contain only lightweight metadata.

---

# 12. Outcome representation

The automatic selector must rank events by **diversity of outcomes**, not simply by success count.

Use data already available in rollout fragments.

For each restarted fragment, compute a compact outcome feature.

Preferred feature:

```python
psi_i = concat(
    final_normalized_state,
    mean_normalized_state_over_last_k_steps,
    cumulative_reward,
    terminal_flag,
    success_once,
    success_at_end,
)
```

Suggested:

```text
k = min(5, fragment length)
```

Do not rely only on final reward because distinct physical outcomes can have similar reward.

If ManiSkill task-specific success flags are available, include them, but keep the state-based representation primary.

Normalize continuous components before distance computation.

---

# 13. Distinct-outcome clustering

For all restarted fragments from the same event:

```text
Ψ1, Ψ2, ..., ΨN
```

cluster them into outcome groups.

Use a simple, transparent method.

Recommended:

### Option A — agglomerative clustering

Distance:

```math
d(i,j) = ||Psi_i - Psi_j||_2
```

with a fixed threshold.

### Option B — radius clustering

Greedily create a new cluster when:

```math
min_j d(Psi_i, center_j) > delta_outcome
```

Prefer radius clustering because it is easy to explain and has no dependency on knowing the number of clusters.

Add a CLI parameter:

```bash
--video-outcome-radius
```

Choose a conservative default after inspecting state-feature scales.

The implementation must log the radius used.

---

# 14. Diversity score

Compute at least two metrics.

## 14.1 Distinct outcome count

```text
K = number of outcome clusters
```

## 14.2 Pairwise diversity

```math
D_pair =
2 / (N(N-1))
sum_{i<j} ||Psi_i - Psi_j||_2
```

Normalize if needed.

## 14.3 Recommended ranking score

Use:

```math
score =
log(1 + K)
*
D_pair
*
log(1 + N)
```

where:

- `K` = distinct cluster count
- `D_pair` = mean pairwise outcome distance
- `N` = number of restarted rollout fragments

This rewards:

- many distinct outcomes,
- strongly different outcomes,
- enough rollout samples.

Do not include success rate directly in the primary ranking score.

Success may be logged as secondary metadata.

---

# 15. Minimum validity criteria

An event is valid only if:

```text
allocated restarted fragments >= 3
distinct outcome clusters >= 2
pairwise diversity > 0
```

Recommended stronger filter for upload:

```text
allocated restarted fragments >= 4
distinct outcome clusters >= 3
```

Make these configurable.

---

# 16. Candidate retention

Do not save unlimited videos.

Keep only the best `video_max_candidates` metadata packages during training.

Recommended default:

```text
video_max_candidates = 20
```

After training:

1. compute/recompute diversity scores,
2. sort descending,
3. render/encode only the best candidates,
4. upload only top `video_upload_top_k`.

Default:

```text
video_upload_top_k = 5
```

This avoids filling W&B with hundreds of clips.

---

# 17. W&B logging

Use a dedicated namespace.

For each uploaded candidate:

```python
wandb_run.log({
    f"video_events/{event_id}": wandb.Video(
        str(mp4_path),
        format="mp4",
        caption=caption,
    ),
})
```

Also log metadata:

```text
video_event/<event_id>/training_step
video_event/<event_id>/region_id
video_event/<event_id>/p
video_event/<event_id>/sigma
video_event/<event_id>/allocation_prob
video_event/<event_id>/allocated_fragments
video_event/<event_id>/distinct_outcomes
video_event/<event_id>/pairwise_diversity
video_event/<event_id>/ranking_score
video_event/<event_id>/success_count
```

Prefer uploading candidates to the same training run if convenient.

If this makes the run too heavy, create a linked evaluation run with:

```text
group = original training group
job_type = "video_diagnostics"
```

Preserve:

```text
source training run id
source checkpoint
training step
policy version
seed
```

---

# 18. Add allocation probability to region logs

In `scripts/train_v3.py`, update the existing `regions.jsonl` output.

Currently each region logs `allocated_fragments`.

Also log:

```python
allocation_prob=float(probs[r.region_id])
```

if the probability exists for that region.

Example:

```python
row.update(
    step=collector.total_steps,
    policy_version=version,
    allocation_prob=float(probs[r.region_id]),
    allocated_fragments=allocations.get(r.region_id, 0),
)
```

Ensure indexing is correct even if region IDs and probability tensor indices ever diverge.

Prefer an explicit `{region_id: probability}` mapping.

---

# 19. Capture exact fragments associated with an event

The event must contain the exact rollout fragments produced by the allocation round.

From the `batch` returned by:

```python
collector.collect_round(...)
```

select:

```python
event_fragments = [
    f for f in batch
    if f.source_region_id == region_id
    and f.category == "REGION_ACQUISITION"
]
```

Store:

```text
states
actions
next_states
rewards
terminated
truncated
chain_region_id
slot_id
policy_version
source_region_id
```

These are the scientific ground truth for the event video.

---

# 20. Capture restart snapshot identity

Do not simply display "restart from R7".

Record exactly which archived snapshot was used for each allocated fragment.

Currently `VectorFragmentCollector.collect_round(...)` samples snapshots internally.

Modify the collector so that each produced `RolloutFragment` can optionally retain lightweight restart metadata:

```text
restart_region_id
restart_snapshot_index
restart_snapshot_timestep
restart_snapshot_elapsed_steps
```

Avoid duplicating large simulator state objects into every fragment if unnecessary.

If exact replay later requires simulator state, save the selected snapshot once per unique restart instance.

---

# 21. Modify vector acquisition carefully

Any modification to:

```text
src/herp/vector_acquisition.py
```

must satisfy:

- no change to sampled region IDs,
- no change to sampled snapshots,
- no change to RNG order,
- no additional calls to `torch.multinomial`,
- no additional environment steps,
- no change to PPO batch contents,
- no change to interaction accounting.

Diagnostic capture must observe existing decisions, not recreate them inside the training loop.

Add tests for this.

---

# 22. Video rendering

Create:

```text
scripts/render_mira_video_events.py
```

Responsibilities:

1. load event packages,
2. rank events if not already ranked,
3. select top K,
4. reconstruct/render each event,
5. draw minimal diagnostic HUD,
6. encode MP4,
7. optionally upload to W&B.

Use:

```text
render_mode="rgb_array"
num_envs=1
```

for the diagnostic environment.

Use the same:

```text
env_id
control_mode
reward_mode
obs_mode
```

as the source training run.

Use MP4/H.264 if available.

Suggested render defaults:

```text
720p
30 fps
```

Keep local candidate videos independent from final ICRA size compression.

The author will later choose one event and compress/edit it for submission.

---

# 23. Exact replay preference

For every candidate fragment:

1. restore the exact saved snapshot,
2. replay the exact stored actions,
3. capture rendered frames.

Do not call the policy to generate new actions during event rendering unless exact replay is impossible.

This ensures the candidate video shows the actual training experience.

Verify replay consistency using state error:

```math
max_t ||s_t^{replay} - s_t^{logged}||_\infty
```

Log this value.

Target:

```text
< 1e-3
```

If replay diverges beyond tolerance:

- flag the candidate,
- do not silently claim exact replay,
- prefer another event for upload.

---

# 24. Pre-event reconstruction

The 10-second pre-event segment should preferably come from the actual training trajectory associated with the relevant slot or source trajectory.

If exact pre-event reconstruction is difficult in the first implementation:

### Phase 1 acceptable implementation

Maintain a rolling rendered frame buffer from one designated diagnostic training slot.

### Phase 2 preferred implementation

Store enough state/action history to replay the exact preceding trajectory for the selected event.

Do not fabricate a pre-event trajectory from an unrelated episode.

---

# 25. Event selection should happen after training

The final top-event selection should happen after the run completes.

Workflow:

```text
training
  -> collect candidate event packages
  -> finish training
  -> score all candidates
  -> rank candidates
  -> render top K
  -> upload top K to W&B
```

This is preferred over immediately uploading every event.

If the process crashes before training completes, retain saved event packages so rendering can be resumed separately.

---

# 26. Summary artifact

After ranking, write:

```text
video_events/ranking.json
```

Example:

```json
[
  {
    "rank": 1,
    "event_id": "step421888_r7",
    "step": 421888,
    "region_id": 7,
    "p": 0.812,
    "sigma": 0.637,
    "allocation_prob": 0.274,
    "allocated_fragments": 8,
    "distinct_outcomes": 4,
    "pairwise_diversity": 0.73,
    "ranking_score": 2.91,
    "mp4": "event_step421888_r7.mp4"
  }
]
```

This file should make manual review easy.

---

# 27. Tests

Add unit/integration tests.

## 27.1 No behavioral change

Run the same short seeded training job with capture:

```text
OFF
ON
```

Compare:

```text
region allocations
environment-step counts
policy parameter checksum after a short deterministic test
```

They should be identical, except for wall-clock time and diagnostic outputs.

## 27.2 Event extraction

Synthetic fragments with known outcomes should produce expected:

```text
distinct outcome count
pairwise diversity
ranking order
```

## 27.3 Replay consistency

For a short ManiSkill event:

```text
restore snapshot
replay actions
compare next states
```

Verify replay error is below tolerance.

## 27.4 W&B smoke test

Upload one tiny generated clip and verify:

```text
wandb.Video
metadata
run linkage
```

---

# 28. Performance considerations

Training speed remains more important than video generation.

Requirements:

- do not render every vector slot,
- do not encode MP4 during every training step,
- do not block GPU training on ffmpeg,
- do not retain unlimited raw frames,
- avoid copying all vector observations to CPU solely for video capture.

If rendering causes noticeable training slowdown:

1. reduce diagnostic capture FPS,
2. use asynchronous local encoding only if safely implemented,
3. otherwise save event replay data and render after training.

Prefer post-training replay if there is any doubt.

---

# 29. Recommended first experiment

Use one ManiSkill task where:

- MIRA reaches meaningful intermediate states,
- restarts are frequent enough,
- the environment produces visibly different continuation outcomes.

Start with:

```text
PickCube-v1
```

unless recent experiments show another task has substantially clearer restart diversity.

Suggested command shape:

```bash
python scripts/train_v3.py \
  --benchmark maniskill \
  --env-id PickCube-v1 \
  --method herp \
  --num-envs <existing_setting> \
  --total-timesteps <normal_training_budget> \
  --wandb-mode online \
  --capture-video-events \
  --video-pre-seconds 10 \
  --video-post-seconds 20 \
  --video-max-candidates 20 \
  --video-upload-top-k 5 \
  --output-dir outputs/mira_video_pickcube_s0
```

Use the project's current MIRA method name if it has been renamed from `herp` in the CLI.

Do not silently change training hyperparameters merely to obtain better-looking footage.

---

# 30. Deliverables

Codex should finish with:

```text
src/herp/video_events.py
scripts/render_mira_video_events.py
```

plus minimal modifications to:

```text
scripts/train_v3.py
src/herp/vector_acquisition.py
```

and tests.

Required outputs from one completed training run:

```text
<output_dir>/
    video_events/
        event_*.json
        event_*.pt
        event_*.mp4
        ranking.json
```

W&B should contain the top candidate videos.

---

# 31. Acceptance criteria

The task is complete when all of the following are true:

1. Training with video capture disabled behaves exactly as before.
2. Training with capture enabled preserves the same algorithmic decisions under the same seed.
3. Candidate events correspond to real MIRA allocation/restart events.
4. Each event contains:
   - pre-event context,
   - restart,
   - allocated rollout fragments,
   - resulting outcomes.
5. Diagnostic overlay shows real logged values.
6. Outcome diversity is computed automatically.
7. Candidates are ranked automatically.
8. Only top-K candidate videos are uploaded to W&B.
9. Each uploaded video can be traced back to:
   - training run,
   - seed,
   - training step,
   - policy version,
   - source region,
   - exact restart fragments.
10. The author can inspect the W&B candidates and manually select one clip for the final ICRA submission.

---

# 32. Priority order

Implement in this order:

### P0 — Must have

1. event metadata capture
2. exact restart-fragment association
3. outcome feature computation
4. diversity scoring
5. candidate ranking
6. post-training rendering
7. W&B upload

### P1 — Strongly preferred

8. 10-second pre-event context
9. replay-consistency verification
10. diagnostic HUD

### P2 — Optional

11. richer partition diagnostic information
12. automated final video concatenation

Do not spend time on P2 until P0 and P1 are verified.

---

# 33. Final instruction to Codex

Before modifying the code:

1. read the current `scripts/train_v3.py`,
2. read `src/herp/vector_acquisition.py`,
3. read `src/herp/archive.py`,
4. read `src/herp/envs/maniskill.py`,
5. read the existing W&B/video example in `scripts/make_showcase_videos.py`.

Preserve current algorithmic behavior.

When uncertain between:

```text
better-looking video
```

and:

```text
more faithful scientific recording
```

always choose the latter.
