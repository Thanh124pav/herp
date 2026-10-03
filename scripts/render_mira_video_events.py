#!/usr/bin/env python3
"""Render and optionally upload exact MIRA allocation-event replays."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from herp.envs.dmc import DMCAdapter
from herp.envs.maniskill import ManiSkillAdapter
from herp.video_events import CandidateStore, VideoEvent, load_event, save_event


def _frame_uint8(frame, resolution: int) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        if frame.size and float(frame.max()) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if resolution > 0 and frame.shape[0] != resolution:
        from PIL import Image
        width = max(2, int(round(frame.shape[1] * resolution / frame.shape[0])))
        frame = np.asarray(Image.fromarray(frame).resize((width, resolution), Image.Resampling.LANCZOS))
    return frame


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    line_height = 16
    width = max((draw.textlength(line, font=font) for line in lines), default=0)
    draw.rounded_rectangle((8, 8, width + 24, 16 + line_height * len(lines)),
                           radius=4, fill=(0, 0, 0, 175))
    for index, line in enumerate(lines):
        draw.text((16, 12 + index * line_height), line, fill=(255, 255, 255, 255), font=font)
    return np.asarray(image)


def event_hud(event: VideoEvent, mode: str, rollout: int | None = None) -> list[str]:
    lines = [
        f"mode: {mode}",
        f"step: {event.training_step}",
        f"policy version: {event.policy_version}",
        f"source: R{event.region_id}",
    ]
    if mode == "ordinary training" and rollout is not None:
        lines.append(f"trajectory 0: {rollout} / {len(event.pre_fragments)}")
    if mode == "region restart":
        cluster = None if rollout is None else event.cluster_assignments[rollout - 1] + 1
        lines.extend([
            f"p_v: {event.p_ema:.3f}",
            f"sigma_v: {event.sigma_raw:.3f}",
            f"allocation prob: {event.allocation_prob:.3f}",
            f"allocated fragments: {event.allocated_fragments}",
            f"restart rollout: {rollout} / {len(event.rollout_fragments)}",
            f"distinct outcome class: {cluster} / {event.distinct_outcome_count}",
        ])
    return lines


def event_benchmark(event: VideoEvent) -> str:
    """Resolve old event packages that predate the explicit benchmark field."""
    benchmark = str(event.source.get("benchmark", "")).lower()
    if benchmark:
        return benchmark
    fragments = event.rollout_fragments or event.pre_fragments
    for fragment in fragments:
        snapshot = getattr(getattr(fragment, "restart_snapshot", None), "env_state", None)
        if snapshot is not None and snapshot.__class__.__module__.endswith(".dmc"):
            return "dmc"
    return "maniskill"


def render_adapter_frame(adapter, benchmark: str) -> np.ndarray:
    if benchmark == "dmc":
        # Direct access also works in an active trainer that imported DMCAdapter
        # before this renderer gained DMC support.
        return np.asarray(adapter.env.physics.render(height=480, width=640, camera_id=0))
    return np.asarray(adapter.render())


def replay_fragment(adapter, fragment, hud: list[str], resolution: int,
                    benchmark: str = "maniskill"):
    if fragment.restart_snapshot is None:
        return [], float("inf")
    restored = adapter.restore_state(torch.tensor([0]), [fragment.restart_snapshot.env_state])
    errors = [float((restored[0].cpu() - fragment.states[0]).abs().max())]
    frames = [draw_hud(_frame_uint8(render_adapter_frame(adapter, benchmark), resolution), hud)]
    low, high = adapter.action_low(), adapter.action_high()
    for index, action in enumerate(fragment.actions):
        applied = action[None].to(adapter.device).clamp(low, high)
        nxt, _reward, term, trunc, info = adapter.step(applied)
        actual_next = nxt
        if bool(term[0] | trunc[0]) and isinstance(info, dict) and "final_observation" in info:
            actual_next = info["final_observation"]
        expected = fragment.next_states[index]
        errors.append(float((actual_next[0].detach().cpu() - expected).abs().max()))
        frames.append(draw_hud(_frame_uint8(render_adapter_frame(adapter, benchmark), resolution), hud))
    return frames, max(errors)


def encode_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise RuntimeError("No replay frames were produced")
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p",
                            macro_block_size=2) as writer:
        for frame in frames:
            writer.append_data(frame)


def render_event(event: VideoEvent, output: Path, *, fps: int = 30,
                 pre_seconds: float = 10, post_seconds: float = 20,
                 max_seconds: float = 40, resolution: int = 720,
                 replay_tolerance: float = 1e-3) -> Path:
    source = event.source
    benchmark = event_benchmark(event)
    if benchmark == "dmc":
        os.environ.setdefault("MUJOCO_GL", "egl")
        adapter = DMCAdapter(source["env_id"], device="cpu").make(
            1, int(source.get("seed", 0)))
    elif benchmark == "maniskill":
        sim_backend = source.get("sim_backend", "physx_cuda")
        device = source.get("device", "cuda" if sim_backend == "physx_cuda" else "cpu")
        adapter = ManiSkillAdapter(
            source["env_id"],
            control_mode=source.get("control_mode", "pd_ee_delta_pose"),
            obs_mode=source.get("obs_mode", "state"),
            reward_mode=source.get("reward_mode", "dense"),
            sim_backend=sim_backend,
            render_backend=source.get("render_backend", "gpu"),
            device=device,
            enable_rendering=True,
            ignore_terminations=True,
        ).make(1, int(source.get("seed", 0)))
    else:
        raise ValueError(f"Unsupported video-event benchmark: {benchmark!r}")
    adapter.reset(seed=int(source.get("seed", 0)))

    errors = []
    pre_frames = []
    for index, fragment in enumerate(event.pre_fragments, 1):
        frames, error = replay_fragment(
            adapter, fragment, event_hud(event, "ordinary training", index), resolution, benchmark)
        if math.isfinite(error):
            errors.append(error)
            pre_frames.extend(frames)
    pre_limit = int(pre_seconds * fps)
    pre_frames = pre_frames[-pre_limit:] if pre_limit else []

    post_frames = []
    # Reserve the final diagnostic hold inside the hard duration cap.
    hard_cap = max(1, int(max_seconds * fps) - fps)
    desired_post = int(post_seconds * fps)
    for index, fragment in enumerate(event.rollout_fragments, 1):
        frames, error = replay_fragment(
            adapter, fragment, event_hud(event, "region restart", index), resolution, benchmark)
        errors.append(error)
        # Never cut a rollout in the middle. Stop before a new rollout only
        # after the desired post window, or if the next one exceeds hard max.
        if post_frames and len(pre_frames) + len(post_frames) + len(frames) > hard_cap:
            break
        post_frames.extend(frames)
        if len(post_frames) >= desired_post and index == len(event.rollout_fragments):
            break

    all_frames = pre_frames + post_frames
    max_error = max(errors, default=float("inf"))
    event.replay_max_state_error = max_error
    event.replay_exact = bool(math.isfinite(max_error) and max_error < replay_tolerance)
    if all_frames:
        summary = [
            f"region=R{event.region_id} | allocated={event.allocated_fragments}",
            f"distinct_outcomes={event.distinct_outcome_count} | diversity={event.pairwise_diversity:.3f}",
            f"replay max error={max_error:.2e} | exact={event.replay_exact}",
        ]
        held = draw_hud(all_frames[-1], summary)
        all_frames.extend([held] * fps)
    encode_mp4(all_frames, output, fps)
    adapter.close()
    return output


def _fresh_event_adapter(event: VideoEvent):
    """Create an isolated simulator instance for one independently editable clip."""
    source = event.source
    benchmark = event_benchmark(event)
    if benchmark == "dmc":
        os.environ.setdefault("MUJOCO_GL", "egl")
        adapter = DMCAdapter(source["env_id"], device="cpu").make(
            1, int(source.get("seed", 0)))
    elif benchmark == "maniskill":
        sim_backend = source.get("sim_backend", "physx_cuda")
        device = source.get("device", "cuda" if sim_backend == "physx_cuda" else "cpu")
        adapter = ManiSkillAdapter(
            source["env_id"],
            control_mode=source.get("control_mode", "pd_ee_delta_pose"),
            obs_mode=source.get("obs_mode", "state"),
            reward_mode=source.get("reward_mode", "dense"),
            sim_backend=sim_backend,
            render_backend=source.get("render_backend", "gpu"),
            device=device,
            enable_rendering=True,
            ignore_terminations=True,
        ).make(1, int(source.get("seed", 0)))
    else:
        raise ValueError(f"Unsupported video-event benchmark: {benchmark!r}")
    adapter.reset(seed=int(source.get("seed", 0)))
    return adapter, benchmark


def _replay_isolated_clip(event: VideoEvent, fragment, hud, resolution: int):
    adapter, benchmark = _fresh_event_adapter(event)
    try:
        return replay_fragment(adapter, fragment, hud, resolution, benchmark)
    finally:
        adapter.close()


def render_event_clips(event: VideoEvent, event_dir: Path, *, fps: int = 30,
                       resolution: int = 720, replay_tolerance: float = 1e-3) -> dict:
    """Render every origin trajectory and restart rollout as an independent MP4."""
    root = event_dir / f"event_{event.event_id}_clips"
    origin_dir = root / "origins"
    rollout_dir = root / "rollouts"
    entries = {"origins": [], "rollouts": []}
    errors = []

    for index, fragment in enumerate(event.pre_fragments, 1):
        frames, error = _replay_isolated_clip(
            event, fragment, event_hud(event, "ordinary training", index), resolution)
        path = origin_dir / f"trajectory_0_{index:03d}_fragment_{fragment.fragment_id:06d}.mp4"
        encode_mp4(frames, path, fps)
        errors.append(error)
        entries["origins"].append({
            "index": index, "fragment_id": int(fragment.fragment_id),
            "steps": len(fragment.actions), "duration_seconds": len(frames) / fps,
            "tags": ["ORIGIN", "TRAJECTORY_0", f"REGION_R{event.region_id}"],
            "replay_error": error, "path": str(path.relative_to(event_dir)),
        })

    for index, fragment in enumerate(event.rollout_fragments, 1):
        outcome_class = int(event.cluster_assignments[index - 1]) + 1
        frames, error = _replay_isolated_clip(
            event, fragment, event_hud(event, "region restart", index), resolution)
        path = rollout_dir / (
            f"rollout_{index:03d}_class_{outcome_class:03d}_"
            f"fragment_{fragment.fragment_id:06d}.mp4")
        encode_mp4(frames, path, fps)
        errors.append(error)
        entries["rollouts"].append({
            "index": index, "outcome_class": outcome_class,
            "fragment_id": int(fragment.fragment_id), "steps": len(fragment.actions),
            "duration_seconds": len(frames) / fps,
            "tags": ["ROLLOUT", "RESTART", f"REGION_R{event.region_id}",
                     f"OUTCOME_CLASS_{outcome_class}"],
            "replay_error": error, "path": str(path.relative_to(event_dir)),
        })

    max_error = max(errors, default=float("inf"))
    event.replay_max_state_error = max_error
    event.replay_exact = bool(math.isfinite(max_error) and max_error < replay_tolerance)
    manifest = {
        "event_id": event.event_id,
        "training_step": event.training_step,
        "region_id": event.region_id,
        "allocated_fragments": event.allocated_fragments,
        "origin_clip_count": len(entries["origins"]),
        "rollout_clip_count": len(entries["rollouts"]),
        "replay_max_state_error": max_error,
        "replay_exact": event.replay_exact,
        **entries,
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    event.source["clip_manifest"] = str(manifest_path.relative_to(event_dir))
    return manifest

def upload_event(run, wandb_module, event: VideoEvent, mp4: Path, prefix: str) -> None:
    key = f"{prefix}/{event.event_id}"
    caption = (
        f"MIRA step={event.training_step} region=R{event.region_id} "
        f"allocated={event.allocated_fragments} distinct={event.distinct_outcome_count} "
        f"pairwise={event.pairwise_diversity:.3f} score={event.diversity_score:.3f}"
    )
    payload = {
        key: wandb_module.Video(str(mp4), format="mp4", caption=caption),
        f"video_event/{event.event_id}/training_step": event.training_step,
        f"video_event/{event.event_id}/region_id": event.region_id,
        f"video_event/{event.event_id}/p": event.p_ema,
        f"video_event/{event.event_id}/sigma": event.sigma_raw,
        f"video_event/{event.event_id}/allocation_prob": event.allocation_prob,
        f"video_event/{event.event_id}/allocated_fragments": event.allocated_fragments,
        f"video_event/{event.event_id}/distinct_outcomes": event.distinct_outcome_count,
        f"video_event/{event.event_id}/pairwise_diversity": event.pairwise_diversity,
        f"video_event/{event.event_id}/ranking_score": event.diversity_score,
        f"video_event/{event.event_id}/success_count": event.success_count,
        f"video_event/{event.event_id}/replay_max_state_error": event.replay_max_state_error,
        "global_env_steps": event.training_step,
    }
    run.log(payload)


def upload_event_clips(run, wandb_module, event: VideoEvent,
                       event_dir: Path, manifest: dict, prefix: str) -> None:
    """Upload clips under separate W&B keys so editors receive independent assets."""
    metadata = {
        f"video_event/{event.event_id}/training_step": event.training_step,
        f"video_event/{event.event_id}/region_id": event.region_id,
        f"video_event/{event.event_id}/p": event.p_ema,
        f"video_event/{event.event_id}/sigma": event.sigma_raw,
        f"video_event/{event.event_id}/allocation_prob": event.allocation_prob,
        f"video_event/{event.event_id}/allocated_fragments": event.allocated_fragments,
        f"video_event/{event.event_id}/origin_clips": manifest["origin_clip_count"],
        f"video_event/{event.event_id}/rollout_clips": manifest["rollout_clip_count"],
        f"video_event/{event.event_id}/distinct_outcomes": event.distinct_outcome_count,
        f"video_event/{event.event_id}/pairwise_diversity": event.pairwise_diversity,
        f"video_event/{event.event_id}/ranking_score": event.diversity_score,
        f"video_event/{event.event_id}/replay_max_state_error": event.replay_max_state_error,
        "global_env_steps": event.training_step,
    }
    run.log(metadata)
    for clip in manifest["origins"]:
        path = event_dir / clip["path"]
        key = f"{prefix}/{event.event_id}/trajectory_0/{clip['index']:03d}"
        caption = (f"[ORIGIN][TRAJECTORY_0][R{event.region_id}] "
                   f"trajectory {clip['index']}/{manifest['origin_clip_count']} | "
                   f"fragment={clip['fragment_id']} | steps={clip['steps']}")
        run.log({key: wandb_module.Video(str(path), format="mp4", caption=caption),
                 "global_env_steps": event.training_step})
    for clip in manifest["rollouts"]:
        path = event_dir / clip["path"]
        key = f"{prefix}/{event.event_id}/rollouts/{clip['index']:03d}"
        caption = (f"[ROLLOUT][RESTART][R{event.region_id}] "
                   f"rollout {clip['index']}/{manifest['rollout_clip_count']} | "
                   f"class={clip['outcome_class']} | fragment={clip['fragment_id']} | "
                   f"steps={clip['steps']}")
        run.log({key: wandb_module.Video(str(path), format="mp4", caption=caption),
                 "global_env_steps": event.training_step})


def render_ranked_events(event_dir: str | Path, *, top_k: int, fps: int = 30,
                         pre_seconds: float = 10, post_seconds: float = 20,
                         max_seconds: float = 40, resolution: int = 720,
                         replay_tolerance: float = 1e-3, wandb_run=None,
                         wandb_module=None, wandb_prefix: str = "mira_events") -> list[VideoEvent]:
    event_dir = Path(event_dir)
    store = CandidateStore(event_dir, max_candidates=10**9)
    ranked = store.rank()
    rendered = []
    uploaded = 0
    for event in ranked:
        if wandb_run is None and len(rendered) >= top_k:
            break
        if wandb_run is not None and uploaded >= top_k:
            break
        manifest = render_event_clips(
            event, event_dir, fps=fps, resolution=resolution,
            replay_tolerance=replay_tolerance)
        save_event(event, event_dir)
        # A divergent candidate is kept locally and the next ranked candidate
        # is tried; it is never silently presented as an exact replay.
        if wandb_run is not None and event.replay_exact:
            upload_event_clips(
                wandb_run, wandb_module, event, event_dir, manifest, wandb_prefix)
            uploaded += 1
        rendered.append(event)
    store.events = {event.event_id: event for event in ranked}
    store.rank()
    return rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pre-seconds", type=float, default=10)
    parser.add_argument("--post-seconds", type=float, default=20)
    parser.add_argument("--max-seconds", type=float, default=40)
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--replay-tolerance", type=float, default=1e-3)
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", default="herp-v3")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-run-id", default="",
                        help="Resume an existing W&B run for video upload")
    parser.add_argument("--wandb-prefix", default="mira_events")
    args = parser.parse_args(argv)

    run = module = None
    if args.wandb_mode != "disabled":
        import wandb as module
        run = module.init(project=args.wandb_project, entity=args.wandb_entity or None,
                          group=args.wandb_group or None, job_type="video_diagnostics",
                          id=args.wandb_run_id or None,
                          resume="must" if args.wandb_run_id else None,
                          mode=args.wandb_mode)
    rendered = render_ranked_events(
        args.event_dir, top_k=args.top_k, fps=args.fps,
        pre_seconds=args.pre_seconds, post_seconds=args.post_seconds,
        max_seconds=args.max_seconds, resolution=args.resolution,
        replay_tolerance=args.replay_tolerance, wandb_run=run,
        wandb_module=module, wandb_prefix=args.wandb_prefix)
    if run is not None:
        run.finish()
    print(json.dumps({"rendered": len(rendered), "event_dir": args.event_dir}))


if __name__ == "__main__":
    main()
