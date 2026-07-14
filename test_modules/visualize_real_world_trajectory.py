#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize real-world Falcon bridge replay samples.

The recorder writes one JSON sidecar and one NPZ per policy step. This viewer
loads those samples without ROS or Habitat, then displays the depth observation
with the goal and action information used by the policy.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    cv2 = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_ROOT = REPO_ROOT / "test_modules" / "test_results"
ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}
ACTION_COLORS = {
    0: (100, 100, 100),
    1: (70, 180, 90),
    2: (80, 150, 240),
    3: (240, 150, 80),
}


@dataclass(frozen=True)
class SampleRef:
    json_path: Optional[Path]
    npz_path: Path
    meta: Dict

    @property
    def name(self) -> str:
        if self.json_path is not None:
            return self.json_path.stem
        return self.npz_path.stem


@dataclass
class FrameData:
    depth: np.ndarray
    depth_kind: str
    max_depth_m: float
    goal: Optional[np.ndarray]
    action_id: Optional[int]
    probs: Optional[np.ndarray]
    meta: Dict
    npz_path: Path


def _ensure_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for the viewer. Install opencv-python or run "
            "with --summary-only to inspect sample metadata."
        )


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_npz_path(json_path: Path, meta: Dict) -> Path:
    candidates: List[Path] = []
    obs_npz = meta.get("obs_npz")
    if obs_npz:
        raw = Path(obs_npz)
        candidates.append(raw)
        candidates.append(json_path.parent / raw.name)

        norm = str(obs_npz).replace("\\", "/")
        marker = "ranger_nav/"
        if marker in norm:
            candidates.append(REPO_ROOT / norm.split(marker, 1)[1])

    candidates.append(json_path.with_suffix(".npz"))
    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError("Could not find NPZ for {}. Tried: {}".format(json_path, tried))


def _direct_replay_files(path: Path) -> List[Path]:
    json_files = sorted(path.glob("bridge_policy_replay_*.json"))
    if json_files:
        return json_files
    return sorted(path.glob("bridge_policy_replay_*.npz"))


def _find_latest_replay_dir(path: Path) -> Path:
    direct = _direct_replay_files(path)
    if direct:
        return path

    by_parent: Dict[Path, List[Path]] = defaultdict(list)
    for file_path in path.rglob("bridge_policy_replay_*.json"):
        by_parent[file_path.parent].append(file_path)

    if not by_parent:
        for file_path in path.rglob("bridge_policy_replay_*.npz"):
            by_parent[file_path.parent].append(file_path)

    if not by_parent:
        raise FileNotFoundError(
            "No bridge_policy_replay_*.json or *.npz found under {}".format(path)
        )

    return max(
        by_parent,
        key=lambda parent: max(p.stat().st_mtime for p in by_parent[parent]),
    )


def _resolve_replay_input(path: Path) -> Tuple[Path, List[Path]]:
    path = path.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if path.is_file():
        if path.suffix.lower() not in (".json", ".npz"):
            raise ValueError("Replay file must be .json or .npz: {}".format(path))
        return path.parent, [path]

    if not path.is_dir():
        raise FileNotFoundError("Replay path does not exist: {}".format(path))

    replay_dir = _find_latest_replay_dir(path)
    files = _direct_replay_files(replay_dir)
    if not files:
        raise FileNotFoundError("No replay files found in {}".format(replay_dir))
    return replay_dir, files


def _build_samples(files: Sequence[Path]) -> List[SampleRef]:
    samples: List[SampleRef] = []
    for file_path in files:
        if file_path.suffix.lower() == ".json":
            meta = _load_json(file_path)
            samples.append(
                SampleRef(
                    json_path=file_path,
                    npz_path=_resolve_npz_path(file_path, meta),
                    meta=meta,
                )
            )
        elif file_path.suffix.lower() == ".npz":
            samples.append(SampleRef(json_path=None, npz_path=file_path, meta={}))
    return samples


def _squeeze_depth(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _array_or_none(data, key: str) -> Optional[np.ndarray]:
    if key not in data:
        return None
    arr = np.asarray(data[key])
    if arr.size == 0:
        return None
    return arr


def load_frame(sample: SampleRef, depth_mode: str) -> FrameData:
    with np.load(sample.npz_path) as data:
        keys = set(data.files)
        if depth_mode == "meter":
            depth_key = "depth_meter" if "depth_meter" in keys else "depth"
        elif depth_mode == "normalized":
            depth_key = "depth"
        else:
            depth_key = "depth_meter" if "depth_meter" in keys else "depth"

        if depth_key not in keys:
            raise KeyError("{} does not contain a depth array".format(sample.npz_path))

        depth = _squeeze_depth(data[depth_key])
        depth_kind = "meter" if depth_key == "depth_meter" else "normalized"

        goal = _array_or_none(data, "goal")
        if goal is not None:
            goal = goal.astype(np.float32).reshape(-1)

        action = _array_or_none(data, "action")
        action_id = int(action.reshape(-1)[0]) if action is not None else None

        probs = _array_or_none(data, "probs")
        if probs is not None:
            probs = probs.astype(np.float32).reshape(-1)

    if goal is None and "polar_r" in sample.meta and "polar_theta" in sample.meta:
        goal = np.array(
            [sample.meta["polar_r"], sample.meta["polar_theta"]],
            dtype=np.float32,
        )
    if action_id is None and "action_id" in sample.meta:
        action_id = int(sample.meta["action_id"])
    if probs is None and sample.meta.get("action_probs") is not None:
        probs = np.asarray(sample.meta["action_probs"], dtype=np.float32)

    return FrameData(
        depth=depth,
        depth_kind=depth_kind,
        max_depth_m=float(sample.meta.get("max_depth_m", 10.0)),
        goal=goal,
        action_id=action_id,
        probs=probs,
        meta=sample.meta,
        npz_path=sample.npz_path,
    )


def _finite_stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    valid = arr[finite]
    if valid.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "p50": float(np.percentile(valid, 50)),
        "p95": float(np.percentile(valid, 95)),
    }


def _put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _text_width(text: str, scale: float, thickness: int = 1) -> int:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return int(size[0])


def _wrap_text(text: str, max_width: int, scale: float = 0.5) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        trial = current + " " + word
        if _text_width(trial, scale) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _put_wrapped(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    max_width: int,
    scale: float = 0.5,
    color: Tuple[int, int, int] = (215, 215, 215),
    line_gap: int = 19,
) -> int:
    for line in _wrap_text(text, max_width=max_width, scale=scale):
        _put_text(img, line, (x, y), scale=scale, color=color)
        y += line_gap
    return y


def _fmt_float(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        if not math.isfinite(float(value)):
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"
    return "{:.{}f}".format(float(value), precision)


def _render_depth(
    depth: np.ndarray,
    depth_kind: str,
    max_depth_m: float,
    display_size: int,
    clip_max: Optional[float],
) -> np.ndarray:
    hi = float(clip_max) if clip_max is not None else (max_depth_m if depth_kind == "meter" else 1.0)
    if hi <= 0:
        hi = 1.0

    arr = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(arr)
    normalized = np.clip(arr, 0.0, hi) / hi
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    vis_u8 = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)

    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    color = cv2.applyColorMap(vis_u8, colormap)
    color[~finite] = (0, 0, 0)
    return cv2.resize(color, (display_size, display_size), interpolation=cv2.INTER_NEAREST)


def _goal_xy(goal: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
    if goal is None or goal.size < 2:
        return None
    r = float(goal[0])
    theta = float(goal[1])
    return r * math.cos(theta), r * math.sin(theta)


def _collect_goal_trace(samples: Sequence[SampleRef]) -> List[Optional[Tuple[float, float]]]:
    trace: List[Optional[Tuple[float, float]]] = []
    for sample in samples:
        if "polar_r" in sample.meta and "polar_theta" in sample.meta:
            goal = np.array([sample.meta["polar_r"], sample.meta["polar_theta"]], dtype=np.float32)
            trace.append(_goal_xy(goal))
        else:
            trace.append(None)
    return trace


def _draw_goal_trace(
    panel: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    trace: Sequence[Optional[Tuple[float, float]]],
    index: int,
) -> None:
    valid_points = [p for p in trace if p is not None]
    if not valid_points:
        return

    cv2.rectangle(panel, (x, y), (x + w, y + h), (46, 49, 54), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (90, 94, 100), 1)
    _put_text(panel, "Goal trace (robot frame)", (x + 10, y + 23), scale=0.48, color=(220, 220, 220))

    plot_top = y + 34
    plot_bottom = y + h - 16
    center_x = x + w // 2
    robot_y = plot_bottom
    max_range = max(1.0, max(math.hypot(px, py) for px, py in valid_points))
    scale = min((w - 32) / (2.0 * max_range), (plot_bottom - plot_top - 8) / max_range)

    def to_px(point: Tuple[float, float]) -> Tuple[int, int]:
        forward, lateral = point
        return (
            int(round(center_x + lateral * scale)),
            int(round(robot_y - forward * scale)),
        )

    cv2.line(panel, (center_x, plot_top), (center_x, plot_bottom), (75, 78, 84), 1)
    cv2.line(panel, (x + 16, robot_y), (x + w - 16, robot_y), (75, 78, 84), 1)

    last_px: Optional[Tuple[int, int]] = None
    for point in trace[: index + 1]:
        if point is None:
            continue
        current_px = to_px(point)
        if last_px is not None:
            cv2.line(panel, last_px, current_px, (90, 180, 255), 2)
        last_px = current_px

    current = trace[index] if 0 <= index < len(trace) else None
    if current is not None:
        cv2.circle(panel, to_px(current), 5, (60, 220, 255), -1)

    robot = np.array(
        [
            [center_x, robot_y - 10],
            [center_x - 7, robot_y + 4],
            [center_x + 7, robot_y + 4],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(panel, robot, (230, 230, 230))


def _draw_probability_bars(
    panel: np.ndarray,
    x: int,
    y: int,
    w: int,
    probs: Optional[np.ndarray],
    action_id: Optional[int],
) -> int:
    _put_text(panel, "Action probabilities", (x, y), scale=0.52, color=(235, 235, 235))
    y += 22
    if probs is None:
        _put_text(panel, "N/A", (x, y), scale=0.5, color=(185, 185, 185))
        return y + 24

    max_label_width = 98
    bar_x = x + max_label_width
    bar_w = max(50, w - max_label_width - 52)
    for action_idx, prob in enumerate(probs.tolist()):
        label = ACTION_NAMES.get(action_idx, "ACTION_{}".format(action_idx))
        selected = action_id == action_idx
        color = ACTION_COLORS.get(action_idx, (160, 160, 160))
        row_color = (255, 255, 255) if selected else (205, 205, 205)
        _put_text(panel, label, (x, y + 14), scale=0.42, color=row_color)
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + 15), (70, 73, 79), -1)
        filled = int(round(bar_w * max(0.0, min(1.0, float(prob)))))
        cv2.rectangle(panel, (bar_x, y), (bar_x + filled, y + 15), color, -1)
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + 15), (105, 108, 114), 1)
        _put_text(panel, "{:.3f}".format(float(prob)), (bar_x + bar_w + 8, y + 14), scale=0.42, color=row_color)
        y += 22
    return y + 6


def render_frame(
    sample: SampleRef,
    frame: FrameData,
    index: int,
    total: int,
    args,
    goal_trace: Sequence[Optional[Tuple[float, float]]],
) -> np.ndarray:
    depth_vis = _render_depth(
        frame.depth,
        depth_kind=frame.depth_kind,
        max_depth_m=frame.max_depth_m,
        display_size=args.display_size,
        clip_max=args.clip_max,
    )

    panel = np.full((args.display_size, args.panel_width, 3), (35, 38, 43), dtype=np.uint8)
    x = 18
    y = 30
    content_w = args.panel_width - 2 * x

    _put_text(panel, "Falcon real-world replay", (x, y), scale=0.62, color=(245, 245, 245), thickness=2)
    y += 28
    _put_text(panel, "Frame {}/{}".format(index + 1, total), (x, y), scale=0.54, color=(220, 220, 220))
    y += 23
    y = _put_wrapped(panel, sample.name, x, y, content_w, scale=0.42, color=(170, 178, 188), line_gap=16)
    y += 8

    if frame.goal is not None and frame.goal.size >= 2:
        r = float(frame.goal[0])
        theta = float(frame.goal[1])
        _put_text(panel, "Goal observed by policy", (x, y), scale=0.52, color=(235, 235, 235))
        y += 22
        _put_text(panel, "r: {} m".format(_fmt_float(r)), (x, y), scale=0.5)
        y += 20
        _put_text(
            panel,
            "theta: {} rad / {} deg".format(_fmt_float(theta), _fmt_float(math.degrees(theta), 1)),
            (x, y),
            scale=0.5,
        )
        y += 20
        xy = _goal_xy(frame.goal)
        if xy is not None:
            _put_text(
                panel,
                "forward: {} m   lateral: {} m".format(_fmt_float(xy[0]), _fmt_float(xy[1])),
                (x, y),
                scale=0.48,
            )
            y += 22
    else:
        _put_text(panel, "Goal: N/A", (x, y), scale=0.5)
        y += 24

    action_name = ACTION_NAMES.get(frame.action_id, "ACTION_{}".format(frame.action_id))
    action_color = ACTION_COLORS.get(frame.action_id, (200, 200, 200))
    _put_text(panel, "Action: {} ({})".format(frame.action_id, action_name), (x, y), scale=0.55, color=action_color, thickness=2)
    y += 22
    lin = frame.meta.get("cmd_linear_x")
    ang = frame.meta.get("cmd_angular_z")
    _put_text(
        panel,
        "cmd vx={}  wz={}".format(_fmt_float(lin), _fmt_float(ang)),
        (x, y),
        scale=0.48,
        color=(210, 210, 210),
    )
    y += 28

    y = _draw_probability_bars(panel, x, y, content_w, frame.probs, frame.action_id)

    stats = _finite_stats(frame.depth)
    depth_unit = "m" if frame.depth_kind == "meter" else "norm"
    _put_text(panel, "Depth ({})".format(depth_unit), (x, y), scale=0.52, color=(235, 235, 235))
    y += 22
    _put_text(
        panel,
        "min {}  p50 {}  p95 {}".format(
            _fmt_float(stats["min"]),
            _fmt_float(stats["p50"]),
            _fmt_float(stats["p95"]),
        ),
        (x, y),
        scale=0.47,
    )
    y += 19
    _put_text(
        panel,
        "mean {}  max {}".format(_fmt_float(stats["mean"]), _fmt_float(stats["max"])),
        (x, y),
        scale=0.47,
    )
    y += 28

    depth_stamp = frame.meta.get("depth_msg_stamp")
    polar_stamp = frame.meta.get("polar_msg_stamp")
    if depth_stamp is not None and polar_stamp is not None:
        dt_ms = 1000.0 * (float(depth_stamp) - float(polar_stamp))
        _put_text(panel, "depth-polar dt: {} ms".format(_fmt_float(dt_ms, 1)), (x, y), scale=0.47)
        y += 22

    plot_h = 150
    plot_y = min(max(y + 5, args.display_size - plot_h - 54), args.display_size - plot_h - 36)
    _draw_goal_trace(panel, x, plot_y, content_w, plot_h, goal_trace, index)

    controls = "space play/pause   A/D or arrows step   Q quit"
    _put_text(panel, controls, (x, args.display_size - 16), scale=0.41, color=(160, 166, 176))

    canvas = np.hstack([depth_vis, panel])
    label = "{} clipped to {}".format(
        "depth_meter" if frame.depth_kind == "meter" else "depth_norm",
        _fmt_float(args.clip_max if args.clip_max is not None else (frame.max_depth_m if frame.depth_kind == "meter" else 1.0)),
    )
    cv2.rectangle(canvas, (8, 8), (8 + _text_width(label, 0.48) + 12, 34), (20, 20, 20), -1)
    _put_text(canvas, label, (14, 27), scale=0.48, color=(235, 235, 235))
    return canvas


def _iter_indices(total: int, start: int, stride: int, max_frames: Optional[int]) -> Iterable[int]:
    count = 0
    for idx in range(start, total, max(1, stride)):
        if max_frames is not None and count >= max_frames:
            break
        yield idx
        count += 1


def _print_summary(replay_dir: Path, samples: Sequence[SampleRef], args) -> None:
    first = load_frame(samples[0], args.depth_mode)
    last = load_frame(samples[-1], args.depth_mode)
    action_counts = Counter(
        int(s.meta["action_id"]) for s in samples if "action_id" in s.meta
    )

    print("Replay directory: {}".format(replay_dir))
    print("Sample count: {}".format(len(samples)))
    print("First sample: {}".format(samples[0].name))
    print("Last sample: {}".format(samples[-1].name))
    print("Depth shape: {} ({})".format(tuple(first.depth.shape), first.depth_kind))
    if first.goal is not None:
        print("First goal [r, theta]: [{:.6f}, {:.6f}]".format(float(first.goal[0]), float(first.goal[1])))
    if last.goal is not None:
        print("Last goal [r, theta]: [{:.6f}, {:.6f}]".format(float(last.goal[0]), float(last.goal[1])))
    if action_counts:
        parts = []
        for action_id in sorted(action_counts):
            parts.append(
                "{}:{}({})".format(
                    action_id,
                    ACTION_NAMES.get(action_id, "ACTION_{}".format(action_id)),
                    action_counts[action_id],
                )
            )
        print("Action counts: {}".format(" ".join(parts)))
    else:
        print("Action counts: N/A")


def _open_video_writer(path: Path, fps: float, frame_shape: Tuple[int, int, int]):
    _ensure_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame_shape[:2]
    suffix = path.suffix.lower()
    fourcc_name = "mp4v" if suffix in (".mp4", ".m4v") else "XVID"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer: {}".format(path))
    return writer


def run_viewer(replay_dir: Path, samples: Sequence[SampleRef], args) -> None:
    _ensure_cv2()
    if not samples:
        raise RuntimeError("No samples to display.")

    start = max(0, min(args.start, len(samples) - 1))
    goal_trace = _collect_goal_trace(samples)
    window_name = "Falcon real-world trajectory"
    delay_ms = max(1, int(round(1000.0 / max(0.1, args.fps))))
    writer = None

    if args.no_window:
        first_idx = next(iter(_iter_indices(len(samples), start, args.stride, args.max_frames)))
        first_frame_data = load_frame(samples[first_idx], args.depth_mode)
        first_canvas = render_frame(samples[first_idx], first_frame_data, first_idx, len(samples), args, goal_trace)
        if args.save_video:
            writer = _open_video_writer(Path(args.save_video), args.fps, first_canvas.shape)
            writer.write(first_canvas)
        elif args.export_dir:
            Path(args.export_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(args.export_dir) / "{:06d}.png".format(first_idx)), first_canvas)
        else:
            print("No window requested and no --save-video/--export-dir target set.")
            print("Use --summary-only for metadata or omit --no-window for interactive display.")
            return

        for idx in _iter_indices(len(samples), start + args.stride, args.stride, None):
            if args.max_frames is not None and idx >= start + args.stride * args.max_frames:
                break
            frame_data = load_frame(samples[idx], args.depth_mode)
            canvas = render_frame(samples[idx], frame_data, idx, len(samples), args, goal_trace)
            if writer is not None:
                writer.write(canvas)
            if args.export_dir:
                cv2.imwrite(str(Path(args.export_dir) / "{:06d}.png".format(idx)), canvas)
        if writer is not None:
            writer.release()
            print("Saved video: {}".format(args.save_video))
        if args.export_dir:
            print("Exported frames to: {}".format(args.export_dir))
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    index = start
    playing = args.play

    try:
        while True:
            frame_data = load_frame(samples[index], args.depth_mode)
            canvas = render_frame(samples[index], frame_data, index, len(samples), args, goal_trace)
            if writer is None and args.save_video:
                writer = _open_video_writer(Path(args.save_video), args.fps, canvas.shape)
            if writer is not None:
                writer.write(canvas)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKeyEx(delay_ms if playing else 0)

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                playing = not playing
            elif key in (ord("a"), ord("A"), 81, 2424832, 65361):
                index = max(0, index - args.stride)
                playing = False
            elif key in (ord("d"), ord("D"), 83, 2555904, 65363):
                index = min(len(samples) - 1, index + args.stride)
                playing = False
            elif key in (ord("r"), ord("R")):
                index = start
                playing = False
            elif playing:
                if index + args.stride < len(samples):
                    index += args.stride
                else:
                    playing = False
    finally:
        if writer is not None:
            writer.release()
            print("Saved video: {}".format(args.save_video))
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize recorded Falcon real-world trajectory replay samples."
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=DEFAULT_REPLAY_ROOT,
        help=(
            "Replay run directory, parent directory, JSON file, or NPZ file. "
            "If a parent directory is given, the newest replay run under it is used."
        ),
    )
    parser.add_argument(
        "--depth-mode",
        choices=["auto", "meter", "normalized"],
        default="auto",
        help="Use depth_meter when available, normalized depth, or auto-prefer meters.",
    )
    parser.add_argument("--clip-max", type=float, default=None, help="Depth visualization upper clip.")
    parser.add_argument("--display-size", type=int, default=640, help="Depth display size in pixels.")
    parser.add_argument("--panel-width", type=int, default=460, help="Right information panel width.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--stride", type=int, default=1, help="Frame step size.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit exported/no-window frames.")
    parser.add_argument("--play", action="store_true", help="Start in playback mode.")
    parser.add_argument("--no-window", action="store_true", help="Do not open a GUI window.")
    parser.add_argument("--summary-only", action="store_true", help="Print replay summary and exit.")
    parser.add_argument("--save-video", type=Path, default=None, help="Optional output video path.")
    parser.add_argument("--export-dir", type=Path, default=None, help="Optional directory for rendered PNG frames.")
    return parser.parse_args()


def main():
    args = parse_args()
    replay_dir, files = _resolve_replay_input(args.replay)
    samples = _build_samples(files)
    if not samples:
        raise RuntimeError("No replay samples found.")

    _print_summary(replay_dir, samples, args)
    if args.summary_only:
        return
    run_viewer(replay_dir, samples, args)


if __name__ == "__main__":
    main()
