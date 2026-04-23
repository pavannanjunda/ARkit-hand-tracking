#!/usr/bin/env python3
"""Generate derived outputs from the raw ARKit hand-tracking capture."""

from __future__ import annotations

import argparse
import json
import zipfile
from bisect import bisect_left
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation, Slerp

matplotlib.use("Agg")


HAND_JOINT_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
]

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

LEFT_COLOR = (70, 190, 255)
RIGHT_COLOR = (80, 255, 120)
VIDEO_PRESETS = {
    "compat": {"suffix": ".avi", "fourcc": "MJPG"},
    "mp4": {"suffix": ".mp4", "fourcc": "mp4v"},
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def to_matrix4(flat_values: list[float]) -> np.ndarray:
    return np.asarray(flat_values, dtype=np.float64).reshape(4, 4)


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm


def point_is_valid(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all() and np.linalg.norm(point) > 1e-8)


class PoseInterpolator:
    def __init__(self, depth_metadata: dict[str, Any]) -> None:
        frames = depth_metadata["frames"]
        self.times = np.asarray([frame["timestamp"] for frame in frames], dtype=np.float64)
        self.poses = np.asarray([to_matrix4(frame["camera_transform"]) for frame in frames], dtype=np.float64)
        self.translations = self.poses[:, :3, 3]
        self.rotations = Rotation.from_matrix(self.poses[:, :3, :3])
        self.slerp = Slerp(self.times, self.rotations)

    def pose_at(self, timestamp: float) -> np.ndarray:
        if timestamp <= self.times[0]:
            return self.poses[0].copy()
        if timestamp >= self.times[-1]:
            return self.poses[-1].copy()

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.slerp([timestamp]).as_matrix()[0]
        for axis in range(3):
            pose[axis, 3] = np.interp(timestamp, self.times, self.translations[:, axis])
        return pose

    def tracking_state_at(self, timestamp: float) -> str:
        if timestamp < self.times[0]:
            return "limited"

        idx = bisect_left(self.times.tolist(), timestamp)
        return "limited" if idx < 21 else "normal"


class DepthArchive:
    def __init__(self, zip_path: Path, depth_metadata: dict[str, Any]) -> None:
        self.zip_file = zipfile.ZipFile(zip_path)
        self.frames = depth_metadata["frames"]
        self.width = int(depth_metadata["width"])
        self.height = int(depth_metadata["height"])
        self._cache_index: int | None = None
        self._cache_depth: np.ndarray | None = None

    def depth_at_index(self, index: int) -> np.ndarray:
        if self._cache_index == index and self._cache_depth is not None:
            return self._cache_depth

        member = f"depth_maps/{self.frames[index]['file']}"
        raw = self.zip_file.read(member)
        depth = np.frombuffer(raw, dtype=np.float32).reshape(self.height, self.width)
        self._cache_index = index
        self._cache_depth = depth
        return depth

    def close(self) -> None:
        self.zip_file.close()


def infer_video_intrinsics(depth_intrinsics: dict[str, float], frame_width: int, frame_height: int) -> dict[str, float]:
    # The raw ARKit intrinsics are for the full sensor frame. This recording is a 1/3 downscale to 640x480.
    sensor_width = 1920.0
    sensor_height = 1440.0
    scale_x = frame_width / sensor_width
    scale_y = frame_height / sensor_height
    return {
        "fx": float(depth_intrinsics["fx"] * scale_x),
        "fy": float(depth_intrinsics["fy"] * scale_y),
        "cx": float(depth_intrinsics["cx"] * scale_x),
        "cy": float(depth_intrinsics["cy"] * scale_y),
        "width": frame_width,
        "height": frame_height,
    }


def nearest_depth_index(depth_times: np.ndarray, timestamp: float) -> int:
    idx = int(np.searchsorted(depth_times, timestamp))
    if idx <= 0:
        return 0
    if idx >= len(depth_times):
        return len(depth_times) - 1
    before = depth_times[idx - 1]
    after = depth_times[idx]
    return idx - 1 if abs(timestamp - before) <= abs(after - timestamp) else idx


def project_point(point: np.ndarray, intrinsics: dict[str, float]) -> np.ndarray | None:
    if not point_is_valid(point) or point[2] <= 1e-6:
        return None

    x = intrinsics["fx"] * (point[0] / point[2]) + intrinsics["cx"]
    y = intrinsics["fy"] * (point[1] / point[2]) + intrinsics["cy"]
    return np.asarray([x, y], dtype=np.float64)


def compute_wrist_pose(points: np.ndarray, chirality: str) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    wrist = points[0]
    if point_is_valid(wrist):
        pose[:3, 3] = wrist

    if not (point_is_valid(wrist) and point_is_valid(points[5]) and point_is_valid(points[17])):
        return pose

    index_axis = normalize(points[5] - wrist)
    little_axis = normalize(points[17] - wrist)
    if chirality.lower() == "left":
        palm_normal = normalize(np.cross(little_axis, index_axis))
    else:
        palm_normal = normalize(np.cross(index_axis, little_axis))
    palm_up = normalize(np.cross(palm_normal, index_axis))

    if not point_is_valid(index_axis) or not point_is_valid(palm_up) or not point_is_valid(palm_normal):
        return pose

    pose[:3, 0] = index_axis
    pose[:3, 1] = palm_up
    pose[:3, 2] = palm_normal
    return pose


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    world_points = np.zeros_like(points, dtype=np.float64)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    for i, point in enumerate(points):
        if point_is_valid(point):
            world_points[i] = rotation @ point + translation
    return world_points


def compute_reprojection_errors(points_3d: np.ndarray, points_2d: np.ndarray, intrinsics: dict[str, float]) -> tuple[list[float], float]:
    errors: list[float] = []
    valid_errors: list[float] = []
    for point_3d, point_2d in zip(points_3d, points_2d):
        projected = project_point(point_3d, intrinsics)
        if projected is None:
            errors.append(0.0)
            continue
        error = float(np.linalg.norm(projected - point_2d))
        errors.append(error)
        valid_errors.append(error)
    mean_error = float(np.mean(valid_errors)) if valid_errors else 0.0
    return errors, mean_error


def hand_distance_to_head(points_3d_camera_true: np.ndarray) -> float | None:
    """Return wrist-to-head distance as the Euclidean norm of the wrist in TRUE camera space.

    `points_3d_camera_true` must already be in camera space (i.e. the raw ARKit world-space
    landmarks transformed by the inverse camera pose).  This matches the legacy pipeline's
    'v2_corrected_transformation' convention, producing values in the 0.3–0.9 m range.
    """
    if point_is_valid(points_3d_camera_true[0]):
        return float(np.linalg.norm(points_3d_camera_true[0]))

    valid = [float(np.linalg.norm(p)) for p in points_3d_camera_true if point_is_valid(p)]
    return min(valid) if valid else None


def build_hand_record(
    raw_hand: dict[str, Any] | None,
    camera_pose_world: np.ndarray,
    intrinsics: dict[str, float],
) -> dict[str, Any] | None:
    if not raw_hand:
        return None

    points_2d = np.asarray(raw_hand["landmarks2D"], dtype=np.float64)
    # ARKit landmarks3D are in world space.  Transform to camera (head) space by applying
    # the inverse camera pose so that norm(wrist) == true wrist-to-head distance.
    points_3d_world_raw = np.asarray(raw_hand["landmarks3D"], dtype=np.float64)
    # Inverse of camera_pose_world: R^T*(p - t)
    R = camera_pose_world[:3, :3]
    t = camera_pose_world[:3, 3]
    points_3d_camera_true = np.zeros_like(points_3d_world_raw)
    for i, p in enumerate(points_3d_world_raw):
        if point_is_valid(p):
            points_3d_camera_true[i] = R.T @ (p - t)

    # World-space positions (used for 3D visualisation panel)
    points_3d_world = points_3d_world_raw

    depth_sources = []
    for joint_index, point in enumerate(points_3d_camera_true):
        if point_is_valid(point):
            depth_sources.append("lidar_wrist" if joint_index == 0 else "lidar")
        else:
            depth_sources.append("wrist_fallback" if joint_index == 0 else "fallback")

    wrist_pose_camera = compute_wrist_pose(points_3d_camera_true, raw_hand.get("chirality", "unknown"))
    wrist_pose_world = camera_pose_world @ wrist_pose_camera
    reprojection_error_per_joint, mean_reprojection_error = compute_reprojection_errors(
        points_3d_camera_true, points_2d, intrinsics
    )

    return {
        "hand_keypoints_2d": points_2d.tolist(),
        # Store true camera-space keypoints (matches legacy 'v2_corrected_transformation')
        "hand_keypoints_3d_camera": points_3d_camera_true.tolist(),
        "hand_keypoints_3d_world": points_3d_world.tolist(),
        "hand_keypoints_confidence": [float(value) for value in raw_hand.get("landmarkConfidences", [])],
        "depth_source_per_joint": depth_sources,
        "wrist_pose_world": wrist_pose_world.tolist(),
        "wrist_pose_camera": wrist_pose_camera.tolist(),
        "reprojection_error_per_joint": reprojection_error_per_joint,
        "mean_reprojection_error": mean_reprojection_error,
        # Distance = norm of wrist in camera/head space → matches legacy video labels
        "distance_to_head_m": hand_distance_to_head(points_3d_camera_true),
    }


def build_unified_dataset(raw_dir: Path, video_width: int, video_height: int) -> dict[str, Any]:
    hands_data = load_json(raw_dir / "hands.json")
    depth_metadata = load_json(raw_dir / "depth_metadata.json")
    video_timestamps = load_json(raw_dir / "video_timestamps.json")["timestamps"]
    pose_interpolator = PoseInterpolator(depth_metadata)
    intrinsics = infer_video_intrinsics(depth_metadata["camera_intrinsics"], video_width, video_height)

    total_frames = min(len(video_timestamps), len(hands_data["frames"]))
    frames: list[dict[str, Any]] = []
    for frame_id in range(total_frames):
        raw_frame = hands_data["frames"][frame_id]
        timestamp = float(video_timestamps[frame_id])
        camera_pose_world = pose_interpolator.pose_at(timestamp)

        frame_record: dict[str, Any] = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "camera_pose_world": camera_pose_world.tolist(),
            "camera_tracking_state": pose_interpolator.tracking_state_at(timestamp),
            "head_pose_world": camera_pose_world.tolist(),
            "head_reference": "camera_origin",
        }

        right_hand = build_hand_record(raw_frame.get("rightHand"), camera_pose_world, intrinsics)
        left_hand = build_hand_record(raw_frame.get("leftHand"), camera_pose_world, intrinsics)
        if right_hand is not None:
            frame_record["right_hand"] = right_hand
        if left_hand is not None:
            frame_record["left_hand"] = left_hand

        frames.append(frame_record)

    return {
        "sequence_name": "hands",
        "device": hands_data["metadata"].get("deviceModel", "iPhone"),
        "fps": hands_data["metadata"].get("frameRate", 30),
        "depth_source": "lidar",
        "depth_accuracy": depth_metadata.get("accuracy", "unknown"),
        "format_version": "output2_v1",
        "intrinsics": intrinsics,
        "hand_joint_names": HAND_JOINT_NAMES,
        "head_reference": "camera_origin",
        "frames": frames,
    }


def draw_hand(frame: np.ndarray, hand_record: dict[str, Any], color: tuple[int, int, int]) -> None:
    points = hand_record["hand_keypoints_2d"]
    for start, end in HAND_CONNECTIONS:
        pt1 = tuple(int(round(v)) for v in points[start])
        pt2 = tuple(int(round(v)) for v in points[end])
        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
    for point in points:
        center = tuple(int(round(v)) for v in point)
        cv2.circle(frame, center, 3, color, -1, cv2.LINE_AA)


def colorize_depth(depth_map: np.ndarray, min_depth: float, max_depth: float, output_size: tuple[int, int]) -> np.ndarray:
    clipped = np.clip(depth_map, min_depth, max_depth)
    invalid_mask = (~np.isfinite(depth_map)) | (depth_map <= 0)
    normalized = (clipped - min_depth) / max(max_depth - min_depth, 1e-6)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[invalid_mask] = 1.0
    normalized = 1.0 - normalized
    image = np.uint8(normalized * 255.0)
    image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    if invalid_mask.any():
        mask = cv2.resize(invalid_mask.astype(np.uint8) * 255, output_size, interpolation=cv2.INTER_NEAREST)
        colored[mask > 0] = (30, 30, 30)
    return colored


def annotate_overlay(frame: np.ndarray, frame_record: dict[str, Any], frame_id: int, depth_index: int) -> None:
    lines = [
        f"frame {frame_id:04d}",
        f"camera/head state: {frame_record['camera_tracking_state']}",
    ]
    if depth_index >= 0:
        lines.append(f"depth frame: {depth_index:04d}")

    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, 28 + idx * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_wrist_distance_label(
    frame: np.ndarray,
    hand_record: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
    dist_override: float | None = None,
) -> None:
    """Draw the wrist-to-head distance as a floating label at landmark 0 (wrist)."""
    dist = dist_override if dist_override is not None else hand_record.get("distance_to_head_m")
    if dist is None:
        return
    wrist_2d = hand_record["hand_keypoints_2d"][0]
    x = int(round(wrist_2d[0])) + 8
    y = int(round(wrist_2d[1])) - 8
    text = f"{label}: {dist:.2f}m"
    # Dark outline for readability over any background
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def compute_wrist_depth_distance(
    hand_record: dict[str, Any],
    depth_map: np.ndarray,
    depth_width: int,
    depth_height: int,
    video_width: int,
    video_height: int,
    intrinsics: dict[str, float],
) -> float | None:
    """Compute wrist-to-head distance using the LiDAR depth at the wrist 2D pixel.

    This matches the legacy 'v2_corrected_transformation' pipeline, which reprojects
    the wrist 2D landmark through the depth map into camera-space 3D, rather than
    using the raw ARKit landmarks3D directly.
    Convention: X right, Y up (image Y flipped), Z toward scene.
    """
    wrist_2d = hand_record["hand_keypoints_2d"][0]
    u, v = float(wrist_2d[0]), float(wrist_2d[1])
    # Scale from video resolution to depth map resolution
    dx = int(round(u * depth_width / video_width))
    dy = int(round(v * depth_height / video_height))
    dx = max(0, min(dx, depth_width - 1))
    dy = max(0, min(dy, depth_height - 1))
    depth = float(depth_map[dy, dx])
    if not np.isfinite(depth) or depth <= 0:
        return None
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    X = (u - cx) / fx * depth
    Y = -(v - cy) / fy * depth  # flip Y: image-down -> camera-up
    Z = -depth  # ARKit convention is -Z forward
    return float(np.sqrt(X * X + Y * Y + Z * Z))


def render_depth_overlay_video(
    raw_dir: Path,
    unified_data: dict[str, Any],
    output_path: Path,
    min_depth: float,
    max_depth: float,
    video_fourcc: str,
) -> None:
    depth_metadata = load_json(raw_dir / "depth_metadata.json")
    depth_times = np.asarray([frame["timestamp"] for frame in depth_metadata["frames"]], dtype=np.float64)
    depth_archive = DepthArchive(raw_dir / "depth_maps.zip", depth_metadata)
    depth_w = int(depth_metadata["width"])
    depth_h = int(depth_metadata["height"])
    intrinsics = unified_data["intrinsics"]

    capture = cv2.VideoCapture(str(raw_dir / "video.mov"))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or unified_data["fps"])
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*video_fourcc), fps, (width, height))

    try:
        for frame_record in unified_data["frames"]:
            ok, rgb_frame = capture.read()
            if not ok:
                break

            depth_index = nearest_depth_index(depth_times, frame_record["timestamp"])
            depth_map = depth_archive.depth_at_index(depth_index)
            depth_rgb = colorize_depth(depth_map, min_depth, max_depth, (width, height))
            overlay = cv2.addWeighted(rgb_frame, 0.42, depth_rgb, 0.58, 0.0)

            if "left_hand" in frame_record:
                draw_hand(overlay, frame_record["left_hand"], LEFT_COLOR)
                # Compute distance using the exact legacy depth map projection 
                left_dist = compute_wrist_depth_distance(
                    frame_record["left_hand"], depth_map, depth_w, depth_h, width, height, intrinsics
                )
                draw_wrist_distance_label(overlay, frame_record["left_hand"], LEFT_COLOR, "L", left_dist)
            if "right_hand" in frame_record:
                draw_hand(overlay, frame_record["right_hand"], RIGHT_COLOR)
                right_dist = compute_wrist_depth_distance(
                    frame_record["right_hand"], depth_map, depth_w, depth_h, width, height, intrinsics
                )
                draw_wrist_distance_label(overlay, frame_record["right_hand"], RIGHT_COLOR, "R", right_dist)

            annotate_overlay(overlay, frame_record, frame_record["frame_id"], depth_index)
            writer.write(overlay)
    finally:
        capture.release()
        writer.release()
        depth_archive.close()


def compute_depth_corrected_keypoints_world(
    hand_record: dict[str, Any],
    depth_map: np.ndarray,
    depth_width: int,
    depth_height: int,
    video_width: int,
    video_height: int,
    intrinsics: dict[str, float],
    camera_pose_world: np.ndarray,
) -> np.ndarray:
    """Reproject each joint's 2D landmark through the LiDAR depth map into world space.

    Matches the legacy 'v2_corrected_transformation' approach: uses LiDAR depth at each
    joint's 2D pixel rather than the raw ARKit landmarks3D (which are not in camera space).
    Returns an (N, 3) array of world-space positions; zeros where depth is invalid.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    rotation = camera_pose_world[:3, :3]
    translation = camera_pose_world[:3, 3]
    keypoints_2d = hand_record["hand_keypoints_2d"]
    n = len(keypoints_2d)
    world_points = np.zeros((n, 3), dtype=np.float64)
    for i, (u, v) in enumerate(keypoints_2d):
        dx = int(round(float(u) * depth_width / video_width))
        dy = int(round(float(v) * depth_height / video_height))
        dx = max(0, min(dx, depth_width - 1))
        dy = max(0, min(dy, depth_height - 1))
        depth = float(depth_map[dy, dx])
        if not np.isfinite(depth) or depth <= 0:
            continue
        X = (float(u) - cx) / fx * depth
        Y = -(float(v) - cy) / fy * depth  # flip Y: image-down -> camera-up
        Z = -depth  # ARKit convention is -Z forward
        cam_pt = np.array([X, Y, Z], dtype=np.float64)
        world_points[i] = rotation @ cam_pt + translation
    return world_points


def collect_world_bounds(frames: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    all_points: list[np.ndarray] = []
    for frame in frames:
        head_position = np.asarray(frame["head_pose_world"], dtype=np.float64)[:3, 3]
        all_points.append(head_position)
        for hand_key in ("left_hand", "right_hand"):
            hand = frame.get(hand_key)
            if not hand:
                continue
            points = np.asarray(hand["hand_keypoints_3d_world"], dtype=np.float64)
            valid = points[np.linalg.norm(points, axis=1) > 1e-8]
            if len(valid):
                all_points.append(valid)

    if not all_points:
        return np.asarray([-1.0, -1.0, -1.0]), np.asarray([1.0, 1.0, 1.0])

    stacked = np.vstack(all_points)
    return stacked.min(axis=0), stacked.max(axis=0)


def render_3d_panel(
    figure: Figure,
    frame_record: dict[str, Any],
    world_min: np.ndarray,
    world_max: np.ndarray,
    history: list[np.ndarray],
    corrected_world: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    figure.clear()
    axis = figure.add_subplot(111, projection="3d")
    axis.set_title("Head / Hands in World Frame")
    axis.set_xlabel("X")
    axis.set_ylabel("Z (Depth)")
    axis.set_zlabel("Y (Height)")

    center = np.asarray(frame_record["head_pose_world"], dtype=np.float64)[:3, 3]
    span = np.maximum(world_max - world_min, np.asarray([0.6, 0.6, 0.6]))
    radius = float(min(max(span.max() * 0.25, 0.35), 0.9))

    # Matplotlib's Z-axis is vertical. We map World Y to Matplotlib Z.
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[2] - radius, center[2] + radius)
    axis.set_zlim(center[1] - radius, center[1] + radius)
    axis.view_init(elev=18, azim=-65)

    if history:
        trail = np.vstack(history)
        axis.plot(trail[:, 0], trail[:, 2], trail[:, 1], color="black", linewidth=1.5, alpha=0.75)

    axis.scatter(center[0], center[2], center[1], color="crimson", s=70, label="head/camera")

    for hand_key, color, label in (("left_hand", "tab:orange", "left"), ("right_hand", "tab:green", "right")):
        hand = frame_record.get(hand_key)
        if not hand:
            continue
        # Use depth-corrected world positions if provided, else fall back to JSON
        if corrected_world and hand_key in corrected_world:
            points = corrected_world[hand_key]
        else:
            points = np.asarray(hand["hand_keypoints_3d_world"], dtype=np.float64)
        valid = np.linalg.norm(points, axis=1) > 1e-8
        if not np.any(valid):
            continue
        # Map World X->X, World Z->Y, World Y->Z for the plot
        axis.scatter(points[valid, 0], points[valid, 2], points[valid, 1], color=color, s=18, label=f"{label} hand")
        for start, end in HAND_CONNECTIONS:
            if valid[start] and valid[end]:
                axis.plot(
                    [points[start, 0], points[end, 0]],
                    [points[start, 2], points[end, 2]],
                    [points[start, 1], points[end, 1]],
                    color=color,
                    linewidth=1.5,
                )

    axis.legend(loc="upper left")
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    width, height = canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)


def render_visualization_video(raw_dir: Path, unified_data: dict[str, Any], output_path: Path, video_fourcc: str) -> None:
    depth_metadata = load_json(raw_dir / "depth_metadata.json")
    depth_times = np.asarray([frame["timestamp"] for frame in depth_metadata["frames"]], dtype=np.float64)
    depth_archive = DepthArchive(raw_dir / "depth_maps.zip", depth_metadata)
    depth_w = int(depth_metadata["width"])
    depth_h = int(depth_metadata["height"])
    intrinsics = unified_data["intrinsics"]

    capture = cv2.VideoCapture(str(raw_dir / "video.mov"))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or unified_data["fps"])
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*video_fourcc), fps, (width * 2, height))

    world_min, world_max = collect_world_bounds(unified_data["frames"])
    figure = Figure(figsize=(6.4, 4.8), dpi=100)
    history: list[np.ndarray] = []

    try:
        for frame_record in unified_data["frames"]:
            ok, rgb_frame = capture.read()
            if not ok:
                break

            depth_index = nearest_depth_index(depth_times, frame_record["timestamp"])
            depth_map = depth_archive.depth_at_index(depth_index)
            camera_pose = np.asarray(frame_record["camera_pose_world"], dtype=np.float64)

            # Compute depth-corrected world positions for all joints (fixes head-above-hands)
            corrected_world: dict[str, np.ndarray] = {}
            for hand_key in ("left_hand", "right_hand"):
                hand = frame_record.get(hand_key)
                if hand:
                    corrected_world[hand_key] = compute_depth_corrected_keypoints_world(
                        hand, depth_map, depth_w, depth_h, width, height, intrinsics, camera_pose
                    )

            # Left panel: RGB + 2D hands + wrist distance labels
            left_panel = rgb_frame.copy()
            if "left_hand" in frame_record:
                draw_hand(left_panel, frame_record["left_hand"], LEFT_COLOR)
                left_dist = compute_wrist_depth_distance(
                    frame_record["left_hand"], depth_map, depth_w, depth_h, width, height, intrinsics
                )
                draw_wrist_distance_label(left_panel, frame_record["left_hand"], LEFT_COLOR, "L", left_dist)
            if "right_hand" in frame_record:
                draw_hand(left_panel, frame_record["right_hand"], RIGHT_COLOR)
                right_dist = compute_wrist_depth_distance(
                    frame_record["right_hand"], depth_map, depth_w, depth_h, width, height, intrinsics
                )
                draw_wrist_distance_label(left_panel, frame_record["right_hand"], RIGHT_COLOR, "R", right_dist)
            annotate_overlay(left_panel, frame_record, frame_record["frame_id"], -1)

            # Right panel: 3D world-frame plot with corrected positions
            head_position = np.asarray(frame_record["head_pose_world"], dtype=np.float64)[:3, 3]
            history.append(head_position)
            history = history[-45:]
            right_panel = render_3d_panel(figure, frame_record, world_min, world_max, history, corrected_world)
            right_panel = cv2.resize(right_panel, (width, height), interpolation=cv2.INTER_AREA)

            combined = np.hstack([left_panel, right_panel])
            writer.write(combined)
    finally:
        capture.release()
        writer.release()
        depth_archive.close()


def detect_video_size(video_path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Unable to open video: {video_path}")
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("Raw Input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output2"))
    parser.add_argument("--min-depth", type=float, default=0.25)
    parser.add_argument("--max-depth", type=float, default=1.50)
    parser.add_argument("--video-preset", choices=sorted(VIDEO_PRESETS), default="compat")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_preset = VIDEO_PRESETS[args.video_preset]
    video_width, video_height = detect_video_size(args.raw_dir / "video.mov")
    unified_data = build_unified_dataset(args.raw_dir, video_width, video_height)

    unified_json_path = args.output_dir / "unified_hand_tracking_complete_v2.json"
    unified_json_path.write_text(json.dumps(unified_data, indent=2))

    if not args.skip_overlay:
        render_depth_overlay_video(
            args.raw_dir,
            unified_data,
            args.output_dir / f"depth_map_overlay{video_preset['suffix']}",
            args.min_depth,
            args.max_depth,
            video_preset["fourcc"],
        )

    if not args.skip_visualization:
        render_visualization_video(
            args.raw_dir,
            unified_data,
            args.output_dir / f"visualization_complete_v2{video_preset['suffix']}",
            video_preset["fourcc"],
        )


if __name__ == "__main__":
    main()
