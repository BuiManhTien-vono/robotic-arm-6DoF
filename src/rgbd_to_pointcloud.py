import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import scipy.io as scio


def load_vlm_bbox(result_path: str | Path) -> list[int]:
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    bbox = result.get("bbox") or result.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox in {result_path}: {bbox}")
    return [int(v) for v in bbox]


def load_camera_from_mat(meta_path: str | Path) -> dict[str, float]:
    meta = scio.loadmat(str(meta_path))
    intrinsic = meta["intrinsic_matrix"]
    factor_depth = float(np.asarray(meta["factor_depth"]).reshape(-1)[0])
    return {
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "depth_scale": factor_depth,
    }


def clamp_bbox(bbox: list[int], width: int, height: int, margin: int = 0) -> list[int]:
    x_min, y_min, x_max, y_max = bbox
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin

    x_min = max(0, min(width - 1, x_min))
    y_min = max(0, min(height - 1, y_min))
    x_max = max(0, min(width, x_max))
    y_max = max(0, min(height, y_max))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid bbox after clamping: {[x_min, y_min, x_max, y_max]}")
    return [x_min, y_min, x_max, y_max]


def load_rgb_depth(rgb_path: str | Path, depth_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(depth_path)
    if depth_raw.ndim != 2:
        raise ValueError(f"Depth image must be single-channel, got shape {depth_raw.shape}")
    return rgb, depth_raw


def crop_rgbd_by_bbox(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    bbox: list[int],
    margin: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], list[int]]:
    height, width = depth_raw.shape
    if rgb.shape[:2] != depth_raw.shape:
        raise ValueError(f"RGB/depth shape mismatch: {rgb.shape[:2]} vs {depth_raw.shape}")

    bbox = clamp_bbox(bbox, width, height, margin)
    x_min, y_min, x_max, y_max = bbox
    rgb_crop = rgb[y_min:y_max, x_min:x_max]
    depth_crop = depth_raw[y_min:y_max, x_min:x_max]
    return rgb_crop, depth_crop, (x_min, y_min), bbox


def create_point_cloud_from_crop(
    rgb_crop: np.ndarray,
    depth_crop_raw: np.ndarray,
    offset_xy: tuple[int, int],
    intrinsics: dict[str, float],
    *,
    max_depth: float = 2.0,
    min_depth: float = 0.05,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    depth = depth_crop_raw.astype(np.float32) / float(intrinsics["depth_scale"])
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    if not np.any(valid):
        raise ValueError("No valid depth points found inside bbox crop.")

    h, w = depth.shape
    x_offset, y_offset = offset_xy
    u_local, v_local = np.meshgrid(np.arange(w), np.arange(h))
    u = u_local + x_offset
    v = v_local + y_offset

    z = depth[valid]
    x = (u[valid].astype(np.float32) - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (v[valid].astype(np.float32) - intrinsics["cy"]) * z / intrinsics["fy"]
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = (rgb_crop[valid].astype(np.float32) / 255.0).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, points, colors


def save_crop_artifacts(
    output_dir: str | Path,
    rgb_crop: np.ndarray,
    depth_crop_raw: np.ndarray,
    pcd: o3d.geometry.PointCloud,
    points: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_crop_path = output_dir / "target_rgb_crop.png"
    depth_vis_path = output_dir / "target_depth_crop.png"
    ply_path = output_dir / "target_object.ply"
    points_path = output_dir / "target_points.npy"
    meta_path = output_dir / "target_pointcloud_meta.json"

    cv2.imwrite(str(rgb_crop_path), cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))

    depth_nonzero = depth_crop_raw[depth_crop_raw > 0]
    if depth_nonzero.size > 0:
        depth_min = float(depth_nonzero.min())
        depth_max = float(depth_nonzero.max())
        depth_vis = np.clip((depth_crop_raw.astype(np.float32) - depth_min) / max(depth_max - depth_min, 1.0), 0, 1)
        depth_vis = (depth_vis * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros(depth_crop_raw.shape, dtype=np.uint8)
    cv2.imwrite(str(depth_vis_path), depth_vis)

    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)
    np.save(points_path, points)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "rgb_crop": str(rgb_crop_path),
        "depth_crop": str(depth_vis_path),
        "point_cloud": str(ply_path),
        "points": str(points_path),
        "metadata": str(meta_path),
    }
