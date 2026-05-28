import argparse
import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm_localizer import localize_target_object
from src.coordinate_transform import transform_grasp_camera_to_world


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "vlm_panda_sim"
PANDA_SIM_SCRIPT = PROJECT_ROOT / "scripts" / "06_run_panda_pick_place_sim.py"
CAMERA_FOV_DEG = 55.0
CAMERA_NEAR = 0.02
CAMERA_FAR = 3.0


def first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_GRASPNET_REPO = first_existing_path(
    [
        PROJECT_ROOT / "graspnet-baseline",
        PROJECT_ROOT.parent / "graspnet-baseline",
        PROJECT_ROOT.parent.parent / "graspnet-baseline",
        Path.cwd() / "graspnet-baseline",
    ]
)
DEFAULT_CHECKPOINT = first_existing_path(
    [
        DEFAULT_GRASPNET_REPO / "checkpoints" / "checkpoint-rs.tar",
        PROJECT_ROOT / "graspnet-baseline" / "checkpoints" / "checkpoint-rs.tar",
        PROJECT_ROOT.parent / "graspnet-baseline" / "checkpoints" / "checkpoint-rs.tar",
        PROJECT_ROOT.parent.parent / "graspnet-baseline" / "checkpoints" / "checkpoint-rs.tar",
        Path.cwd() / "graspnet-baseline" / "checkpoints" / "checkpoint-rs.tar",
    ]
)


def load_panda_module():
    spec = importlib.util.spec_from_file_location("panda_pick_place_sim", PANDA_SIM_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {PANDA_SIM_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_camera(
    *,
    width: int = 960,
    height: int = 720,
    target=(0.55, -0.10, 0.18),
    distance: float = 1.15,
    yaw: float = 45.0,
    pitch: float = -45.0,
    roll: float = 0.0,
):
    data = render_camera_data(
        width=width,
        height=height,
        target=target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
    )
    return data["rgb"], data["segmentation"]


def camera_intrinsics(width: int, height: int, fov_deg: float = CAMERA_FOV_DEG) -> dict[str, float]:
    focal = height / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    return {
        "fx": float(focal),
        "fy": float(focal),
        "cx": float(width / 2.0),
        "cy": float(height / 2.0),
        "depth_scale": 1.0,
    }


def world_from_view_matrix_cv(view_matrix: list[float]) -> np.ndarray:
    view = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4, order="F")
    t_world_camera_gl = np.linalg.inv(view)
    t_camera_gl_camera_cv = np.eye(4, dtype=np.float64)
    t_camera_gl_camera_cv[:3, :3] = np.diag([1.0, -1.0, -1.0])
    return (t_world_camera_gl @ t_camera_gl_camera_cv).astype(np.float32)


def depth_buffer_to_meters(
    depth_buffer: np.ndarray,
    *,
    near: float = CAMERA_NEAR,
    far: float = CAMERA_FAR,
) -> np.ndarray:
    depth_buffer = depth_buffer.astype(np.float32)
    return (far * near / (far - (far - near) * depth_buffer)).astype(np.float32)


def render_camera_data(
    *,
    width: int = 960,
    height: int = 720,
    target=(0.55, -0.10, 0.18),
    distance: float = 1.15,
    yaw: float = 45.0,
    pitch: float = -45.0,
    roll: float = 0.0,
):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        upAxisIndex=2,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=CAMERA_FOV_DEG,
        aspect=width / height,
        nearVal=CAMERA_NEAR,
        farVal=CAMERA_FAR,
    )
    _, _, rgba, depth_buffer, seg = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    )
    rgb = np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)
    depth_m = depth_buffer_to_meters(np.reshape(depth_buffer, (height, width)))
    seg = np.reshape(seg, (height, width)).astype(np.int64)
    return {
        "rgb": rgb,
        "depth_m": depth_m,
        "segmentation": seg,
        "intrinsics": camera_intrinsics(width, height),
        "T_world_camera_cv": world_from_view_matrix_cv(view_matrix),
        "camera": {
            "width": int(width),
            "height": int(height),
            "target": list(map(float, target)),
            "distance": float(distance),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "fov_deg": float(CAMERA_FOV_DEG),
            "near": float(CAMERA_NEAR),
            "far": float(CAMERA_FAR),
        },
    }


def decode_object_uid(segmentation_value: np.ndarray) -> np.ndarray:
    return segmentation_value & ((1 << 24) - 1)


def bbox_from_object(segmentation: np.ndarray, object_id: int) -> list[int]:
    object_uid = decode_object_uid(segmentation)
    ys, xs = np.where(object_uid == object_id)
    if xs.size == 0:
        raise ValueError(f"Object {object_id} is not visible in the rendered segmentation.")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def bbox_center(bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def has_text_term(text: str, term: str) -> bool:
    term = normalize_text(term).strip()
    if not term:
        return False
    if " " in term:
        return term in text
    return re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", text) is not None


COLOR_ALIASES = {
    "red": ("red", "do"),
    "green": ("green", "xanh la", "xanh luc"),
    "blue": ("blue", "xanh duong", "xanh lam"),
    "yellow": ("yellow", "vang"),
    "magenta": ("magenta", "hong", "tim"),
    "cyan": ("cyan", "xanh ngoc"),
    "brown": ("brown", "nau"),
    "white": ("white", "trang"),
}


def semantic_score(metadata: dict | None, text_hint: str | None) -> int:
    if not metadata or not text_hint:
        return 0

    text = normalize_text(text_hint)
    object_type = normalize_text(str(metadata.get("type", "")))
    shape = normalize_text(str(metadata.get("shape", object_type)))
    color_name = normalize_text(str(metadata.get("color_name", "")))
    score = 0

    if object_type and has_text_term(text, object_type):
        score += 10
    if shape and has_text_term(text, shape):
        score += 8

    round_terms = ("round", "circular", "circle", "sphere", "ball", "tron", "hinh tron", "qua cau", "hinh cau")
    cylinder_terms = ("cylinder", "cylindrical", "bottle", "chai", "tru", "hinh tru")
    box_terms = ("box", "cube", "square", "book", "vuong", "hinh vuong", "hinh hop", "khoi hop", "chu nhat")

    if any(has_text_term(text, term) for term in round_terms):
        if shape == "sphere":
            score += 12
        elif shape == "cylinder":
            score += 4
        elif shape == "box":
            score -= 5
    if any(has_text_term(text, term) for term in cylinder_terms):
        score += 10 if shape == "cylinder" else -2
    if any(has_text_term(text, term) for term in box_terms):
        score += 10 if shape == "box" else -2

    for canonical_color, aliases in COLOR_ALIASES.items():
        if any(has_text_term(text, alias) for alias in aliases):
            score += 6 if color_name == canonical_color else -1

    return score


def visible_object_bboxes(segmentation: np.ndarray, candidate_ids: list[int]) -> dict[int, list[int]]:
    visible_bboxes = {}
    for object_id in candidate_ids:
        try:
            visible_bboxes[int(object_id)] = bbox_from_object(segmentation, object_id)
        except ValueError:
            continue
    if not visible_bboxes:
        raise RuntimeError("No movable objects are visible in the rendered segmentation.")
    return visible_bboxes


def choose_semantic_object(
    *,
    visible_bboxes: dict[int, list[int]],
    bbox: list[int],
    object_metadata: dict[int, dict] | None,
    text_hint: str | None,
) -> tuple[int, dict] | None:
    if not object_metadata or not text_hint:
        return None

    scores = {
        object_id: semantic_score(object_metadata.get(object_id), text_hint)
        for object_id in visible_bboxes
    }
    best_score = max(scores.values(), default=0)
    if best_score <= 0:
        return None

    target_center = bbox_center(bbox)
    distances = {
        object_id: float(np.linalg.norm(target_center - bbox_center(object_bbox)))
        for object_id, object_bbox in visible_bboxes.items()
    }
    ious = {object_id: bbox_iou(bbox, object_bbox) for object_id, object_bbox in visible_bboxes.items()}
    best_candidates = [object_id for object_id, score in scores.items() if score == best_score]
    selected_id = min(best_candidates, key=lambda object_id: (-ious[object_id], distances[object_id]))

    return int(selected_id), {
        "semantic_scores": {int(object_id): int(score) for object_id, score in scores.items()},
        "selected_metadata": object_metadata.get(int(selected_id), {}),
        "fallback_distance_px": float(distances[selected_id]),
        "fallback_iou": float(ious[selected_id]),
    }


def fallback_object_from_bbox(
    segmentation: np.ndarray,
    bbox: list[int],
    candidate_ids: list[int],
    *,
    object_metadata: dict[int, dict] | None = None,
    text_hint: str | None = None,
) -> dict:
    visible_bboxes = visible_object_bboxes(segmentation, candidate_ids)

    semantic_choice = choose_semantic_object(
        visible_bboxes=visible_bboxes,
        bbox=bbox,
        object_metadata=object_metadata,
        text_hint=text_hint,
    )
    if semantic_choice is not None:
        selected_id, semantic_details = semantic_choice
        return {
            "object_id": int(selected_id),
            "bbox_used": bbox,
            "pixel_counts": {},
            "selected_pixel_ratio": 0.0,
            "fallback_used": True,
            "fallback_reason": "bbox did not contain object pixels; selected by VLM/user semantic hint",
            "selected_object_bbox": visible_bboxes[selected_id],
            **semantic_details,
        }

    ious = {object_id: bbox_iou(bbox, object_bbox) for object_id, object_bbox in visible_bboxes.items()}
    best_iou_id = max(ious, key=ious.get)
    if ious[best_iou_id] > 0:
        return {
            "object_id": int(best_iou_id),
            "bbox_used": bbox,
            "pixel_counts": {},
            "selected_pixel_ratio": 0.0,
            "fallback_used": True,
            "fallback_reason": "no object pixels inside VLM bbox; selected highest-IoU visible object bbox",
            "fallback_iou": float(ious[best_iou_id]),
            "selected_object_bbox": visible_bboxes[best_iou_id],
        }

    target_center = bbox_center(bbox)
    distances = {
        object_id: float(np.linalg.norm(target_center - bbox_center(object_bbox)))
        for object_id, object_bbox in visible_bboxes.items()
    }
    closest_id = min(distances, key=distances.get)
    return {
        "object_id": int(closest_id),
        "bbox_used": bbox,
        "pixel_counts": {},
        "selected_pixel_ratio": 0.0,
        "fallback_used": True,
        "fallback_reason": "no object pixels or IoU; selected nearest visible object bbox center",
        "fallback_distance_px": float(distances[closest_id]),
        "selected_object_bbox": visible_bboxes[closest_id],
    }


def select_object_from_bbox(
    segmentation: np.ndarray,
    bbox: list[int],
    candidate_ids: list[int],
    *,
    object_metadata: dict[int, dict] | None = None,
    text_hint: str | None = None,
) -> dict:
    height, width = segmentation.shape
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, min(width - 1, int(x_min)))
    y_min = max(0, min(height - 1, int(y_min)))
    x_max = max(0, min(width, int(x_max)))
    y_max = max(0, min(height, int(y_max)))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid bbox: {bbox}")

    object_uid = decode_object_uid(segmentation[y_min:y_max, x_min:x_max])
    counts = {}
    for object_id in candidate_ids:
        count = int(np.count_nonzero(object_uid == object_id))
        if count > 0:
            counts[int(object_id)] = count
    if not counts:
        return fallback_object_from_bbox(
            segmentation,
            [x_min, y_min, x_max, y_max],
            candidate_ids,
            object_metadata=object_metadata,
            text_hint=text_hint,
        )

    selected_id = max(counts, key=counts.get)
    visible_bboxes = visible_object_bboxes(segmentation, candidate_ids)
    semantic_choice = choose_semantic_object(
        visible_bboxes=visible_bboxes,
        bbox=[x_min, y_min, x_max, y_max],
        object_metadata=object_metadata,
        text_hint=text_hint,
    )
    if semantic_choice is not None:
        semantic_id, semantic_details = semantic_choice
        selected_score = semantic_details["semantic_scores"].get(int(selected_id), 0)
        semantic_score_value = semantic_details["semantic_scores"].get(int(semantic_id), 0)
        if semantic_id != selected_id and semantic_score_value > selected_score:
            total = max(1, int((x_max - x_min) * (y_max - y_min)))
            return {
                "object_id": int(semantic_id),
                "bbox_used": [x_min, y_min, x_max, y_max],
                "pixel_counts": counts,
                "selected_pixel_ratio": counts[selected_id] / total,
                "fallback_used": True,
                "fallback_reason": "semantic hint overrode bbox pixel majority",
                "bbox_pixel_object_id": int(selected_id),
                "selected_object_bbox": visible_bboxes[semantic_id],
                **semantic_details,
            }

    total = max(1, int((x_max - x_min) * (y_max - y_min)))
    return {
        "object_id": int(selected_id),
        "bbox_used": [x_min, y_min, x_max, y_max],
        "pixel_counts": counts,
        "selected_pixel_ratio": counts[selected_id] / total,
        "fallback_used": False,
        "selected_object_bbox": visible_bboxes.get(int(selected_id), [x_min, y_min, x_max, y_max]),
        "selected_metadata": object_metadata.get(int(selected_id), {}) if object_metadata else {},
    }


def draw_bbox(image: np.ndarray, bbox: list[int], label: str, output_path: Path) -> None:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    cv2.putText(
        image_bgr,
        label,
        (x_min, max(24, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)


def save_rgb(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def create_object_point_cloud_from_render(
    render_data: dict,
    object_id: int,
    *,
    min_depth: float = 0.05,
    max_depth: float = CAMERA_FAR,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rgb = render_data["rgb"]
    depth_m = render_data["depth_m"]
    segmentation = render_data["segmentation"]
    intrinsics = render_data["intrinsics"]

    object_uid = decode_object_uid(segmentation)
    mask = object_uid == int(object_id)
    valid = mask & np.isfinite(depth_m) & (depth_m >= min_depth) & (depth_m <= max_depth)
    ys, xs = np.where(valid)
    if xs.size == 0:
        raise RuntimeError(f"No valid RGB-D points found for object_id={object_id}.")

    z = depth_m[ys, xs].astype(np.float32)
    x = ((xs.astype(np.float32) - intrinsics["cx"]) * z / intrinsics["fx"]).astype(np.float32)
    y = ((ys.astype(np.float32) - intrinsics["cy"]) * z / intrinsics["fy"]).astype(np.float32)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = (rgb[ys, xs].astype(np.float32) / 255.0).astype(np.float32)

    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    t_world_camera = np.asarray(render_data["T_world_camera_cv"], dtype=np.float32)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    points_world = (t_world_camera @ points_h.T).T[:, :3]

    meta = {
        "object_id": int(object_id),
        "num_points": int(points.shape[0]),
        "bbox": bbox,
        "camera_intrinsics": {key: float(value) for key, value in intrinsics.items()},
        "camera": render_data["camera"],
        "centroid_camera_xyz_m": points.mean(axis=0).astype(float).tolist(),
        "bounds_camera_min_xyz_m": points.min(axis=0).astype(float).tolist(),
        "bounds_camera_max_xyz_m": points.max(axis=0).astype(float).tolist(),
        "centroid_world_xyz_m": points_world.mean(axis=0).astype(float).tolist(),
        "bounds_world_min_xyz_m": points_world.min(axis=0).astype(float).tolist(),
        "bounds_world_max_xyz_m": points_world.max(axis=0).astype(float).tolist(),
        "T_world_camera_cv": t_world_camera.astype(float).tolist(),
    }
    return points, colors, meta


def save_depth_visual(depth_m: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.isfinite(depth_m)
    if np.any(finite):
        depth_min = float(depth_m[finite].min())
        depth_max = float(depth_m[finite].max())
        depth_vis = np.clip((depth_m - depth_min) / max(depth_max - depth_min, 1e-6), 0, 1)
        depth_vis = (depth_vis * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros(depth_m.shape, dtype=np.uint8)
    cv2.imwrite(str(output_path), depth_vis)


def run_graspnet_for_object(
    render_data: dict,
    object_id: int,
    *,
    output_dir: Path,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    adapter=None,
    num_point: int = 20000,
    num_view: int = 300,
    top_k: int = 20,
    collision_thresh: float = 0.01,
    voxel_size: float = 0.01,
) -> tuple[dict, object]:
    from src.grasp_selector import grasp_group_to_dicts, select_best_grasp
    from src.graspnet_adapter import GraspNetAdapter
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=True)
    points, colors, pointcloud_meta = create_object_point_cloud_from_render(render_data, object_id)

    if adapter is None:
        checkpoint_path = Path(checkpoint_path).resolve()
        repo_root = checkpoint_path.parent.parent
        adapter = GraspNetAdapter(checkpoint_path=checkpoint_path, repo_root=repo_root, num_view=num_view)

    grasp_group, cloud, sampled = adapter.predict_from_points(
        points,
        colors,
        num_point=num_point,
        collision_thresh=collision_thresh,
        voxel_size=voxel_size,
    )
    top_grasps = grasp_group_to_dicts(grasp_group, top_k=top_k)
    best_grasp = select_best_grasp(grasp_group)
    if best_grasp is None:
        raise RuntimeError("GraspNet did not return any valid grasp candidates for the selected object.")

    t_world_camera = np.asarray(render_data["T_world_camera_cv"], dtype=np.float32)
    grasp_world = transform_grasp_camera_to_world(
        best_grasp,
        r_world_camera=t_world_camera[:3, :3],
        t_world_camera=t_world_camera[:3, 3],
    )

    np.save(output_dir / "target_points_camera.npy", points)
    np.save(output_dir / "sampled_points_camera.npy", sampled)
    np.save(output_dir / "target_grasps.npy", grasp_group.grasp_group_array)
    o3d.io.write_point_cloud(str(output_dir / "target_cloud_for_graspnet.ply"), cloud, write_ascii=False)
    (output_dir / "target_pointcloud_meta.json").write_text(
        json.dumps(pointcloud_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "top_grasps.json").write_text(json.dumps(top_grasps, indent=2), encoding="utf-8")

    summary = {
        "status": "graspnet_completed",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": int(adapter.epoch),
        "selected_object_id": int(object_id),
        "num_input_points": int(points.shape[0]),
        "num_sampled_points": int(sampled.shape[0]),
        "num_grasps_after_filtering": int(len(grasp_group)),
        "collision_thresh": float(collision_thresh),
        "voxel_size": float(voxel_size),
        "pointcloud": pointcloud_meta,
        "best_grasp_camera": best_grasp,
        "best_grasp_world": grasp_world,
    }
    (output_dir / "best_grasp.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Panda pick-place selected by VLM bbox.")
    parser.add_argument("--command", default="Hay gap vat the phu hop nhat tren ban.")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--num-objects", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--realtime-sleep", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--mock-object-index", type=int, default=None, help="Skip VLM and use bbox of this object index.")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"))
    parser.add_argument("--vlm-backend", default=None, help="VLM backend: qwen-local or gemini. Default: VLM_BACKEND or qwen-local.")
    parser.add_argument("--vlm-model", default=None, help="Model id/name for the selected VLM backend.")
    parser.add_argument("--camera-width", type=int, default=960)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--no-graspnet", action="store_true", help="Use the old object-center heuristic instead of GraspNet.")
    parser.add_argument("--use-graspnet-orientation", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Motion slowdown multiplier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panda_module = load_panda_module()
    sim = panda_module.PandaPickPlaceSim(
        gui=args.gui,
        realtime_sleep=args.realtime_sleep,
        seed=args.seed,
        grasp_assist=True,
        motion_slowdown=args.speed_scale,
    )
    sim.connect()
    try:
        sim.setup_scene(args.num_objects)
        render_data = render_camera_data(width=args.camera_width, height=args.camera_height)
        rgb = render_data["rgb"]
        segmentation = render_data["segmentation"]
        render_path = output_dir / "01_render_rgb.png"
        save_rgb(rgb, render_path)
        save_depth_visual(render_data["depth_m"], output_dir / "01_render_depth.png")

        if args.bbox:
            vlm_result = {
                "target_object": "manual_bbox",
                "bbox": [int(v) for v in args.bbox],
                "confidence": 1.0,
                "reason": "manual bbox supplied by user",
            }
        elif args.mock_object_index is not None:
            object_id = sim.object_ids[args.mock_object_index]
            bbox = bbox_from_object(segmentation, object_id)
            vlm_result = {
                "target_object": f"mock_object_{object_id}",
                "bbox": bbox,
                "confidence": 1.0,
                "reason": "mock object bbox from PyBullet segmentation",
            }
        else:
            vlm_result = localize_target_object(
                render_path,
                args.command,
                backend=args.vlm_backend,
                model=args.vlm_model,
                extra_rules=(
                    "- Select only one small movable object resting on the gray table.\n"
                    "- Do not select the robot arm, gripper, floor, table, gray platform, or blue bin/tray.\n"
                    "- If the user asks for a color and the blue tray/bin matches, ignore the tray/bin and choose the best small tabletop object instead."
                ),
            )

        vlm_path = output_dir / "02_vlm_result.json"
        vlm_path.write_text(json.dumps(vlm_result, indent=2, ensure_ascii=False), encoding="utf-8")

        selection = select_object_from_bbox(
            segmentation,
            vlm_result["bbox"],
            sim.object_ids,
            object_metadata=sim.object_metadata,
            text_hint=f'{args.command} {vlm_result.get("target_object", "")} {vlm_result.get("reason", "")}',
        )
        selected_object_id = selection["object_id"]
        draw_bbox(
            rgb,
            selection.get("selected_object_bbox", selection["bbox_used"]),
            f'{vlm_result["target_object"]} -> id {selected_object_id}',
            output_dir / "03_selected_bbox.png",
        )

        if args.no_graspnet:
            graspnet_summary = {"status": "skipped", "reason": "--no-graspnet"}
            pick_result = sim.pick_and_place(selected_object_id)
        else:
            graspnet_summary, _ = run_graspnet_for_object(
                render_data,
                selected_object_id,
                output_dir=output_dir / "04_graspnet",
                checkpoint_path=args.checkpoint,
                num_point=args.num_point,
                num_view=args.num_view,
                top_k=args.top_k,
                collision_thresh=args.collision_thresh,
                voxel_size=args.voxel_size,
            )
            pick_result = sim.pick_and_place_with_graspnet_pose(
                selected_object_id,
                graspnet_summary["best_grasp_world"],
                use_grasp_orientation=args.use_graspnet_orientation,
            )
        summary = {
            "status": "completed",
            "command": args.command,
            "robot": "franka_panda/panda.urdf",
            "num_objects": args.num_objects,
            "seed": args.seed,
            "render_rgb": str(render_path),
            "vlm_result": vlm_result,
            "selection": selection,
            "graspnet": graspnet_summary,
            "pick_place": pick_result,
        }
        summary_path = output_dir / "vlm_panda_result.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved result: {summary_path}")

        if args.gui and args.keep_open:
            print("Press Ctrl+C in the terminal to close the PyBullet GUI.")
            while True:
                sim.step_sim(1)
    finally:
        if not (args.gui and args.keep_open):
            sim.disconnect()


if __name__ == "__main__":
    main()
