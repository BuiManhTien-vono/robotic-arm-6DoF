from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_R_WORLD_CAMERA = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)
DEFAULT_T_WORLD_CAMERA = np.array([0.0, 0.0, 0.55], dtype=np.float32)


def load_best_grasp(best_grasp_path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(best_grasp_path).read_text(encoding="utf-8"))
    grasp = data.get("best_grasp")
    if not grasp:
        raise ValueError(f"No best_grasp field found in {best_grasp_path}")
    return grasp


def transform_grasp_camera_to_world(
    grasp: dict[str, Any],
    *,
    r_world_camera: np.ndarray = DEFAULT_R_WORLD_CAMERA,
    t_world_camera: np.ndarray = DEFAULT_T_WORLD_CAMERA,
) -> dict[str, Any]:
    translation_camera = np.asarray(grasp["translation"], dtype=np.float32)
    rotation_camera = np.asarray(grasp["rotation_matrix"], dtype=np.float32)
    r_world_camera = np.asarray(r_world_camera, dtype=np.float32).reshape(3, 3)
    t_world_camera = np.asarray(t_world_camera, dtype=np.float32).reshape(3)

    translation_world = r_world_camera @ translation_camera + t_world_camera
    rotation_world = r_world_camera @ rotation_camera
    rpy_world = Rotation.from_matrix(rotation_world).as_euler("xyz", degrees=False)
    quaternion_world = Rotation.from_matrix(rotation_world).as_quat()

    return {
        "translation": translation_world.astype(float).tolist(),
        "rotation_matrix": rotation_world.astype(float).tolist(),
        "rpy_xyz": rpy_world.astype(float).tolist(),
        "quaternion_xyzw": quaternion_world.astype(float).tolist(),
        "pose_6dof_xyz_rpy": np.concatenate([translation_world, rpy_world]).astype(float).tolist(),
        "source_camera_grasp": grasp,
        "T_world_camera": {
            "rotation_matrix": r_world_camera.astype(float).tolist(),
            "translation": t_world_camera.astype(float).tolist(),
        },
    }


def make_downward_orientation(target_position: list[float] | np.ndarray) -> dict[str, Any]:
    position = np.asarray(target_position, dtype=np.float32)
    yaw = float(np.arctan2(position[1], position[0]))
    rpy = np.array([np.pi, 0.0, yaw], dtype=np.float32)
    quat = Rotation.from_euler("xyz", rpy).as_quat()
    return {
        "rpy_xyz": rpy.astype(float).tolist(),
        "quaternion_xyzw": quat.astype(float).tolist(),
    }


def make_pre_grasp_pose(
    grasp_world: dict[str, Any],
    *,
    retreat_distance: float = 0.10,
) -> dict[str, Any]:
    grasp_position = np.asarray(grasp_world["translation"], dtype=np.float32)
    if grasp_world.get("orientation_source") == "downward_stable_for_ik":
        approach_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        grasp_rotation = np.asarray(grasp_world["rotation_matrix"], dtype=np.float32)
        approach_axis = grasp_rotation[:, 0]
        norm = np.linalg.norm(approach_axis)
        if norm < 1e-6:
            approach_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            approach_axis = approach_axis / norm

    pre_position = grasp_position - retreat_distance * approach_axis
    pre_position[2] = max(pre_position[2], grasp_position[2] + 0.05)

    pose = dict(grasp_world)
    pose["translation"] = pre_position.astype(float).tolist()
    pose["pose_6dof_xyz_rpy"] = (
        pre_position.tolist() + list(grasp_world["rpy_xyz"])
    )
    return pose
