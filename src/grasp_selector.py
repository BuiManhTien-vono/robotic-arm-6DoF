from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def grasp_array_to_dict(row: np.ndarray) -> dict[str, Any]:
    row = np.asarray(row, dtype=np.float32)
    rotation_matrix = row[4:13].reshape(3, 3)
    translation = row[13:16]
    try:
        rpy = Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=False)
    except ValueError:
        rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return {
        "score": float(row[0]),
        "width": float(row[1]),
        "height": float(row[2]),
        "depth": float(row[3]),
        "rotation_matrix": rotation_matrix.astype(float).tolist(),
        "translation": translation.astype(float).tolist(),
        "rpy_xyz": rpy.astype(float).tolist(),
        "pose_6dof_xyz_rpy": np.concatenate([translation, rpy]).astype(float).tolist(),
        "object_id": int(row[16]),
    }


def grasp_group_to_dicts(grasp_group, top_k: int = 20) -> list[dict[str, Any]]:
    grasp_group.nms()
    grasp_group.sort_by_score()
    grasp_group = grasp_group[:top_k]
    return [grasp_array_to_dict(row) for row in grasp_group.grasp_group_array]


def select_best_grasp(grasp_group) -> dict[str, Any] | None:
    grasp_group.nms()
    grasp_group.sort_by_score()
    if len(grasp_group) == 0:
        return None
    return grasp_array_to_dict(grasp_group.grasp_group_array[0])
