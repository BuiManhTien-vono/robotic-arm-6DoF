from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ROOT = PROJECT_ROOT / "graspnet-baseline"


def add_graspnet_to_path(repo_root: str | Path = DEFAULT_REPO_ROOT) -> Path:
    repo_root = Path(repo_root).resolve()
    for path in (
        repo_root,
        repo_root / "models",
        repo_root / "dataset",
        repo_root / "utils",
        repo_root / "pointnet2",
        repo_root / "knn",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


class GraspNetAdapter:
    def __init__(
        self,
        checkpoint_path: str | Path,
        repo_root: str | Path = DEFAULT_REPO_ROOT,
        *,
        num_view: int = 300,
        device: str | None = None,
    ) -> None:
        self.repo_root = add_graspnet_to_path(repo_root)
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(self.checkpoint_path)

        from graspnet import GraspNet

        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.net = GraspNet(
            input_feature_dim=0,
            num_view=num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        self.net.to(self.device)
        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.epoch = int(checkpoint.get("epoch", -1))
        self.net.eval()

    @staticmethod
    def sample_points(points: np.ndarray, num_point: int) -> np.ndarray:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        if len(points) == 0:
            raise ValueError("Cannot sample from an empty point cloud.")

        if len(points) >= num_point:
            idxs = np.random.choice(len(points), num_point, replace=False)
        else:
            idxs1 = np.arange(len(points))
            idxs2 = np.random.choice(len(points), num_point - len(points), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        return points[idxs].astype(np.float32)

    @staticmethod
    def make_open3d_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
        if colors is not None and len(colors) == len(points):
            cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
        return cloud

    def predict_from_points(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        *,
        num_point: int = 20000,
        collision_thresh: float = 0.01,
        voxel_size: float = 0.01,
    ):
        from collision_detector import ModelFreeCollisionDetector
        from graspnet import pred_decode
        from graspnetAPI import GraspGroup

        sampled = self.sample_points(points, num_point)
        end_points = {
            "point_clouds": torch.from_numpy(sampled[np.newaxis].astype(np.float32)).to(self.device)
        }
        if colors is not None:
            end_points["cloud_colors"] = colors

        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)

        grasp_group = GraspGroup(grasp_preds[0].detach().cpu().numpy())
        if collision_thresh > 0 and len(grasp_group) > 0:
            detector = ModelFreeCollisionDetector(points.astype(np.float32), voxel_size=voxel_size)
            collision_mask = detector.detect(
                grasp_group,
                approach_dist=0.05,
                collision_thresh=collision_thresh,
            )
            grasp_group = grasp_group[~collision_mask]

        grasp_group.nms()
        grasp_group.sort_by_score()
        cloud = self.make_open3d_cloud(points, colors)
        return grasp_group, cloud, sampled
