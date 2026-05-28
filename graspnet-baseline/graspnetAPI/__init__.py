import numpy as np
import open3d as o3d


class GraspGroup:
    """Small local fallback for GraspNet API's GraspGroup used by demo.py."""

    def __init__(self, grasp_group_array=None):
        if grasp_group_array is None:
            grasp_group_array = np.zeros((0, 17), dtype=np.float32)
        arr = np.asarray(grasp_group_array, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != 17:
            raise ValueError(f"Expected grasp array with 17 columns, got {arr.shape}")
        self.grasp_group_array = arr

    def __len__(self):
        return len(self.grasp_group_array)

    def __getitem__(self, index):
        return GraspGroup(self.grasp_group_array[index])

    @property
    def scores(self):
        return self.grasp_group_array[:, 0]

    @property
    def widths(self):
        return self.grasp_group_array[:, 1]

    @property
    def heights(self):
        return self.grasp_group_array[:, 2]

    @property
    def depths(self):
        return self.grasp_group_array[:, 3]

    @property
    def rotation_matrices(self):
        return self.grasp_group_array[:, 4:13].reshape(-1, 3, 3)

    @property
    def translations(self):
        return self.grasp_group_array[:, 13:16]

    @property
    def object_ids(self):
        return self.grasp_group_array[:, 16]

    def nms(self, *args, **kwargs):
        return self

    def sort_by_score(self):
        order = np.argsort(-self.scores)
        self.grasp_group_array = self.grasp_group_array[order]
        return self

    def save_npy(self, path):
        np.save(path, self.grasp_group_array)

    def to_open3d_geometry_list(self):
        geoms = []
        if len(self) == 0:
            return geoms

        max_score = float(np.max(self.scores))
        min_score = float(np.min(self.scores))
        denom = max(max_score - min_score, 1e-6)

        for score, width, height, depth, rotation, translation in zip(
            self.scores,
            self.widths,
            self.heights,
            self.depths,
            self.rotation_matrices,
            self.translations,
        ):
            finger_len = 0.06
            y = max(float(width) / 2.0, 0.005)
            z = max(float(height) / 2.0, 0.005)
            d = float(depth)
            local = np.array(
                [
                    [d, -y, -z],
                    [d, -y, z],
                    [d, y, -z],
                    [d, y, z],
                    [d - finger_len, -y, -z],
                    [d - finger_len, -y, z],
                    [d - finger_len, y, -z],
                    [d - finger_len, y, z],
                    [0.0, 0.0, 0.0],
                    [d - finger_len, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            points = local @ rotation.T + translation
            lines = np.array(
                [
                    [0, 1],
                    [2, 3],
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                    [4, 5],
                    [6, 7],
                    [8, 9],
                ],
                dtype=np.int32,
            )
            confidence = (float(score) - min_score) / denom
            color = [1.0 - confidence, confidence, 0.1]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
            geoms.append(line_set)
        return geoms


class GraspNetEval:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "Local graspnetAPI fallback only supports GraspGroup for demo inference. "
            "Install the official graspnetAPI package to use GraspNetEval."
        )
