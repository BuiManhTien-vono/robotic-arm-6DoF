import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasp_selector import grasp_group_to_dicts, select_best_grasp
from src.graspnet_adapter import GraspNetAdapter
from src.rgbd_to_pointcloud import (
    create_point_cloud_from_crop,
    crop_rgbd_by_bbox,
    load_camera_from_mat,
    load_rgb_depth,
    load_vlm_bbox,
)


EXAMPLE_DATA = PROJECT_ROOT / "graspnet-baseline" / "doc" / "example_data"
DEFAULT_RGB = EXAMPLE_DATA / "color.png"
DEFAULT_DEPTH = EXAMPLE_DATA / "depth.png"
DEFAULT_META = EXAMPLE_DATA / "meta.mat"
DEFAULT_VLM_RESULT = PROJECT_ROOT / "data" / "outputs" / "vlm_result.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "graspnet-baseline" / "checkpoints" / "checkpoint-rs.tar"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "graspnet_target"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GraspNet on the VLM target crop.")
    parser.add_argument("--rgb", default=str(DEFAULT_RGB))
    parser.add_argument("--depth", default=str(DEFAULT_DEPTH))
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--vlm-result", default=str(DEFAULT_VLM_RESULT))
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--vis", action="store_true", help="Open Open3D visualizer for top grasps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = args.bbox if args.bbox else load_vlm_bbox(args.vlm_result)
    intrinsics = load_camera_from_mat(args.meta)
    rgb, depth_raw = load_rgb_depth(args.rgb, args.depth)
    rgb_crop, depth_crop, offset_xy, bbox_used = crop_rgbd_by_bbox(
        rgb,
        depth_raw,
        bbox,
        margin=args.margin,
    )
    pcd, points, colors = create_point_cloud_from_crop(
        rgb_crop,
        depth_crop,
        offset_xy,
        intrinsics,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    adapter = GraspNetAdapter(
        checkpoint_path=args.checkpoint,
        num_view=args.num_view,
    )
    grasp_group, cloud, sampled = adapter.predict_from_points(
        points,
        colors,
        num_point=args.num_point,
        collision_thresh=args.collision_thresh,
        voxel_size=args.voxel_size,
    )
    top_grasps = grasp_group_to_dicts(grasp_group, top_k=args.top_k)
    best_grasp = select_best_grasp(grasp_group)

    grasps_npy = output_dir / "target_grasps.npy"
    top_json = output_dir / "top_grasps.json"
    best_json = output_dir / "best_grasp.json"
    sampled_npy = output_dir / "sampled_points.npy"
    cloud_ply = output_dir / "target_cloud_for_graspnet.ply"

    np.save(grasps_npy, grasp_group.grasp_group_array)
    np.save(sampled_npy, sampled)
    o3d.io.write_point_cloud(str(cloud_ply), cloud, write_ascii=False)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": adapter.epoch,
        "bbox_original": [int(v) for v in bbox],
        "bbox_used": [int(v) for v in bbox_used],
        "num_input_points": int(points.shape[0]),
        "num_sampled_points": int(sampled.shape[0]),
        "num_grasps_after_filtering": int(len(grasp_group)),
        "collision_thresh": args.collision_thresh,
        "best_grasp": best_grasp,
    }
    best_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    top_json.write_text(json.dumps(top_grasps, indent=2), encoding="utf-8")

    if args.vis and len(grasp_group) > 0:
        top_group = grasp_group[: args.top_k]
        o3d.visualization.draw_geometries([cloud, *top_group.to_open3d_geometry_list()])

    print(json.dumps(summary, indent=2))
    print("Saved artifacts:")
    print(f"- grasps_npy: {grasps_npy}")
    print(f"- top_grasps_json: {top_json}")
    print(f"- best_grasp_json: {best_json}")
    print(f"- sampled_points: {sampled_npy}")
    print(f"- point_cloud: {cloud_ply}")


if __name__ == "__main__":
    main()
