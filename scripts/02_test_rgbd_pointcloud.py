import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rgbd_to_pointcloud import (
    create_point_cloud_from_crop,
    crop_rgbd_by_bbox,
    load_camera_from_mat,
    load_rgb_depth,
    load_vlm_bbox,
    save_crop_artifacts,
)


EXAMPLE_DATA = PROJECT_ROOT / "graspnet-baseline" / "doc" / "example_data"
DEFAULT_RGB = EXAMPLE_DATA / "color.png"
DEFAULT_DEPTH = EXAMPLE_DATA / "depth.png"
DEFAULT_META = EXAMPLE_DATA / "meta.mat"
DEFAULT_VLM_RESULT = PROJECT_ROOT / "data" / "outputs" / "vlm_result.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "pointcloud"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a target point cloud from RGB-D and a VLM bbox.")
    parser.add_argument("--rgb", default=str(DEFAULT_RGB), help="RGB image path")
    parser.add_argument("--depth", default=str(DEFAULT_DEPTH), help="Depth image path")
    parser.add_argument("--meta", default=str(DEFAULT_META), help="GraspNet meta.mat path")
    parser.add_argument("--vlm-result", default=str(DEFAULT_VLM_RESULT), help="VLM JSON output path")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"))
    parser.add_argument("--margin", type=int, default=8, help="Pixel margin added around bbox")
    parser.add_argument("--min-depth", type=float, default=0.05, help="Minimum valid depth in meters")
    parser.add_argument("--max-depth", type=float, default=2.0, help="Maximum valid depth in meters")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bbox = args.bbox if args.bbox else load_vlm_bbox(args.vlm_result)
    intrinsics = load_camera_from_mat(args.meta)
    rgb, depth_raw = load_rgb_depth(args.rgb, args.depth)

    rgb_crop, depth_crop, offset_xy, clamped_bbox = crop_rgbd_by_bbox(
        rgb,
        depth_raw,
        bbox,
        margin=args.margin,
    )
    pcd, points, _ = create_point_cloud_from_crop(
        rgb_crop,
        depth_crop,
        offset_xy,
        intrinsics,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    centroid = points.mean(axis=0)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    metadata = {
        "rgb": str(Path(args.rgb).resolve()),
        "depth": str(Path(args.depth).resolve()),
        "meta": str(Path(args.meta).resolve()),
        "vlm_result": str(Path(args.vlm_result).resolve()) if not args.bbox else None,
        "bbox_original": [int(v) for v in bbox],
        "bbox_used": [int(v) for v in clamped_bbox],
        "bbox_margin": args.margin,
        "crop_shape_hw": [int(rgb_crop.shape[0]), int(rgb_crop.shape[1])],
        "num_points": int(points.shape[0]),
        "centroid_xyz_m": centroid.astype(float).tolist(),
        "bounds_min_xyz_m": bounds_min.astype(float).tolist(),
        "bounds_max_xyz_m": bounds_max.astype(float).tolist(),
        "intrinsics": intrinsics,
    }

    paths = save_crop_artifacts(args.output_dir, rgb_crop, depth_crop, pcd, points, metadata)
    print(json.dumps(metadata, indent=2))
    print("Saved artifacts:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
