import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.coordinate_transform import (
    make_downward_orientation,
    make_pre_grasp_pose,
    transform_grasp_camera_to_world,
)
from src.grasp_selector import grasp_group_to_dicts, select_best_grasp
from src.graspnet_adapter import GraspNetAdapter
from src.pybullet_controller import PyBulletRobotController, result_to_dict
from src.rgbd_to_pointcloud import (
    create_point_cloud_from_crop,
    crop_rgbd_by_bbox,
    load_camera_from_mat,
    load_rgb_depth,
    load_vlm_bbox,
    save_crop_artifacts,
)
from src.vlm_localizer import localize_target_object


EXAMPLE_DATA = PROJECT_ROOT / "graspnet-baseline" / "doc" / "example_data"
DEFAULT_RGB = EXAMPLE_DATA / "color.png"
DEFAULT_DEPTH = EXAMPLE_DATA / "depth.png"
DEFAULT_META = EXAMPLE_DATA / "meta.mat"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "graspnet-baseline" / "checkpoints" / "checkpoint-rs.tar"
DEFAULT_VLM_RESULT = PROJECT_ROOT / "data" / "outputs" / "vlm_result.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "full_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the semantic grasping baseline end to end.")
    parser.add_argument("--command", default="Hay gap vat the phu hop nhat trong canh.")
    parser.add_argument("--rgb", default=str(DEFAULT_RGB))
    parser.add_argument("--depth", default=str(DEFAULT_DEPTH))
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--use-existing-vlm", action="store_true", help="Read bbox from --vlm-result instead of calling the VLM.")
    parser.add_argument("--vlm-result", default=str(DEFAULT_VLM_RESULT))
    parser.add_argument("--vlm-backend", default=None, help="VLM backend: qwen-local or gemini. Default: VLM_BACKEND or qwen-local.")
    parser.add_argument("--vlm-model", default=None, help="Model id/name for the selected VLM backend.")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"))
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--gui", action="store_true", help="Open PyBullet GUI.")
    parser.add_argument("--no-pybullet", action="store_true", help="Stop after GraspNet/pose transform.")
    parser.add_argument("--use-grasp-orientation", action="store_true")
    parser.add_argument("--retreat-distance", type=float, default=0.10)
    parser.add_argument("--lift-height", type=float, default=0.08)
    parser.add_argument("--place-position", nargs=3, type=float, default=[0.45, -0.25, 0.35])
    parser.add_argument("--steps", type=int, default=240)
    return parser.parse_args()


def save_bbox_image(rgb_path: Path, bbox: list[int], label: str, output_path: Path) -> None:
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(rgb_path)
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    cv2.putText(
        image,
        label,
        (x_min, max(24, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = Path(args.rgb)
    depth_path = Path(args.depth)
    meta_path = Path(args.meta)

    if args.bbox:
        vlm_result = {
            "target_object": "manual_target",
            "bbox": [int(v) for v in args.bbox],
            "confidence": 1.0,
            "reason": "manual bbox",
        }
    elif args.use_existing_vlm:
        vlm_result = json.loads(Path(args.vlm_result).read_text(encoding="utf-8"))
    else:
        vlm_result = localize_target_object(
            rgb_path,
            args.command,
            backend=args.vlm_backend,
            model=args.vlm_model,
        )

    bbox_source = vlm_result.get("bbox") or vlm_result.get("bbox_2d")
    if not isinstance(bbox_source, list) or len(bbox_source) != 4:
        raise ValueError(f"Invalid VLM bbox: {bbox_source}")
    bbox = [int(v) for v in bbox_source]
    vlm_json = output_dir / "01_vlm_result.json"
    vlm_json.write_text(json.dumps(vlm_result, indent=2, ensure_ascii=False), encoding="utf-8")
    save_bbox_image(
        rgb_path,
        bbox,
        f'{vlm_result["target_object"]} {vlm_result["confidence"]:.2f}',
        output_dir / "01_vlm_bbox.png",
    )

    intrinsics = load_camera_from_mat(meta_path)
    rgb, depth_raw = load_rgb_depth(rgb_path, depth_path)
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
    pointcloud_meta = {
        "bbox_original": bbox,
        "bbox_used": [int(v) for v in bbox_used],
        "bbox_margin": args.margin,
        "crop_shape_hw": [int(rgb_crop.shape[0]), int(rgb_crop.shape[1])],
        "num_points": int(points.shape[0]),
        "centroid_xyz_m": points.mean(axis=0).astype(float).tolist(),
        "bounds_min_xyz_m": points.min(axis=0).astype(float).tolist(),
        "bounds_max_xyz_m": points.max(axis=0).astype(float).tolist(),
        "intrinsics": intrinsics,
    }
    pointcloud_paths = save_crop_artifacts(
        output_dir / "02_pointcloud",
        rgb_crop,
        depth_crop,
        pcd,
        points,
        pointcloud_meta,
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
    if best_grasp is None:
        raise RuntimeError("GraspNet did not return any valid grasp candidates.")

    grasp_dir = output_dir / "03_graspnet"
    grasp_dir.mkdir(parents=True, exist_ok=True)
    np.save(grasp_dir / "target_grasps.npy", grasp_group.grasp_group_array)
    np.save(grasp_dir / "sampled_points.npy", sampled)
    o3d.io.write_point_cloud(str(grasp_dir / "target_cloud_for_graspnet.ply"), cloud, write_ascii=False)
    (grasp_dir / "top_grasps.json").write_text(json.dumps(top_grasps, indent=2), encoding="utf-8")
    grasp_summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": adapter.epoch,
        "num_input_points": int(points.shape[0]),
        "num_sampled_points": int(sampled.shape[0]),
        "num_grasps_after_filtering": int(len(grasp_group)),
        "collision_thresh": args.collision_thresh,
        "best_grasp": best_grasp,
    }
    (grasp_dir / "best_grasp.json").write_text(json.dumps(grasp_summary, indent=2), encoding="utf-8")

    grasp_world = transform_grasp_camera_to_world(best_grasp)
    if not args.use_grasp_orientation:
        stable_orientation = make_downward_orientation(grasp_world["translation"])
        grasp_world["rpy_xyz"] = stable_orientation["rpy_xyz"]
        grasp_world["quaternion_xyzw"] = stable_orientation["quaternion_xyzw"]
        grasp_world["pose_6dof_xyz_rpy"] = grasp_world["translation"] + stable_orientation["rpy_xyz"]
        grasp_world["orientation_source"] = "downward_stable_for_ik"
    else:
        grasp_world["orientation_source"] = "graspnet_transformed"
    pre_grasp_world = make_pre_grasp_pose(grasp_world, retreat_distance=args.retreat_distance)

    pybullet_summary = {
        "status": "skipped",
        "grasp_world": grasp_world,
        "pre_grasp_world": pre_grasp_world,
    }
    if not args.no_pybullet:
        controller = PyBulletRobotController(gui=args.gui)
        controller.connect()
        try:
            controller.load_scene()
            controller.load_robot()
            object_id = controller.add_grasp_object(grasp_world["translation"])
            controller.add_target_marker(grasp_world["translation"])
            controller.add_target_marker(args.place_position, rgba=[0.1, 0.35, 1.0, 0.85])

            pre_result = controller.execute_pose(
                pre_grasp_world["translation"],
                grasp_world["quaternion_xyzw"],
                steps=args.steps,
            )
            grasp_result = controller.execute_pose(
                grasp_world["translation"],
                grasp_world["quaternion_xyzw"],
                steps=args.steps,
            )
            constraint_id = controller.attach_body_to_ee(object_id)
            controller.step(steps=60)

            lift_pose = dict(grasp_world)
            lift_pose["translation"] = [
                grasp_world["translation"][0],
                grasp_world["translation"][1],
                grasp_world["translation"][2] + args.lift_height,
            ]
            lift_result = controller.execute_pose(
                lift_pose["translation"],
                grasp_world["quaternion_xyzw"],
                steps=args.steps,
            )
            place_result = controller.execute_pose(
                args.place_position,
                grasp_world["quaternion_xyzw"],
                steps=args.steps,
            )
            controller.detach_body(constraint_id)
            controller.step(steps=120)

            pybullet_summary = {
                "status": "pick_place_executed",
                "robot_urdf": controller.robot_urdf,
                "ee_link_index": controller.ee_link_index,
                "arm_joint_indices": controller.arm_joint_indices,
                "object_id": object_id,
                "place_position": [float(v) for v in args.place_position],
                "pre_grasp_ik": result_to_dict(pre_result),
                "grasp_ik": result_to_dict(grasp_result),
                "lift_ik": result_to_dict(lift_result),
                "place_ik": result_to_dict(place_result),
                "grasp_world": grasp_world,
                "pre_grasp_world": pre_grasp_world,
            }
        finally:
            controller.disconnect()

    summary = {
        "status": "full_pipeline_completed",
        "command": args.command,
        "rgb": str(rgb_path.resolve()),
        "depth": str(depth_path.resolve()),
        "vlm": vlm_result,
        "pointcloud": pointcloud_meta,
        "pointcloud_artifacts": pointcloud_paths,
        "graspnet": grasp_summary,
        "pybullet": pybullet_summary,
    }
    summary_path = output_dir / "full_pipeline_result.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved full pipeline result: {summary_path}")


if __name__ == "__main__":
    main()
