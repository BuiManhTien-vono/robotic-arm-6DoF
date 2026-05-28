import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.coordinate_transform import (
    load_best_grasp,
    make_downward_orientation,
    make_pre_grasp_pose,
    transform_grasp_camera_to_world,
)


DEFAULT_BEST_GRASP = PROJECT_ROOT / "data" / "outputs" / "graspnet_target" / "best_grasp.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "pybullet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test PyBullet IK with the best GraspNet pose.")
    parser.add_argument("--best-grasp", default=str(DEFAULT_BEST_GRASP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gui", action="store_true", help="Open PyBullet GUI.")
    parser.add_argument(
        "--use-grasp-orientation",
        action="store_true",
        help="Use transformed GraspNet orientation instead of a stable downward tool orientation.",
    )
    parser.add_argument("--retreat-distance", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only create transformed poses; do not import/run PyBullet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_grasp_camera = load_best_grasp(args.best_grasp)
    grasp_world = transform_grasp_camera_to_world(best_grasp_camera)
    if not args.use_grasp_orientation:
        stable_orientation = make_downward_orientation(grasp_world["translation"])
        grasp_world["rpy_xyz"] = stable_orientation["rpy_xyz"]
        grasp_world["quaternion_xyzw"] = stable_orientation["quaternion_xyzw"]
        grasp_world["pose_6dof_xyz_rpy"] = (
            grasp_world["translation"] + stable_orientation["rpy_xyz"]
        )
        grasp_world["orientation_source"] = "downward_stable_for_ik"
    else:
        grasp_world["orientation_source"] = "graspnet_transformed"

    pre_grasp_world = make_pre_grasp_pose(
        grasp_world,
        retreat_distance=args.retreat_distance,
    )

    summary = {
        "status": "pose_transformed",
        "best_grasp_path": str(Path(args.best_grasp).resolve()),
        "grasp_world": grasp_world,
        "pre_grasp_world": pre_grasp_world,
        "note": (
            "Default T_world_camera is a provisional baseline transform: "
            "camera z -> world x, camera -x -> world y, camera -y -> world z, "
            "with camera height 0.55 m."
        ),
    }

    if not args.dry_run:
        try:
            from src.pybullet_controller import PyBulletRobotController, result_to_dict

            controller = PyBulletRobotController(gui=args.gui)
            controller.connect()
            controller.load_scene()
            controller.load_robot()
            controller.add_target_marker(grasp_world["translation"])
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
            summary.update(
                {
                    "status": "pybullet_ik_executed",
                    "robot_urdf": controller.robot_urdf,
                    "ee_link_index": controller.ee_link_index,
                    "arm_joint_indices": controller.arm_joint_indices,
                    "pre_grasp_ik": result_to_dict(pre_result),
                    "grasp_ik": result_to_dict(grasp_result),
                }
            )
            controller.disconnect()
        except ImportError as exc:
            summary.update(
                {
                    "status": "pybullet_missing",
                    "error": str(exc),
                    "install_hint": (
                        "Install Microsoft Visual C++ Build Tools, then run "
                        "`python -m pip install pybullet`, or run this phase "
                        "inside WSL/Ubuntu where a Linux wheel is available."
                    ),
                }
            )

    output_json = output_dir / "pybullet_ik_result.json"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved result: {output_json}")


if __name__ == "__main__":
    main()
