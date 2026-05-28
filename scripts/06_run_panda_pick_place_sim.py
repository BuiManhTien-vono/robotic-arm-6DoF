import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "panda_sim"


TABLE_TOP_Z = 0.14
TABLE_THICKNESS = 0.08
TABLE_CENTER = np.array([0.57, -0.13, TABLE_TOP_Z - TABLE_THICKNESS / 2], dtype=np.float32)
TABLE_HALF_EXTENTS = np.array([0.34, 0.28, TABLE_THICKNESS / 2], dtype=np.float32)

PRE_GRASP_HEIGHT = 0.15
LIFT_HEIGHT = 0.25
MIN_GRASP_LIFT_Z = TABLE_TOP_Z + 0.08

ARM_JOINTS = list(range(7))
FINGER_JOINTS = [9, 10]
EE_LINK_INDEX = 11
HOME_JOINTS = [0.0, -0.45, 0.0, -2.35, 0.0, 1.95, 0.78]
REST_JOINTS = HOME_JOINTS[:]
PANDA_LOWER_LIMITS = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
PANDA_UPPER_LIMITS = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
PANDA_JOINT_RANGES = [u - l for l, u in zip(PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)]

BIN_CENTER = np.array([0.50, 0.40, 0.05], dtype=np.float32)
BIN_SIZE = np.array([0.30, 0.30, 0.10], dtype=np.float32)
BIN_WALL_THICKNESS = 0.01
OBJECT_TYPE_SEQUENCE = ["box", "cylinder", "sphere", "book", "bottle", "sphere", "box", "cylinder"]
COLOR_PALETTE = [
    ("red", [0.85, 0.05, 0.05, 1.0]),
    ("green", [0.05, 0.70, 0.10, 1.0]),
    ("blue", [0.05, 0.20, 0.90, 1.0]),
    ("yellow", [0.95, 0.80, 0.05, 1.0]),
    ("magenta", [0.90, 0.05, 0.55, 1.0]),
    ("cyan", [0.05, 0.80, 0.85, 1.0]),
    ("brown", [0.35, 0.18, 0.08, 1.0]),
    ("white", [0.90, 0.90, 0.86, 1.0]),
]


class PandaPickPlaceSim:
    def __init__(
        self,
        *,
        gui: bool,
        realtime_sleep: bool,
        seed: int,
        sim_hz: int = 240,
        grasp_assist: bool = True,
        motion_slowdown: float = 1.0,
    ) -> None:
        self.gui = gui
        self.realtime_sleep = realtime_sleep
        self.seed = seed
        self.sim_hz = sim_hz
        self.dt = 1.0 / sim_hz
        self.grasp_assist = grasp_assist
        self.motion_slowdown = max(0.1, float(motion_slowdown))
        self.robot_id: int | None = None
        self.object_ids: list[int] = []
        self.object_metadata: dict[int, dict] = {}
        random.seed(seed)
        np.random.seed(seed)

    def connect(self) -> None:
        if p.isConnected():
            p.disconnect()
        p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.8,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.55, -0.1, 0.2],
            )

    def disconnect(self) -> None:
        if p.isConnected():
            p.disconnect()

    def step_sim(self, steps: int = 1) -> None:
        for _ in range(steps):
            p.stepSimulation()
            if self.realtime_sleep and self.gui:
                time.sleep(self.dt)

    def create_static_box(self, position, half_extents, color) -> int:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return p.createMultiBody(0, col, vis, position)

    def setup_scene(self, num_objects: int) -> None:
        p.loadURDF("plane.urdf")
        self.create_static_box(TABLE_CENTER, TABLE_HALF_EXTENTS, [0.5, 0.5, 0.5, 1.0])
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        self.reset_robot()
        self.create_bin()
        self.object_metadata = {}
        self.object_ids = [self.create_random_object(i) for i in range(num_objects)]
        self.step_sim(120)

    def create_bin(self) -> None:
        self.create_static_box(
            BIN_CENTER,
            [BIN_SIZE[0] / 2, BIN_SIZE[1] / 2, BIN_WALL_THICKNESS],
            [0.2, 0.2, 0.8, 0.5],
        )
        self.create_static_box(
            BIN_CENTER + [BIN_SIZE[0] / 2, 0, BIN_SIZE[2] / 2],
            [BIN_WALL_THICKNESS, BIN_SIZE[1] / 2, BIN_SIZE[2] / 2],
            [0.2, 0.2, 0.8, 0.5],
        )
        self.create_static_box(
            BIN_CENTER - [BIN_SIZE[0] / 2, 0, -BIN_SIZE[2] / 2],
            [BIN_WALL_THICKNESS, BIN_SIZE[1] / 2, BIN_SIZE[2] / 2],
            [0.2, 0.2, 0.8, 0.5],
        )
        self.create_static_box(
            BIN_CENTER + [0, BIN_SIZE[1] / 2, BIN_SIZE[2] / 2],
            [BIN_SIZE[0] / 2, BIN_WALL_THICKNESS, BIN_SIZE[2] / 2],
            [0.2, 0.2, 0.8, 0.5],
        )
        self.create_static_box(
            BIN_CENTER - [0, BIN_SIZE[1] / 2, -BIN_SIZE[2] / 2],
            [BIN_SIZE[0] / 2, BIN_WALL_THICKNESS, BIN_SIZE[2] / 2],
            [0.2, 0.2, 0.8, 0.5],
        )

    def reset_robot(self) -> None:
        assert self.robot_id is not None
        for joint, value in zip(ARM_JOINTS, HOME_JOINTS):
            p.resetJointState(self.robot_id, joint, value)
        self.open_gripper()
        self.step_sim(50)

    def create_random_object(self, index: int) -> int:
        x = random.uniform(TABLE_CENTER[0] - 0.20, TABLE_CENTER[0] + 0.20)
        y = random.uniform(TABLE_CENTER[1] - 0.15, TABLE_CENTER[1] + 0.15)
        z = TABLE_TOP_Z + 0.06
        object_type = OBJECT_TYPE_SEQUENCE[index % len(OBJECT_TYPE_SEQUENCE)]
        color_name, color = random.choice(COLOR_PALETTE)

        if object_type in ("box", "book"):
            half_extents = [
                random.uniform(0.02, 0.05),
                random.uniform(0.02, 0.04),
                random.uniform(0.01, 0.03),
            ]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
            mass = 0.10
            shape = "box"
        elif object_type in ("cylinder", "bottle"):
            radius = random.uniform(0.018, 0.030)
            height = random.uniform(0.05, 0.12 if object_type == "bottle" else 0.08)
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
            mass = 0.12
            shape = "cylinder"
        else:
            radius = random.uniform(0.020, 0.030)
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
            mass = 0.05
            shape = "sphere"

        body = p.createMultiBody(mass, col, vis, [x, y, z])
        p.changeDynamics(body, -1, lateralFriction=1.2, rollingFriction=0.02, spinningFriction=0.02)
        self.object_metadata[int(body)] = {
            "id": int(body),
            "index": int(index),
            "type": object_type,
            "shape": shape,
            "color_name": color_name,
            "color_rgba": [float(value) for value in color],
            "initial_position": [float(x), float(y), float(z)],
        }
        return body

    def is_object_on_table(self, obj_id: int) -> bool:
        pos, _ = p.getBasePositionAndOrientation(obj_id)
        dx = abs(pos[0] - TABLE_CENTER[0])
        dy = abs(pos[1] - TABLE_CENTER[1])
        return (
            dx < TABLE_HALF_EXTENTS[0] + 0.05
            and dy < TABLE_HALF_EXTENTS[1] + 0.05
            and pos[2] > TABLE_TOP_Z - 0.05
        )

    def open_gripper(self) -> None:
        assert self.robot_id is not None
        for joint in FINGER_JOINTS:
            p.setJointMotorControl2(self.robot_id, joint, p.POSITION_CONTROL, 0.04, force=100)

    def close_gripper(self) -> None:
        assert self.robot_id is not None
        for joint in FINGER_JOINTS:
            p.setJointMotorControl2(self.robot_id, joint, p.POSITION_CONTROL, 0.0, force=200)

    def compute_ik(self, position, orientation) -> list[float]:
        assert self.robot_id is not None
        solution = p.calculateInverseKinematics(
            self.robot_id,
            EE_LINK_INDEX,
            position,
            orientation,
            lowerLimits=PANDA_LOWER_LIMITS,
            upperLimits=PANDA_UPPER_LIMITS,
            jointRanges=PANDA_JOINT_RANGES,
            restPoses=REST_JOINTS,
            maxNumIterations=200,
            residualThreshold=1e-5,
        )
        return list(solution[:7])

    def move_to_joints(self, target_joints, steps: int = 100) -> None:
        assert self.robot_id is not None
        steps = max(1, int(steps * self.motion_slowdown))
        start_joints = [p.getJointState(self.robot_id, joint)[0] for joint in ARM_JOINTS]
        for i in range(steps):
            t_linear = (i + 1) / steps
            t_smooth = (1.0 - np.cos(t_linear * np.pi)) / 2.0
            current = [s + (e - s) * t_smooth for s, e in zip(start_joints, target_joints)]
            p.setJointMotorControlArray(
                self.robot_id,
                ARM_JOINTS,
                p.POSITION_CONTROL,
                targetPositions=current,
                forces=[220] * len(ARM_JOINTS),
            )
            self.step_sim(1)

    def move_ee(self, position, orientation, steps: int = 80) -> None:
        self.move_to_joints(self.compute_ik(position, orientation), steps=steps)

    def grasp_candidates(self, obj_id: int):
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        _, _, yaw = p.getEulerFromQuaternion(orn)
        candidates = []
        for delta_yaw in (0.0, np.pi / 2.0, -np.pi / 2.0):
            target_orientation = p.getQuaternionFromEuler([np.pi, 0.0, yaw + delta_yaw])
            candidates.append((np.asarray(pos, dtype=np.float32), target_orientation))
        return candidates

    def pick_and_place(self, obj_id: int) -> dict:
        if not self.is_object_on_table(obj_id):
            return {"object_id": obj_id, "success": False, "reason": "object_not_on_table"}

        for attempt, (grasp_pos, grasp_orn) in enumerate(self.grasp_candidates(obj_id), start=1):
            pre_grasp = grasp_pos + np.array([0.0, 0.0, PRE_GRASP_HEIGHT], dtype=np.float32)
            self.open_gripper()
            self.move_ee(pre_grasp, grasp_orn, steps=150)

            if not self.is_object_on_table(obj_id):
                return {"object_id": obj_id, "success": False, "reason": "object_moved_during_approach"}

            current_pos, _ = p.getBasePositionAndOrientation(obj_id)
            grasp_pos = np.array([current_pos[0], current_pos[1], current_pos[2] + 0.005], dtype=np.float32)
            self.move_ee(grasp_pos, grasp_orn, steps=90)
            self.step_sim(20)
            self.close_gripper()
            self.step_sim(40)

            constraint_id = None
            if self.grasp_assist:
                assert self.robot_id is not None
                constraint_id = p.createConstraint(
                    self.robot_id,
                    EE_LINK_INDEX,
                    obj_id,
                    -1,
                    p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                )

            lift_pos = grasp_pos + np.array([0.0, 0.0, LIFT_HEIGHT], dtype=np.float32)
            self.move_ee(lift_pos, grasp_orn, steps=150)
            lifted_pos, _ = p.getBasePositionAndOrientation(obj_id)

            if lifted_pos[2] > MIN_GRASP_LIFT_Z:
                place_pos = BIN_CENTER + np.array([0.0, 0.0, 0.22], dtype=np.float32)
                self.move_ee(place_pos, grasp_orn, steps=200)
                if constraint_id is not None:
                    p.removeConstraint(constraint_id)
                self.open_gripper()
                self.step_sim(100)
                self.move_to_joints(HOME_JOINTS, steps=150)
                final_pos, _ = p.getBasePositionAndOrientation(obj_id)
                return {
                    "object_id": obj_id,
                    "success": True,
                    "attempt": attempt,
                    "lifted_z": float(lifted_pos[2]),
                    "final_position": list(map(float, final_pos)),
                }

            if constraint_id is not None:
                p.removeConstraint(constraint_id)
            self.open_gripper()
            self.move_ee(pre_grasp, grasp_orn, steps=80)

        return {"object_id": obj_id, "success": False, "reason": "all_grasp_candidates_failed"}

    def pick_and_place_with_graspnet_pose(
        self,
        obj_id: int,
        grasp_world: dict,
        *,
        use_grasp_orientation: bool = False,
    ) -> dict:
        if not self.is_object_on_table(obj_id):
            return {"object_id": obj_id, "success": False, "reason": "object_not_on_table"}

        object_pos, object_orn = p.getBasePositionAndOrientation(obj_id)
        object_pos = np.asarray(object_pos, dtype=np.float32)
        raw_grasp_pos = np.asarray(grasp_world["translation"], dtype=np.float32)

        grasp_pos = raw_grasp_pos.copy()
        xy_offset = float(np.linalg.norm(grasp_pos[:2] - object_pos[:2]))
        adjustment = "none"
        if xy_offset > 0.08:
            grasp_pos[:2] = object_pos[:2]
            adjustment = "xy_clamped_to_selected_object_center"
        grasp_pos[2] = max(float(object_pos[2] + 0.005), float(TABLE_TOP_Z + 0.02))

        if use_grasp_orientation and "quaternion_xyzw" in grasp_world:
            grasp_orn = grasp_world["quaternion_xyzw"]
            orientation_source = "graspnet_transformed"
        else:
            _, _, object_yaw = p.getEulerFromQuaternion(object_orn)
            grasp_orn = p.getQuaternionFromEuler([np.pi, 0.0, object_yaw])
            orientation_source = "downward_stable_from_graspnet_position"

        pre_grasp = grasp_pos + np.array([0.0, 0.0, PRE_GRASP_HEIGHT], dtype=np.float32)
        self.open_gripper()
        self.move_ee(pre_grasp, grasp_orn, steps=150)

        if not self.is_object_on_table(obj_id):
            return {"object_id": obj_id, "success": False, "reason": "object_moved_during_approach"}

        self.move_ee(grasp_pos, grasp_orn, steps=90)
        self.step_sim(20)
        self.close_gripper()
        self.step_sim(40)

        constraint_id = None
        if self.grasp_assist:
            assert self.robot_id is not None
            constraint_id = p.createConstraint(
                self.robot_id,
                EE_LINK_INDEX,
                obj_id,
                -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            )

        lift_pos = grasp_pos + np.array([0.0, 0.0, LIFT_HEIGHT], dtype=np.float32)
        self.move_ee(lift_pos, grasp_orn, steps=150)
        lifted_pos, _ = p.getBasePositionAndOrientation(obj_id)

        if lifted_pos[2] > MIN_GRASP_LIFT_Z:
            place_pos = BIN_CENTER + np.array([0.0, 0.0, 0.22], dtype=np.float32)
            self.move_ee(place_pos, grasp_orn, steps=200)
            if constraint_id is not None:
                p.removeConstraint(constraint_id)
            self.open_gripper()
            self.step_sim(100)
            self.move_to_joints(HOME_JOINTS, steps=150)
            final_pos, _ = p.getBasePositionAndOrientation(obj_id)
            return {
                "object_id": obj_id,
                "success": True,
                "grasp_source": "graspnet_trained_checkpoint",
                "raw_graspnet_translation": raw_grasp_pos.astype(float).tolist(),
                "executed_grasp_translation": grasp_pos.astype(float).tolist(),
                "xy_offset_from_object_center_m": xy_offset,
                "pose_adjustment": adjustment,
                "orientation_source": orientation_source,
                "lifted_z": float(lifted_pos[2]),
                "final_position": list(map(float, final_pos)),
            }

        if constraint_id is not None:
            p.removeConstraint(constraint_id)
        self.open_gripper()
        self.move_ee(pre_grasp, grasp_orn, steps=80)
        return {
            "object_id": obj_id,
            "success": False,
            "reason": "graspnet_pose_lift_failed",
            "grasp_source": "graspnet_trained_checkpoint",
            "raw_graspnet_translation": raw_grasp_pos.astype(float).tolist(),
            "executed_grasp_translation": grasp_pos.astype(float).tolist(),
            "xy_offset_from_object_center_m": xy_offset,
            "pose_adjustment": adjustment,
            "orientation_source": orientation_source,
        }

    def run_all(self, max_objects: int | None = None) -> list[dict]:
        object_ids = self.object_ids[:]
        random.shuffle(object_ids)
        if max_objects is not None:
            object_ids = object_ids[:max_objects]
        results = []
        for obj_id in object_ids:
            print(f"Attempting object {obj_id}...")
            result = self.pick_and_place(obj_id)
            print(result)
            results.append(result)
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Panda pick-and-place simulation like the reference notebook.")
    parser.add_argument("--gui", action="store_true", help="Open PyBullet GUI.")
    parser.add_argument("--num-objects", type=int, default=12)
    parser.add_argument("--max-objects", type=int, default=None, help="Limit how many objects to attempt.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-grasp-assist", action="store_true")
    parser.add_argument("--realtime-sleep", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Motion slowdown multiplier. 2.0 is about twice slower.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--keep-open", action="store_true", help="Keep GUI open after the run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = PandaPickPlaceSim(
        gui=args.gui,
        realtime_sleep=args.realtime_sleep,
        seed=args.seed,
        grasp_assist=not args.no_grasp_assist,
        motion_slowdown=args.speed_scale,
    )
    sim.connect()
    try:
        sim.setup_scene(args.num_objects)
        results = sim.run_all(max_objects=args.max_objects)
        summary = {
            "status": "completed",
            "robot": "franka_panda/panda.urdf",
            "num_objects": args.num_objects,
            "attempted_objects": len(results),
            "success_count": sum(1 for item in results if item["success"]),
            "grasp_assist": not args.no_grasp_assist,
            "seed": args.seed,
            "results": results,
        }
        summary_path = output_dir / "panda_pick_place_result.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
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
