from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class IKResult:
    target_position: list[float]
    target_quaternion_xyzw: list[float]
    joint_positions: list[float]
    reached_position: list[float]
    reached_quaternion_xyzw: list[float]
    position_error_m: float
    num_steps: int
    ik_mode: str = "pose"


class PyBulletRobotController:
    def __init__(
        self,
        *,
        gui: bool = False,
        robot_urdf: str = "xarm/xarm6_robot.urdf",
        ee_link_index: int | None = None,
        fixed_base: bool = True,
    ) -> None:
        try:
            import pybullet as pybullet
            import pybullet_data as pybullet_data
        except ImportError as exc:
            raise ImportError(
                "PyBullet is not installed. On this Windows machine pip tried to "
                "build pybullet from source and failed because Microsoft Visual "
                "C++ Build Tools are missing."
            ) from exc

        self.p = pybullet
        self.pybullet_data = pybullet_data
        self.gui = gui
        self.robot_urdf = robot_urdf
        self.ee_link_index = ee_link_index
        self.fixed_base = fixed_base
        self.client_id: int | None = None
        self.robot_id: int | None = None
        self.arm_joint_indices: list[int] = []
        self.lower_limits: list[float] = []
        self.upper_limits: list[float] = []
        self.joint_ranges: list[float] = []
        self.rest_poses: list[float] = []

    def connect(self) -> None:
        mode = self.p.GUI if self.gui else self.p.DIRECT
        self.client_id = self.p.connect(mode)
        self.p.setAdditionalSearchPath(self.pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.81)
        if self.gui:
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
            self.p.resetDebugVisualizerCamera(
                cameraDistance=1.4,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[0.45, 0.0, 0.35],
            )

    def load_scene(self) -> None:
        self._require_connection()
        self.p.loadURDF("plane.urdf")
        self.p.loadURDF(
            "table/table.urdf",
            basePosition=[0.55, 0.0, -0.63],
            baseOrientation=self.p.getQuaternionFromEuler([0, 0, math.pi / 2]),
            useFixedBase=True,
        )

    def load_robot(self) -> None:
        self._require_connection()
        flags = self.p.URDF_USE_SELF_COLLISION
        self.robot_id = self.p.loadURDF(
            self.robot_urdf,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=self.fixed_base,
            flags=flags,
        )
        self._inspect_robot()

    def _inspect_robot(self) -> None:
        assert self.robot_id is not None
        self.arm_joint_indices.clear()
        self.lower_limits.clear()
        self.upper_limits.clear()
        self.joint_ranges.clear()
        self.rest_poses.clear()

        for joint_index in range(self.p.getNumJoints(self.robot_id)):
            info = self.p.getJointInfo(self.robot_id, joint_index)
            joint_type = info[2]
            if joint_type not in (self.p.JOINT_REVOLUTE, self.p.JOINT_PRISMATIC):
                continue
            lower = float(info[8])
            upper = float(info[9])
            if upper <= lower:
                lower, upper = -math.pi, math.pi
            self.arm_joint_indices.append(joint_index)
            self.lower_limits.append(lower)
            self.upper_limits.append(upper)
            self.joint_ranges.append(upper - lower)
            self.rest_poses.append(0.0)

        if not self.arm_joint_indices:
            raise RuntimeError(f"No movable joints found in {self.robot_urdf}")
        if self.ee_link_index is None:
            self.ee_link_index = self.arm_joint_indices[-1]

    def add_target_marker(self, position: list[float], rgba: list[float] | None = None) -> int:
        rgba = rgba or [1.0, 0.1, 0.1, 0.85]
        visual = self.p.createVisualShape(
            self.p.GEOM_SPHERE,
            radius=0.025,
            rgbaColor=rgba,
        )
        return self.p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=position,
        )

    def add_grasp_object(
        self,
        position: list[float],
        *,
        radius: float = 0.035,
        height: float = 0.12,
        mass: float = 0.05,
    ) -> int:
        collision = self.p.createCollisionShape(
            self.p.GEOM_CYLINDER,
            radius=radius,
            height=height,
        )
        visual = self.p.createVisualShape(
            self.p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0.95, 0.2, 0.1, 1.0],
        )
        return self.p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
        )

    def attach_body_to_ee(self, body_id: int) -> int:
        assert self.robot_id is not None
        assert self.ee_link_index is not None
        return self.p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_link_index,
            childBodyUniqueId=body_id,
            childLinkIndex=-1,
            jointType=self.p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )

    def detach_body(self, constraint_id: int) -> None:
        self.p.removeConstraint(constraint_id)

    def step(self, steps: int = 120) -> None:
        for _ in range(steps):
            self.p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

    def calculate_ik(
        self,
        target_position: list[float],
        target_quaternion_xyzw: list[float] | None,
        *,
        max_iterations: int = 200,
    ) -> list[float]:
        assert self.robot_id is not None
        assert self.ee_link_index is not None
        kwargs = {
            "bodyUniqueId": self.robot_id,
            "endEffectorLinkIndex": self.ee_link_index,
            "targetPosition": target_position,
            "lowerLimits": self.lower_limits,
            "upperLimits": self.upper_limits,
            "jointRanges": self.joint_ranges,
            "restPoses": self.rest_poses,
            "maxNumIterations": max_iterations,
            "residualThreshold": 1e-5,
        }
        if target_quaternion_xyzw is not None:
            kwargs["targetOrientation"] = target_quaternion_xyzw
        solution = self.p.calculateInverseKinematics(**kwargs)
        return [float(solution[i]) for i in range(len(self.arm_joint_indices))]

    def move_joints(
        self,
        joint_positions: list[float],
        *,
        steps: int = 240,
        sleep: bool = False,
    ) -> None:
        assert self.robot_id is not None
        self.p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joint_indices,
            self.p.POSITION_CONTROL,
            targetPositions=joint_positions,
            forces=[120.0] * len(self.arm_joint_indices),
        )
        for _ in range(steps):
            self.p.stepSimulation()
            if sleep and self.gui:
                time.sleep(1.0 / 240.0)

    def get_ee_pose(self) -> tuple[list[float], list[float]]:
        assert self.robot_id is not None
        assert self.ee_link_index is not None
        state = self.p.getLinkState(self.robot_id, self.ee_link_index)
        return list(state[4]), list(state[5])

    def execute_pose(
        self,
        target_position: list[float],
        target_quaternion_xyzw: list[float],
        *,
        steps: int = 240,
        position_only_fallback: bool = True,
        fallback_error_threshold: float = 0.03,
    ) -> IKResult:
        joint_positions = self.calculate_ik(target_position, target_quaternion_xyzw)
        self.move_joints(joint_positions, steps=steps, sleep=self.gui)
        reached_position, reached_quat = self.get_ee_pose()
        error = float(np.linalg.norm(np.asarray(reached_position) - np.asarray(target_position)))
        ik_mode = "pose"

        if position_only_fallback and error > fallback_error_threshold:
            fallback_joints = self.calculate_ik(target_position, None)
            self.move_joints(fallback_joints, steps=steps, sleep=self.gui)
            fallback_position, fallback_quat = self.get_ee_pose()
            fallback_error = float(np.linalg.norm(np.asarray(fallback_position) - np.asarray(target_position)))
            if fallback_error < error:
                joint_positions = fallback_joints
                reached_position = fallback_position
                reached_quat = fallback_quat
                error = fallback_error
                ik_mode = "position_only_fallback"

        return IKResult(
            target_position=list(map(float, target_position)),
            target_quaternion_xyzw=list(map(float, target_quaternion_xyzw)),
            joint_positions=joint_positions,
            reached_position=list(map(float, reached_position)),
            reached_quaternion_xyzw=list(map(float, reached_quat)),
            position_error_m=error,
            num_steps=steps,
            ik_mode=ik_mode,
        )

    def disconnect(self) -> None:
        if self.client_id is not None:
            self.p.disconnect(self.client_id)
            self.client_id = None

    def _require_connection(self) -> None:
        if self.client_id is None:
            raise RuntimeError("Call connect() before using the PyBullet controller.")


def result_to_dict(result: IKResult) -> dict[str, Any]:
    return {
        "target_position": result.target_position,
        "target_quaternion_xyzw": result.target_quaternion_xyzw,
        "joint_positions": result.joint_positions,
        "reached_position": result.reached_position,
        "reached_quaternion_xyzw": result.reached_quaternion_xyzw,
        "position_error_m": result.position_error_m,
        "num_steps": result.num_steps,
        "ik_mode": result.ik_mode,
    }
