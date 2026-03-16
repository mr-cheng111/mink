#!/usr/bin/env python3
"""
Dual ARM620 simulation example:
- one shared Mink planner for both arms
- two independent PID torque controllers (left/right)
"""

from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

try:
    from loop_rate_limiters import RateLimiter
except ImportError:
    class _FallbackRateLimiter:
        def __init__(self, frequency: float, warn: bool = False):
            self.frequency = frequency
            self.period = 1.0 / frequency
            self.dt = self.period
            self.warn = warn
            self._last = time.perf_counter()

        def sleep(self) -> None:
            now = time.perf_counter()
            elapsed = now - self._last
            remaining = self.period - elapsed
            if remaining > 0.0:
                time.sleep(remaining)
                now = time.perf_counter()
            self.dt = now - self._last
            self._last = now

    RateLimiter = _FallbackRateLimiter

import mink

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from simple_pid_controller import SimplePIDController
from arm620.scene_dual_unified_base_builder import build_dual_unified_data


SOLVER = "daqp"
MAX_ITERS = 20
CONTROL_HZ = 200.0

LEFT_ARM_JOINT_NAMES = [
    "left/joint1",
    "left/joint2",
    "left/joint3",
    "left/joint4",
    "left/joint5",
    "left/joint6",
]
RIGHT_ARM_JOINT_NAMES = [
    "right/joint1",
    "right/joint2",
    "right/joint3",
    "right/joint4",
    "right/joint5",
    "right/joint6",
]

LEFT_ARM_ACTUATOR_NAMES = [
    "left/joint1_motor",
    "left/joint2_motor",
    "left/joint3_motor",
    "left/joint4_motor",
    "left/joint5_motor",
    "left/joint6_motor",
]
RIGHT_ARM_ACTUATOR_NAMES = [
    "right/joint1_motor",
    "right/joint2_motor",
    "right/joint3_motor",
    "right/joint4_motor",
    "right/joint5_motor",
    "right/joint6_motor",
]


def joint_qpos_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    return np.array([model.joint(name).qposadr[0] for name in joint_names], dtype=int)


def joint_dof_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    return np.array([model.joint(name).dofadr[0] for name in joint_names], dtype=int)


def actuator_indices(model: mujoco.MjModel, actuator_names: list[str]) -> np.ndarray:
    return np.array([model.actuator(name).id for name in actuator_names], dtype=int)


class DualArmMinkPlanner:
    """Single Mink planner that solves both arms jointly."""

    def __init__(self, model: mujoco.MjModel, solver: str = SOLVER):
        self.configuration = mink.Configuration(model)
        self.solver = solver

        self.left_ee_task = mink.FrameTask(
            frame_name="left/tools_link",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        self.right_ee_task = mink.FrameTask(
            frame_name="right/tools_link",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model=model, cost=1e-2)
        self.tasks = [self.left_ee_task, self.right_ee_task, self.posture_task]

    def initialize(self, qpos: np.ndarray) -> None:
        self.configuration.update(qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

    def update_from_qpos(self, qpos: np.ndarray) -> None:
        self.configuration.update(qpos)

    def set_targets_from_mocap(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.left_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left_target"))
        self.right_ee_task.set_target(
            mink.SE3.from_mocap_name(model, data, "right_target")
        )

    def solve(self, dt: float, max_iters: int = 1) -> None:
        for _ in range(max_iters):
            vel = mink.solve_ik(
                self.configuration, self.tasks, dt, self.solver, damping=1e-3
            )
            self.configuration.integrate_inplace(vel, dt)

    def get_joint_targets(
        self, left_qidx: np.ndarray, right_qidx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.configuration.q[left_qidx], self.configuration.q[right_qidx]


def main() -> None:
    model, data = build_dual_unified_data(y_offset=0.28)
    planner = DualArmMinkPlanner(model)

    l_qidx = joint_qpos_indices(model, LEFT_ARM_JOINT_NAMES)
    r_qidx = joint_qpos_indices(model, RIGHT_ARM_JOINT_NAMES)
    l_didx = joint_dof_indices(model, LEFT_ARM_JOINT_NAMES)
    r_didx = joint_dof_indices(model, RIGHT_ARM_JOINT_NAMES)
    l_aidx = actuator_indices(model, LEFT_ARM_ACTUATOR_NAMES)
    r_aidx = actuator_indices(model, RIGHT_ARM_ACTUATOR_NAMES)

    urdf_path = _HERE / "arm620" / "urdf" / "arm620.urdf"

    left_pid = SimplePIDController(
        kp=np.array([180, 200, 200, 65, 85, 25]),
        kd=np.array([10, 30, 10, 1.5, 1, 1]),
        ki=np.zeros(6),
        n_joints=6,
        torque_limits=np.array([49.0, 49.0, 49.0, 9.0, 9.0, 9.0]),
        urdf_path=str(urdf_path),
        use_gravity_compensation=True,
    )
    right_pid = SimplePIDController(
        kp=np.array([165, 190, 210, 70, 80, 28]),
        kd=np.array([12, 28, 12, 1.7, 1.2, 1.1]),
        ki=np.zeros(6),
        n_joints=6,
        torque_limits=np.array([49.0, 49.0, 49.0, 9.0, 9.0, 9.0]),
        urdf_path=str(urdf_path),
        use_gravity_compensation=True,
    )

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        planner.initialize(data.qpos)

        mink.move_mocap_to_frame(model, data, "left_target", "left/tools_link", "site")
        mink.move_mocap_to_frame(model, data, "right_target", "right/tools_link", "site")

        print("\nDual ARM620 IK + PID torque control started")
        print("Left and right arms use a shared Mink planner and independent PID controllers.")
        print("Both arms are mounted under one shared base_link.")
        print("Drag the red/blue mocap targets in the viewer to move each arm manually.")

        rate = RateLimiter(frequency=CONTROL_HZ, warn=False)
        control_dt = 1.0 / CONTROL_HZ
        sim_substeps = max(1, int(round(control_dt / model.opt.timestep)))

        print(
            f"Control: {CONTROL_HZ:.0f}Hz, model dt: {model.opt.timestep:.4f}s, "
            f"sim substeps/cycle: {sim_substeps}"
        )

        while viewer.is_running():
            dt = rate.dt

            planner.update_from_qpos(data.qpos)
            planner.set_targets_from_mocap(model, data)
            planner.solve(dt, max_iters=MAX_ITERS)

            # Joint targets planned together by one Mink planner.
            l_q_target, r_q_target = planner.get_joint_targets(l_qidx, r_qidx)

            # Measured joint states (left/right independent).
            l_q = data.qpos[l_qidx]
            r_q = data.qpos[r_qidx]
            l_dq = data.qvel[l_didx]
            r_dq = data.qvel[r_didx]

            l_tau = left_pid.compute_torque(
                q=l_q,
                dq=l_dq,
                q_target=l_q_target,
                dq_target=np.zeros(6),
                dt=control_dt,
                debug=False,
            )
            r_tau = right_pid.compute_torque(
                q=r_q,
                dq=r_dq,
                q_target=r_q_target,
                dq_target=np.zeros(6),
                dt=control_dt,
                debug=False,
            )

            data.ctrl[l_aidx] = l_tau
            data.ctrl[r_aidx] = r_tau

            # Keep both gripper actuators fixed.
            data.ctrl[model.actuator("left/robotiq_2f85_v4_actuator").id] = 0.0
            data.ctrl[model.actuator("right/robotiq_2f85_v4_actuator").id] = 0.0

            for _ in range(sim_substeps):
                mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
