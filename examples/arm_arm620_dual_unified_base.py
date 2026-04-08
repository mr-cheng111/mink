#!/usr/bin/env python3
"""Dual ARM620 robot with shared base at (0, 0, 0).

- One shared root body (`base_link`) in world origin
- Shared Mink IK solve for both arms
- Drag `left_target` / `right_target` in viewer
"""

import time
from pathlib import Path
import sys

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
IK_DAMPING = 1e-3
IK_MAX_ITERS = 20
POSITION_COST = 1.0
ORIENTATION_COST = 1.0
CONTROL_HZ = 200.0
LEFT_JOINTS = [
    "left/joint1",
    "left/joint2",
    "left/joint3",
    "left/joint4",
    "left/joint5",
    "left/joint6",
]
RIGHT_JOINTS = [
    "right/joint1",
    "right/joint2",
    "right/joint3",
    "right/joint4",
    "right/joint5",
    "right/joint6",
]
LEFT_ACTUATORS = [
    "left/joint1_motor",
    "left/joint2_motor",
    "left/joint3_motor",
    "left/joint4_motor",
    "left/joint5_motor",
    "left/joint6_motor",
]
RIGHT_ACTUATORS = [
    "right/joint1_motor",
    "right/joint2_motor",
    "right/joint3_motor",
    "right/joint4_motor",
    "right/joint5_motor",
    "right/joint6_motor",
]


def qpos_indices(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    return np.array([model.joint(n).qposadr[0] for n in names], dtype=int)


def dof_indices(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    return np.array([model.joint(n).dofadr[0] for n in names], dtype=int)


def actuator_indices(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    return np.array([model.actuator(n).id for n in names], dtype=int)


def assert_shared_base_link(model: mujoco.MjModel) -> None:
    """Ensure both arm bases are rigidly mounted under the same shared base_link."""
    shared_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    left_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left/base")
    right_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right/base")

    if min(shared_bid, left_bid, right_bid) < 0:
        raise RuntimeError(
            "Unified-base check failed: expected body names "
            "'base_link', 'left/base', 'right/base' were not found."
        )

    left_parent = int(model.body_parentid[left_bid])
    right_parent = int(model.body_parentid[right_bid])

    if left_parent != shared_bid or right_parent != shared_bid:
        left_parent_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, left_parent
        )
        right_parent_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, right_parent
        )
        raise RuntimeError(
            "Unified-base check failed: left/right arm base is not attached to "
            f"shared base_link. left parent={left_parent_name}, "
            f"right parent={right_parent_name}"
        )


def main() -> None:
    model, data = build_dual_unified_data(y_offset=0.28)
    assert_shared_base_link(model)
    configuration = mink.Configuration(model)

    left_task = mink.FrameTask(
        frame_name="left/tools_link",
        frame_type="site",
        position_cost=POSITION_COST,
        orientation_cost=ORIENTATION_COST,
        lm_damping=1.0,
    )
    right_task = mink.FrameTask(
        frame_name="right/tools_link",
        frame_type="site",
        position_cost=POSITION_COST,
        orientation_cost=ORIENTATION_COST,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    kinetic_energy_task = mink.KineticEnergyRegularizationTask(cost=1.0)
    tasks = [left_task, right_task, posture_task, kinetic_energy_task]

    l_qidx = qpos_indices(model, LEFT_JOINTS)
    r_qidx = qpos_indices(model, RIGHT_JOINTS)
    l_didx = dof_indices(model, LEFT_JOINTS)
    r_didx = dof_indices(model, RIGHT_JOINTS)
    l_aidx = actuator_indices(model, LEFT_ACTUATORS)
    r_aidx = actuator_indices(model, RIGHT_ACTUATORS)

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
        kp=np.array([180, 200, 200, 65, 85, 25]),
        kd=np.array([10, 30, 10, 1.5, 1, 1]),
        ki=np.zeros(6),
        n_joints=6,
        torque_limits=np.array([49.0, 49.0, 49.0, 9.0, 9.0, 9.0]),
        urdf_path=str(urdf_path),
        use_gravity_compensation=True,
    )

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # Unified robot starts at world origin.
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        # Initial pose: joint2=30deg forward, joint3=60deg forward.
        _init_q = np.zeros(6)
        _init_q[1] = np.deg2rad(-30)   # joint2
        _init_q[2] = np.deg2rad(-60)   # joint3
        data.qpos[l_qidx] = _init_q
        data.qpos[r_qidx] = _init_q
        mujoco.mj_forward(model, data)

        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        mink.move_mocap_to_frame(model, data, "left_target", "left/tools_link", "site")
        mink.move_mocap_to_frame(model, data, "right_target", "right/tools_link", "site")

        print("\nDual robot created with shared base_link at (0, 0, 0)")
        print("Both arm base_link nodes are rigidly connected to the same shared base_link.")
        print("Shared base_link frame marker is visible (X=red, Y=green, Z=blue).")
        print("Drag left_target/right_target to teleoperate both arms (pose tracking enabled).")

        rate = RateLimiter(frequency=CONTROL_HZ, warn=False)
        control_dt = 1.0 / CONTROL_HZ
        kinetic_energy_task.set_dt(self.control_dt)  # NOTE: This is required!
        sim_substeps = max(1, int(round(control_dt / model.opt.timestep)))
        print(
            f"Control: {CONTROL_HZ:.0f}Hz, model dt: {model.opt.timestep:.4f}s, "
            f"sim substeps/cycle: {sim_substeps}"
        )

        step_count = 0
        while viewer.is_running():
            configuration.update(data.qpos)

            left_task.set_target(mink.SE3.from_mocap_name(model, data, "left_target"))
            right_task.set_target(mink.SE3.from_mocap_name(model, data, "right_target"))

            for _ in range(IK_MAX_ITERS):
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, SOLVER, damping=IK_DAMPING
                )
                configuration.integrate_inplace(vel, rate.dt)

            l_q_target = configuration.q[l_qidx]
            r_q_target = configuration.q[r_qidx]
            l_q = data.qpos[l_qidx]
            r_q = data.qpos[r_qidx]
            l_dq = data.qvel[l_didx]
            r_dq = data.qvel[r_didx]

            l_tau = left_pid.compute_torque(q=l_q, dq=l_dq, q_target=l_q_target, dq_target=np.zeros(6), dt=control_dt, debug=False)
            r_tau = right_pid.compute_torque(q=r_q, dq=r_dq, q_target=r_q_target, dq_target=np.zeros(6), dt=control_dt, debug=False)

            data.ctrl[l_aidx] = l_tau
            data.ctrl[r_aidx] = r_tau
            data.ctrl[model.actuator("left/robotiq_2f85_v4_actuator").id] = 0.0
            data.ctrl[model.actuator("right/robotiq_2f85_v4_actuator").id] = 0.0

            for _ in range(sim_substeps):
                mujoco.mj_step(model, data)

            step_count += 1
            if step_count % 200 == 0:
                l_q_deg  = np.rad2deg(l_q)
                r_q_deg  = np.rad2deg(r_q)
                l_dq_deg = np.rad2deg(l_dq)
                r_dq_deg = np.rad2deg(r_dq)
                print("\n--- Joint Data (step {:6d}) ---".format(step_count))
                print("LEFT  pos(deg): " + " ".join(f"{v:7.2f}" for v in l_q_deg))
                print("LEFT  vel(d/s): " + " ".join(f"{v:7.2f}" for v in l_dq_deg))
                print("LEFT  tau(N·m): " + " ".join(f"{v:7.2f}" for v in l_tau))
                print("RIGHT pos(deg): " + " ".join(f"{v:7.2f}" for v in r_q_deg))
                print("RIGHT vel(d/s): " + " ".join(f"{v:7.2f}" for v in r_dq_deg))
                print("RIGHT tau(N·m): " + " ".join(f"{v:7.2f}" for v in r_tau))

            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
