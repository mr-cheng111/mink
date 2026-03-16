#!/usr/bin/env python3
"""
ARM620 control example using mink IK solver.
Based on mink's arm_iiwa.py and arm_panda.py examples.
"""

from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from simple_pid_controller import SimplePIDController

_XML = _HERE / "arm620" / "scene.xml"

SOLVER = "daqp"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 20


def converge_ik(
    configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters
):
    """Runs up to 'max_iters' of IK steps. Returns True if position and orientation
    are below thresholds, otherwise False."""
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, damping=1e-3)
        configuration.integrate_inplace(vel, dt)
        
        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
        
        if pos_achieved and ori_achieved:
            return True
    return False


def main():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="tools_link",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]

    urdf_path = _HERE / "arm620" / "urdf" / "arm620.urdf"
    pid_controller = SimplePIDController(
        kp=np.array([180, 200, 200, 65, 85, 25]),
        kd=np.array([10, 30, 10, 1.5, 1, 1]),
        ki=np.zeros(6),
        n_joints=6,
        torque_limits=np.array([49.0, 49.0, 49.0, 9.0, 9.0, 9.0]),
        urdf_path=str(urdf_path),
        use_gravity_compensation=True,
    )

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=True
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        try:
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        except:
            print("No 'home' keyframe found, using zero position")
            data.qpos[:] = 0

        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        mink.move_mocap_to_frame(model, data, "target", "tools_link", "site")
        print("Drag the target in the viewer to move the arm manually.")

        rate = RateLimiter(frequency=200.0, warn=False)
        mujoco_dt = model.opt.timestep

        while viewer.is_running():
            dt = rate.dt

            configuration.update(data.qpos)

            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            converge_ik(
                configuration,
                tasks,
                dt,
                SOLVER,
                POS_THRESHOLD,
                ORI_THRESHOLD,
                MAX_ITERS,
            )

            q_target = configuration.q[:6].copy()
            q_current = data.qpos[:6].copy()
            dq_current = data.qvel[:6].copy()

            tau = pid_controller.compute_torque(
                q=q_current,
                dq=dq_current,
                q_target=q_target,
                dq_target=np.zeros(6),
                dt=mujoco_dt,
                debug=False,
            )

            data.ctrl[:6] = tau
            if model.nu > 6:
                data.ctrl[6] = 0.0

            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
