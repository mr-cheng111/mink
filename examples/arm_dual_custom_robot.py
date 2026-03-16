#!/usr/bin/env python3
"""
Dual-arm waypoint planner template (single shared Mink IK solver),
参考 arm_dual_panda.py，适配你自己的双臂模型。

你可以自己决定模型使用方式：
1) 直接加载 XML（--model-source xml --xml /abs/path/to/scene.xml）
2) 使用 ARM620 双臂构建器（--model-source arm620-builder）

示例：
  python mink/examples/arm_dual_custom_robot.py \
    --model-source arm620-builder \
    --left-site left/tools_link --right-site right/tools_link \
    --left-target left_target --right-target right_target

  python mink/examples/arm_dual_custom_robot.py \
    --model-source xml --xml /path/to/your_dual_scene.xml \
    --left-site left_ee_site --right-site right_ee_site \
    --left-target left_target --right-target right_target
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import mink

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-arm shared-IK waypoint planner")
    parser.add_argument("--model-source", choices=["xml", "arm620-builder"], default="arm620-builder")
    parser.add_argument("--xml", type=str, default="", help="Absolute path to your dual-arm XML when model-source=xml")
    parser.add_argument("--y-offset", type=float, default=0.28, help="Used only by arm620-builder")

    parser.add_argument("--left-site", type=str, required=True, help="Left arm end-effector site name")
    parser.add_argument("--right-site", type=str, required=True, help="Right arm end-effector site name")
    parser.add_argument("--left-target", type=str, required=True, help="Left mocap body name")
    parser.add_argument("--right-target", type=str, required=True, help="Right mocap body name")

    parser.add_argument("--left-prefix", type=str, default="left/", help="Prefix to auto-pick left joints")
    parser.add_argument("--right-prefix", type=str, default="right/", help="Prefix to auto-pick right joints")
    parser.add_argument("--left-joints", type=str, default="", help="Comma-separated explicit left joint names")
    parser.add_argument("--right-joints", type=str, default="", help="Comma-separated explicit right joint names")
    parser.add_argument("--num-joints", type=int, default=6, help="Joint count per arm")

    parser.add_argument("--ik-iters", type=int, default=20)
    parser.add_argument("--ik-damping", type=float, default=1e-3)
    parser.add_argument("--pos-threshold", type=float, default=3e-3)
    parser.add_argument("--orientation-cost", type=float, default=0.0)
    parser.add_argument("--frequency", type=float, default=120.0)
    parser.add_argument("--solver", type=str, default="daqp")
    return parser.parse_args()


def load_model_data(args: argparse.Namespace) -> tuple[mujoco.MjModel, mujoco.MjData]:
    if args.model_source == "xml":
        if not args.xml:
            raise ValueError("--model-source xml 时必须提供 --xml")
        model = mujoco.MjModel.from_xml_path(Path(args.xml).as_posix())
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    from arm620.scene_dual_builder import build_dual_arm620_data

    return build_dual_arm620_data(y_offset=args.y_offset)


def _joint_sort_key(name: str) -> tuple[int, str]:
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)) if m else 10**9, name)


def resolve_joint_names(model: mujoco.MjModel, prefix: str, num_joints: int) -> list[str]:
    names: list[str] = []
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname and jname.startswith(prefix):
            names.append(jname)

    names.sort(key=_joint_sort_key)
    if len(names) < num_joints:
        raise ValueError(f"前缀 {prefix} 只找到 {len(names)} 个关节，不足 {num_joints} 个")
    return names[:num_joints]


def to_qpos_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    return np.array([model.joint(name).qposadr[0] for name in joint_names], dtype=int)


def quaternion_error(q_current: np.ndarray, q_target: np.ndarray) -> float:
    return min(np.linalg.norm(q_current - q_target), np.linalg.norm(q_current + q_target))


def update_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_target_name: str,
    right_target_name: str,
    left_pos: np.ndarray,
    left_quat: np.ndarray,
    right_pos: np.ndarray,
    right_quat: np.ndarray,
) -> None:
    data.mocap_pos[model.body(left_target_name).mocapid] = left_pos
    data.mocap_quat[model.body(left_target_name).mocapid] = left_quat
    data.mocap_pos[model.body(right_target_name).mocapid] = right_pos
    data.mocap_quat[model.body(right_target_name).mocapid] = right_quat


def define_waypoints(
    data: mujoco.MjData,
    left_site_id: int,
    right_site_id: int,
    ql: np.ndarray,
    qr: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    left_start = data.site_xpos[left_site_id].copy()
    right_start = data.site_xpos[right_site_id].copy()

    left_mid = left_start.copy()
    left_mid[0] += 0.05
    left_mid[1] -= 0.06

    right_mid = right_start.copy()
    right_mid[0] += 0.05
    right_mid[1] += 0.06

    left_lift = left_mid.copy()
    right_lift = right_mid.copy()
    left_lift[2] += 0.12
    right_lift[2] += 0.12

    left_waypoints = [(left_start, ql.copy()), (left_mid, ql.copy())]
    right_waypoints = [(right_start, qr.copy()), (right_mid, qr.copy())]
    return left_waypoints, right_waypoints, left_lift, right_lift


def reached(
    data: mujoco.MjData,
    left_site_id: int,
    right_site_id: int,
    left_target_pos: np.ndarray,
    right_target_pos: np.ndarray,
    left_target_quat: np.ndarray,
    right_target_quat: np.ndarray,
    pos_threshold: float,
) -> bool:
    left_pos_err = np.linalg.norm(data.site_xpos[left_site_id] - left_target_pos)
    right_pos_err = np.linalg.norm(data.site_xpos[right_site_id] - right_target_pos)

    ql = np.empty(4)
    qr = np.empty(4)
    mujoco.mju_mat2Quat(ql, data.site_xmat[left_site_id])
    mujoco.mju_mat2Quat(qr, data.site_xmat[right_site_id])

    left_ori_err = quaternion_error(ql, left_target_quat)
    right_ori_err = quaternion_error(qr, right_target_quat)

    return (
        left_pos_err <= pos_threshold
        and right_pos_err <= pos_threshold
        and left_ori_err <= 0.02
        and right_ori_err <= 0.02
    )


def main() -> None:
    args = parse_args()

    model, data = load_model_data(args)
    configuration = mink.Configuration(model)

    left_task = mink.FrameTask(
        frame_name=args.left_site,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=args.orientation_cost,
        lm_damping=1.0,
    )
    right_task = mink.FrameTask(
        frame_name=args.right_site,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=args.orientation_cost,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [left_task, right_task, posture_task]

    left_joint_names = [s.strip() for s in args.left_joints.split(",") if s.strip()] if args.left_joints else resolve_joint_names(model, args.left_prefix, args.num_joints)
    right_joint_names = [s.strip() for s in args.right_joints.split(",") if s.strip()] if args.right_joints else resolve_joint_names(model, args.right_prefix, args.num_joints)

    controlled_qidx = np.concatenate([
        to_qpos_indices(model, left_joint_names),
        to_qpos_indices(model, right_joint_names),
    ])

    left_site_id = model.site(args.left_site).id
    right_site_id = model.site(args.right_site).id

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        mink.move_mocap_to_frame(model, data, args.left_target, args.left_site, "site")
        mink.move_mocap_to_frame(model, data, args.right_target, args.right_site, "site")

        ql = np.empty(4)
        qr = np.empty(4)
        mujoco.mju_mat2Quat(ql, data.site_xmat[left_site_id])
        mujoco.mju_mat2Quat(qr, data.site_xmat[right_site_id])

        left_waypoints, right_waypoints, left_lift, right_lift = define_waypoints(
            data,
            left_site_id,
            right_site_id,
            ql,
            qr,
        )

        rate = RateLimiter(frequency=args.frequency, warn=False)
        print("\nDual-arm planner started with shared IK solver")
        print(f"solver={args.solver} model_source={args.model_source}")
        print(f"left joints={left_joint_names}")
        print(f"right joints={right_joint_names}")

        for i, (lwp, rwp) in enumerate(zip(left_waypoints, right_waypoints), start=1):
            print(f"Planning to waypoint {i}...")
            left_pos, left_quat = lwp
            right_pos, right_quat = rwp
            update_targets(
                model,
                data,
                args.left_target,
                args.right_target,
                left_pos,
                left_quat,
                right_pos,
                right_quat,
            )

            ok = False
            for _ in range(1200):
                if not viewer.is_running():
                    return

                configuration.update(data.qpos)
                left_task.set_target(mink.SE3.from_mocap_name(model, data, args.left_target))
                right_task.set_target(mink.SE3.from_mocap_name(model, data, args.right_target))

                for _ in range(args.ik_iters):
                    vel = mink.solve_ik(
                        configuration,
                        tasks,
                        rate.dt,
                        args.solver,
                        damping=args.ik_damping,
                    )
                    configuration.integrate_inplace(vel, rate.dt)

                # 通用模型策略：直接把 IK 结果写回 qpos（不依赖 actuator 命名）
                data.qpos[controlled_qidx] = configuration.q[controlled_qidx]
                mujoco.mj_forward(model, data)

                viewer.sync()
                rate.sleep()

                if reached(
                    data,
                    left_site_id,
                    right_site_id,
                    left_pos,
                    right_pos,
                    left_quat,
                    right_quat,
                    args.pos_threshold,
                ):
                    ok = True
                    break

            print(f"Waypoint {i} {'reached' if ok else 'timeout'}")

        print("Lifting both arms...")
        update_targets(
            model,
            data,
            args.left_target,
            args.right_target,
            left_lift,
            ql,
            right_lift,
            qr,
        )

        while viewer.is_running():
            configuration.update(data.qpos)
            left_task.set_target(mink.SE3.from_mocap_name(model, data, args.left_target))
            right_task.set_target(mink.SE3.from_mocap_name(model, data, args.right_target))

            vel = mink.solve_ik(
                configuration,
                tasks,
                rate.dt,
                args.solver,
                damping=args.ik_damping,
            )
            configuration.integrate_inplace(vel, rate.dt)

            data.qpos[controlled_qidx] = configuration.q[controlled_qidx]
            mujoco.mj_forward(model, data)

            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
