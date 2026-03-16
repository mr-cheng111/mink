#!/usr/bin/env python3
"""
ARM620 Mink IK + PID Torque Control Example
使用 mink 进行逆运动学求解，然后通过 PID 控制器输出力矩
"""

from pathlib import Path
import sys
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

# 添加examples目录到Python路径，以便导入本地模块
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

# 使用本地的简化PID控制器
from simple_pid_controller import SimplePIDController

import mink
_XML = _HERE / "arm620" / "scene.xml"

SOLVER = "daqp"
POS_THRESHOLD = 1e-4
MAX_ITERS = 20


def converge_ik(configuration, tasks, dt, solver, pos_threshold, max_iters):
    """运行 IK 求解直到收敛"""
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, damping=1e-3)
        configuration.integrate_inplace(vel, dt)
        
        err = tasks[0].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        
        if pos_achieved:
            return True
    return False


def main():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    
    # Mink IK 配置
    configuration = mink.Configuration(model)
    
    end_effector_task = mink.FrameTask(
        frame_name="tools_link",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = [end_effector_task, posture_task]
    
    # 初始化PID控制器（使用原项目的PD参数，I=0）
    urdf_path = _HERE / "arm620" / "urdf" / "arm620.urdf"
    
    pid_controller = SimplePIDController(
        kp=np.array([180, 200, 200, 65, 85, 25]),  # 原项目默认Kp
        kd=np.array([10, 30, 10, 1.5, 1, 1]),        # 原项目默认Kd
        ki=np.array([0, 0, 0, 0, 0, 0]),              # Ki=0
        n_joints=6,
        torque_limits=np.array([49.0, 49.0, 49.0, 9.0, 9.0, 9.0]),
        urdf_path=str(urdf_path),
        use_gravity_compensation=True
    )
    print("⚠ 注意：使用原项目PD参数 (Kp=500, Kd=30, Ki=0)")
    
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=True  # 显示右侧UI可以看到mocap控制
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        # 初始化到零位（包括夹爪关节）
        data.qpos[:] = 0  # 初始化所有关节
        mujoco.mj_forward(model, data)
        
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        
        # 移动 mocap target 到末端初始位置 (使用 tools_link)
        mink.move_mocap_to_frame(model, data, "target", "tools_link", "site")
        initial_target_position = data.mocap_pos[0].copy()
        
        print(f"初始目标位置: [{initial_target_position[0]:.3f}, {initial_target_position[1]:.3f}, {initial_target_position[2]:.3f}]")
        
        # 绕Z轴旋转轨迹参数
        radius = np.linalg.norm(initial_target_position[:2])  # 计算到Z轴的距离
        initial_angle = np.arctan2(initial_target_position[1], initial_target_position[0])  # 初始角度
        z_height = initial_target_position[2]  # 保持Z高度不变
        
        angular_speed = 0.3  # rad/s, 旋转角速度
        
        local_time = 0.0
        rate = RateLimiter(frequency=1000.0, warn=False)  # 1000Hz控制频率
        
        # 使用MuJoCo的固定timestep而不是实际控制周期
        mujoco_dt = model.opt.timestep
        
        print(f"旋转半径: {radius:.3f}m, 初始角度: {np.rad2deg(initial_angle):.1f}°")
        print(f"MuJoCo timestep: {mujoco_dt*1000:.1f}ms, 控制频率: 1000Hz")
        print("\n🤖 开始运行 - 使用 Mink IK + PID 力矩控制")
        print("末端执行器将绕Z轴旋转运动")
        print("\n💡 拖动控制方法:")
        print("  1. 双击红色方块(target)可以用鼠标拖动")
        print("  2. 按住 Ctrl + 鼠标左键 也可以拖动选中的物体")
        print("  3. 按 'Space' 暂停/继续自动轨迹")
        print("  4. 每5秒输出一次力矩分解信息")
        
        # 自动轨迹开关
        auto_trajectory = False  # 默认关闭自动轨迹，允许手动控制
        
        # 调试计数器
        debug_counter = 0
        debug_interval = 5000  # 每5000次循环（5秒）输出一次调试信息
        
        while viewer.is_running():
            dt = rate.dt
            local_time += dt
            
            # 只在自动模式下更新轨迹
            if auto_trajectory:
                # 更新目标位置（绕Z轴旋转）
                current_angle = initial_angle + angular_speed * local_time
                
                # 计算旋转后的XY位置，保持Z高度不变
                target_x = radius * np.cos(current_angle)
                target_y = radius * np.sin(current_angle)
                target_z = z_height
                
                data.mocap_pos[0] = np.array([target_x, target_y, target_z])
            # 否则使用当前mocap位置（允许手动拖动）
            
            # Mink IK 求解目标关节位置
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)
            
            converge_ik(configuration, tasks, dt, SOLVER, POS_THRESHOLD, MAX_ITERS)
            
            # 获取IK求解的目标关节位置
            q_target = configuration.q[:6].copy()
            
            # 获取当前关节状态
            q_current = data.qpos[:6].copy()
            dq_current = data.qvel[:6].copy()
            
            # PID控制器计算力矩（每5秒输出一次调试信息）
            debug_counter += 1
            show_debug = (debug_counter % debug_interval == 0)
            
            tau = pid_controller.compute_torque(
                q=q_current,
                dq=dq_current,
                q_target=q_target,
                dq_target=np.zeros(6),
                dt=mujoco_dt,  # 使用MuJoCo的固定timestep，而不是实际控制周期
                debug=show_debug
            )
            
            # 应用力矩到执行器
            data.ctrl[:6] = tau
            
            # 夹爪控制（如果需要）
            if model.nu > 6:
                data.ctrl[6] = 0.0  # Robotiq gripper actuator
            
            # 执行一步仿真
            mujoco.mj_step(model, data)
            
            # 可视化同步
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ 用户中断")
