"""
简化的PID力矩控制器 + 重力补偿
不依赖外部库，专门用于 Mink IK + PID 控制
"""

import numpy as np
from typing import Optional

try:
    from gravity_compensation import GravityCompensationController
    GRAVITY_COMP_AVAILABLE = True
except ImportError as e:
    GRAVITY_COMP_AVAILABLE = False
    print(f"⚠ 重力补偿模块不可用: {e}")
    print(f"  请确保 gravity_compensation.py 在同一目录")
except Exception as e:
    GRAVITY_COMP_AVAILABLE = False
    print(f"⚠ 重力补偿模块加载错误: {e}")
    import traceback
    traceback.print_exc()


class SimplePIDController:
    """
    简单的PID力矩控制器
    
    输入: 目标关节位置 q_target, 当前关节状态 (q, dq)
    输出: 控制力矩 tau
    """
    
    def __init__(
        self,
        kp: float | np.ndarray = 500.0,
        kd: float | np.ndarray = 30.0,
        ki: float | np.ndarray = 100.0,
        n_joints: int = 6,
        torque_limits: Optional[float | np.ndarray] = None,
        urdf_path: Optional[str] = None,
        use_gravity_compensation: bool = True,
    ):
        """
        初始化PID控制器
        
        Args:
            kp: 比例增益 (可以是标量或数组[n_joints,])
            kd: 微分增益 (可以是标量或数组[n_joints,])
            ki: 积分增益 (可以是标量或数组[n_joints,])
            n_joints: 关节数量
            torque_limits: 力矩限制 (可以是标量或数组[n_joints,])
            urdf_path: URDF文件路径（用于重力补偿）
            use_gravity_compensation: 是否启用重力补偿
        """
        self.n_joints = n_joints
        self.use_gravity_compensation = use_gravity_compensation and GRAVITY_COMP_AVAILABLE
        
        # 初始化重力补偿
        self.gravity_comp = None
        if self.use_gravity_compensation and urdf_path is not None:
            try:
                self.gravity_comp = GravityCompensationController(urdf_path)
                print("✓ 重力补偿已启用")
            except Exception as e:
                print(f"⚠ 重力补偿初始化失败: {e}")
                self.use_gravity_compensation = False
        
        # 转换增益为数组
        self.kp = np.atleast_1d(kp)
        if self.kp.size == 1:
            self.kp = np.full(n_joints, self.kp[0])
        
        self.kd = np.atleast_1d(kd)
        if self.kd.size == 1:
            self.kd = np.full(n_joints, self.kd[0])
        
        self.ki = np.atleast_1d(ki)
        if self.ki.size == 1:
            self.ki = np.full(n_joints, self.ki[0])
        
        # 力矩限制 (ARM620: Joint 1-3: ±49 N·m, Joint 4-6: ±9 N·m)
        if torque_limits is None:
            self.torque_limits = np.array([49.0, 49.0, 49.0, 9.0, 9.0, 9.0])
        else:
            self.torque_limits = np.atleast_1d(torque_limits)
            if self.torque_limits.size == 1:
                self.torque_limits = np.full(n_joints, self.torque_limits[0])
        
        # 积分误差累积
        self.integrated_error = np.zeros(n_joints)
        
        # 速度滤波（低通滤波，减少速度噪声）
        self.filtered_velocity = np.zeros(n_joints)
        self.velocity_filter_alpha = 0.3  # 滤波系数，越小越平滑
        
        print(f"✓ Simple PID Controller initialized")
        print(f"  - Kp: {self.kp}")
        print(f"  - Kd: {self.kd}")
        print(f"  - Ki: {self.ki}")
        print(f"  - Torque limits (N·m): {self.torque_limits}")
        print(f"  - Gravity compensation: {self.use_gravity_compensation}")
        print(f"  - Velocity filter alpha: {self.velocity_filter_alpha}")
    
    def compute_torque(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_target: np.ndarray,
        dq_target: Optional[np.ndarray] = None,
        dt: float = 0.002,
        debug: bool = False,
    ) -> np.ndarray:
        """
        计算PID控制力矩
        
        Args:
            q: 当前关节位置 (n_joints,)
            dq: 当前关节速度 (n_joints,)
            q_target: 目标关节位置 (n_joints,)
            dq_target: 目标关节速度 (默认为零)
            dt: 时间步长
            debug: 是否输出调试信息
            
        Returns:
            tau: 控制力矩 (n_joints,)
        """
        if dq_target is None:
            dq_target = np.zeros(self.n_joints)
        
        # 位置误差和速度误差
        error = q_target - q
        error_rate = dq_target - dq
        
        # PID计算
        tau_pid = self.kp * error + self.kd * error_rate
        
        # 积分项
        tau_integral = np.zeros(self.n_joints)
        if np.any(self.ki > 0):
            self.integrated_error += error * dt
            # 积分饱和限制
            self.integrated_error = np.clip(self.integrated_error, -1.0, 1.0)
            tau_integral = self.ki * self.integrated_error
            tau_pid += tau_integral
        
        # 添加重力补偿
        tau_gravity = np.zeros(self.n_joints)
        if self.use_gravity_compensation and self.gravity_comp is not None:
            try:
                # 只传递机械臂的6个关节给重力补偿
                q_arm = q[:self.n_joints]
                dq_arm = dq[:self.n_joints]
                tau_gravity = self.gravity_comp.compute_gravity_torque(q_arm, dq_arm)
                
                if debug and np.any(tau_gravity != 0):
                    print(f"[DEBUG] 重力补偿输入 q: {q_arm}")
                    print(f"[DEBUG] 重力补偿输出: {tau_gravity}")
            except Exception as e:
                if debug:
                    print(f"[ERROR] 重力补偿计算失败: {e}")
                tau_gravity = np.zeros(self.n_joints)
        
        # 总力矩
        tau_total = tau_pid + tau_gravity
        
        # 调试输出
        if debug:
            print("\n=== 力矩分解 ===")
            print(f"PID力矩:      {np.array2string(tau_pid, precision=2, suppress_small=True)}")
            print(f"  - P项:      {np.array2string(self.kp * error, precision=2, suppress_small=True)}")
            print(f"  - D项:      {np.array2string(self.kd * error_rate, precision=2, suppress_small=True)}")
            print(f"  - I项:      {np.array2string(tau_integral, precision=2, suppress_small=True)}")
            print(f"重力补偿力矩: {np.array2string(tau_gravity, precision=2, suppress_small=True)}")
            print(f"总力矩:       {np.array2string(tau_total, precision=2, suppress_small=True)}")
            print(f"位置误差(deg):{np.array2string(np.rad2deg(error), precision=2, suppress_small=True)}")
        
        # 力矩限幅
        tau = np.clip(tau_total, -self.torque_limits, self.torque_limits)
        
        return tau
    
    def reset(self):
        """重置积分误差"""
        self.integrated_error = np.zeros(self.n_joints)
