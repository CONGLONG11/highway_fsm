"""
运动规划模块 (Motion Planner)
=================================
创新点 2: APF + 五次多项式的融合轨迹规划

流程:
  1. 五次多项式生成基准变道轨迹 (保证运动学平滑)
  2. 人工势场对每个轨迹点施加修正 (避开障碍物)
  3. 平滑滤波消除势场修正引入的噪声
  4. 输出最终轨迹点列表
"""
import numpy as np
import math
import carla

from risk_field import RiskField

class QuinticPolynomial:
    """
    五次多项式求解器
    x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5

    给定起点和终点的 位置、速度、加速度，求解系数。
    保证了连续的位置、速度、加速度（即 Minimum Jerk 特性）。
    """

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([
            [T**3,      T**4,       T**5],
            [3*T**2,    4*T**3,     5*T**4],
            [6*T,       12*T**2,    20*T**3]
        ])
        b = np.array([
            xe - self.a0 - self.a1*T - self.a2*T**2,
            vxe - self.a1 - 2*self.a2*T,
            axe - 2*self.a2
        ])

        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        return (self.a0 + self.a1*t + self.a2*t**2 +
                self.a3*t**3 + self.a4*t**4 + self.a5*t**5)

    def calc_first_derivative(self, t):
        return (self.a1 + 2*self.a2*t + 3*self.a3*t**2 +
                4*self.a4*t**3 + 5*self.a5*t**4)

    def calc_second_derivative(self, t):
        return (2*self.a2 + 6*self.a3*t +
                12*self.a4*t**2 + 20*self.a5*t**3)

    def calc_third_derivative(self, t):
        return 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2

class MotionPlanner:
    """运动规划器"""

    def __init__(self, vehicle, config=None):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()

        config = config or {}
        planner_cfg = config.get('planner', {})

        self.lane_change_duration = planner_cfg.get('lane_change_duration', 3.5)
        self.dt = planner_cfg.get('trajectory_dt', 0.05)
        self.max_lat_accel = planner_cfg.get('max_lateral_accel', 3.0)

        # 势场修正器
        apf_config = config.get('apf', {})
        self.risk_field = RiskField(apf_config)
        self.apf_enabled = True

    def generate_keep_lane_trajectory(self, ego_wp, length=30.0):
        """
        生成车道保持的前方参考轨迹
        :return: list of carla.Transform
        """
        trajectory = []
        wps = ego_wp.next(length)
        if not wps:
            return trajectory

        # 沿车道中心线取点
        current_wp = ego_wp
        step = 2.0  # 每 2m 一个点
        distance = 0.0

        while distance < length:
            next_wps = current_wp.next(step)
            if not next_wps:
                break
            current_wp = next_wps[0]
            trajectory.append(current_wp.transform)
            distance += step

        return trajectory

    def generate_lane_change_trajectory(self, start_wp, target_wp,
                                        ego_speed, obstacles=None):
        """
        生成变道轨迹 (五次多项式 + 势场修正)

        :param start_wp: 起点 Waypoint (当前车道)
        :param target_wp: 终点 Waypoint (目标车道)
        :param ego_speed: 当前车速 km/h
        :param obstacles: 感知到的障碍物列表 (用于势场修正)
        :return: list of carla.Transform
        """
        if obstacles is None:
            obstacles = []

        v_ms = max(ego_speed / 3.6, 5.0)  # m/s, 最低 5m/s

        # === 动态计算变道时间 ===
        # 高速时给更长时间，保证横向加速度不超限
        lane_width = start_wp.transform.location.distance(
            target_wp.transform.location
        )
        # T = sqrt(5.77 * d / a_max) (五次多项式峰值加速度估算)
        T_min = math.sqrt(5.77 * lane_width / self.max_lat_accel)
        T = max(T_min, self.lane_change_duration)
        T = min(T, 6.0)  # 不超过 6 秒

        # === 五次多项式规划横向运动 ===
        # 横向: d(0)=0, d'(0)=0, d''(0)=0 → d(T)=lane_width, d'(T)=0, d''(T)=0
        lat_qp = QuinticPolynomial(0, 0, 0, lane_width, 0, 0, T)

        # === 采样生成轨迹 ===
        time_steps = np.arange(0, T, self.dt)
        raw_trajectory = []

        for t in time_steps:
            # 纵向: 匀速前进
            s = v_ms * t

            # 横向: 五次多项式
            alpha = lat_qp.calc_point(t) / lane_width if lane_width > 0 else 0
            alpha = np.clip(alpha, 0, 1)

            # 在起点车道和目标车道之间插值
            wp1_list = start_wp.next(s + 0.1)
            wp2_list = target_wp.next(s + 0.1)

            if not wp1_list or not wp2_list:
                break

            wp1 = wp1_list[0]
            wp2 = wp2_list[0]

            loc1 = wp1.transform.location
            loc2 = wp2.transform.location

            # 位置插值
            x = (1 - alpha) * loc1.x + alpha * loc2.x
            y = (1 - alpha) * loc1.y + alpha * loc2.y
            z = (1 - alpha) * loc1.z + alpha * loc2.z

            raw_trajectory.append(np.array([x, y, z, t]))

        if len(raw_trajectory) < 2:
            return []

        # === 势场修正 (创新点2的核心步骤) ===
        corrected_trajectory = self._apply_apf_correction(
            raw_trajectory, target_wp, obstacles
        )

        # === 计算朝向角并转换为 carla.Transform ===
        final_trajectory = self._compute_yaw_and_convert(corrected_trajectory)

        return final_trajectory

    def _apply_apf_correction(self, raw_trajectory, target_wp, obstacles):
        """
        对原始轨迹施加人工势场修正

        只修正轨迹中段 (头尾保持不变，确保起终点精确)
        """
        if not self.apf_enabled or not obstacles:
            return raw_trajectory

        corrected = []
        n = len(raw_trajectory)

        # 目标位置
        goal_pos = np.array([
            target_wp.transform.location.x,
            target_wp.transform.location.y
        ])

        for i, pt in enumerate(raw_trajectory):
            # 头 10% 和尾 10% 不修正
            ratio = i / n
            if ratio < 0.1 or ratio > 0.9:
                corrected.append(pt.copy())
                continue

            # 修正权重: 中间最大，两端衰减 (钟形曲线)
            weight = math.sin(math.pi * ratio)

            original_pos = pt[:2].copy()
            corrected_pos = self.risk_field.correct_trajectory_point(
                original_pos, goal_pos, obstacles
            )

            # 加权混合
            final_pos = original_pos + weight * (corrected_pos - original_pos)

            new_pt = pt.copy()
            new_pt[0] = final_pos[0]
            new_pt[1] = final_pos[1]
            corrected.append(new_pt)

        # === 平滑滤波 (消除势场引入的锯齿) ===
        corrected = self._smooth_trajectory(corrected, window=5)

        return corrected

    def _smooth_trajectory(self, trajectory, window=5):
        """简单的滑动平均平滑"""
        if len(trajectory) < window:
            return trajectory

        smoothed = []
        half_w = window // 2

        for i in range(len(trajectory)):
            start = max(0, i - half_w)
            end = min(len(trajectory), i + half_w + 1)

            avg_x = np.mean([trajectory[j][0] for j in range(start, end)])
            avg_y = np.mean([trajectory[j][1] for j in range(start, end)])

            pt = trajectory[i].copy()
            pt[0] = avg_x
            pt[1] = avg_y
            smoothed.append(pt)

        return smoothed

    def _compute_yaw_and_convert(self, trajectory):
        """
        根据前后点差分计算 yaw，并转换为 carla.Transform 列表
        """
        result = []

        for i in range(len(trajectory)):
            x, y, z = trajectory[i][0], trajectory[i][1], trajectory[i][2]

            if i < len(trajectory) - 1:
                dx = trajectory[i+1][0] - x
                dy = trajectory[i+1][1] - y
                yaw = math.degrees(math.atan2(dy, dx))
            elif i > 0:
                dx = x - trajectory[i-1][0]
                dy = y - trajectory[i-1][1]
                yaw = math.degrees(math.atan2(dy, dx))
            else:
                yaw = 0.0

            transform = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(pitch=0, yaw=yaw, roll=0)
            )
            result.append(transform)

        return result
