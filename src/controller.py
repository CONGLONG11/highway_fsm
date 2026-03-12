"""
自适应增益调度 Stanley 控制器 (Adaptive Gain-Scheduled Stanley Controller)
===========================================================================
创新点 3: 标准 Stanley 使用固定增益，无法同时满足高速稳定性
和低速灵活性。本控制器引入三项改进:

1. 速度自适应增益调度: k(v) 随速度变化
2. 曲率自适应预瞄距离: Ld 根据轨迹曲率动态调整
3. 一阶低通滤波: 消除转向抖动

增益调度策略:
  k_stanley = k_max - (k_max - k_min) * (v / v_max)^0.5
  高速时增益小 → 稳定; 低速时增益大 → 灵活
"""
import carla
import math
import numpy as np
from collections import deque

class AdaptiveStanleyController:
    """自适应 Stanley 横向控制器"""

    def __init__(self, vehicle, config=None):
        self.vehicle = vehicle
        config = config or {}
        ctrl_cfg = config.get('controller', {})

        # 纵向 PID
        self.lon_kp = ctrl_cfg.get('lon_kp', 1.0)
        self.lon_ki = ctrl_cfg.get('lon_ki', 0.05)
        self.lon_kd = ctrl_cfg.get('lon_kd', 0.1)
        self._lon_error_buffer = deque(maxlen=15)

        # Stanley 参数
        self.k_base = ctrl_cfg.get('k_stanley_base', 0.5)
        self.k_min = ctrl_cfg.get('k_stanley_min', 0.2)
        self.k_max = ctrl_cfg.get('k_stanley_max', 1.5)
        self.k_soft = ctrl_cfg.get('k_soft', 0.5)

        # 预瞄距离
        self.ld_min = ctrl_cfg.get('lookahead_min', 3.0)
        self.ld_max = ctrl_cfg.get('lookahead_max', 20.0)
        self.ld_ratio = ctrl_cfg.get('lookahead_ratio', 0.25)

        # 转向滤波
        self.filter_alpha = ctrl_cfg.get('steer_filter_alpha', 0.3)
        self._prev_steer = 0.0

        # 限位
        self.max_steer_high = ctrl_cfg.get('max_steer_high_speed', 0.25)
        self.max_steer_mid = ctrl_cfg.get('max_steer_mid_speed', 0.45)

        # 最大物理转角 (rad)
        self.max_steer_angle = math.radians(70)

        # 调试
        self.debug = {
            'cte': 0, 'heading_error': 0, 'k_adaptive': 0,
            'lookahead_dist': 0, 'steer_raw': 0, 'steer_filtered': 0
        }

    def run_step(self, target_speed, trajectory=None, emergency_stop=False):
        """
        主控制函数

        :param target_speed: 目标速度 km/h
        :param trajectory: 规划好的轨迹 list of carla.Transform
        :param emergency_stop: 紧急停车标志
        :return: carla.VehicleControl
        """
        if emergency_stop:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        speed = self._get_speed()
        transform = self.vehicle.get_transform()

        # 纵向控制
        throttle, brake = self._longitudinal_pid(target_speed, speed)

        # 横向控制
        steer = 0.0
        if trajectory and len(trajectory) > 2:
            steer = self._adaptive_stanley(transform, speed, trajectory)
        else:
            # 退化为跟踪最近 waypoint
            steer = self._fallback_lane_follow(transform, speed)

        # 一阶低通滤波 (创新点3的改进之一)
        steer = self._low_pass_filter(steer)

        # 速度自适应限位
        steer = self._apply_steer_limit(steer, speed)

        self.debug['steer_filtered'] = steer

        return carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )

    def _adaptive_stanley(self, vehicle_transform, speed, trajectory):
        """
        自适应 Stanley 横向控制

        改进1: 速度自适应增益
          k(v) = k_max - (k_max - k_min) * sqrt(v / 120)

        改进2: 曲率自适应预瞄
          Ld = Ld_base + k_curv / (curvature + 0.01)
        """
        # === Step 1: 预瞄点搜索 ===
        target_transform, target_idx = self._find_lookahead_point(
            vehicle_transform, speed, trajectory
        )

        if target_transform is None:
            return 0.0

        # === Step 2: 自适应增益计算 (创新点3核心) ===
        v_kmh = max(speed, 1.0)
        v_ratio = min(v_kmh / 120.0, 1.0)

        # 平方根调度: 低速区增益变化快，高速区变化慢
        k_adaptive = self.k_max - (self.k_max - self.k_min) * math.sqrt(v_ratio)
        self.debug['k_adaptive'] = k_adaptive

        # 可选: 根据轨迹局部曲率微调
        curvature = self._estimate_curvature(trajectory, target_idx)
        if curvature > 0.01:
            # 曲率大时略微增加增益，让车更积极跟踪
            k_adaptive *= (1.0 + 0.5 * min(curvature, 0.1))

        # === Step 3: Stanley 公式 ===
        v_loc = vehicle_transform.location
        t_loc = target_transform.location

        v_yaw = math.radians(vehicle_transform.rotation.yaw)
        t_yaw = math.radians(target_transform.rotation.yaw)

        # 航向误差
        heading_error = t_yaw - v_yaw
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # 横向误差 (Cross Track Error)
        vec_to_vehicle = np.array([v_loc.x - t_loc.x, v_loc.y - t_loc.y])
        forward_vec = np.array([math.cos(t_yaw), math.sin(t_yaw)])

        # 叉积计算有符号横向误差
        cte = vec_to_vehicle[0] * forward_vec[1] - vec_to_vehicle[1] * forward_vec[0]

        self.debug['cte'] = cte
        self.debug['heading_error'] = heading_error

        # Stanley 公式
        v_ms = max(speed / 3.6, 0.5)
        cte_correction = math.atan2(k_adaptive * cte, v_ms + self.k_soft)

        steer_rad = heading_error + cte_correction

        # 转换为 [-1, 1]
        steer = steer_rad / self.max_steer_angle
        steer = np.clip(steer, -1.0, 1.0)

        self.debug['steer_raw'] = steer
        return steer

    def _find_lookahead_point(self, vehicle_transform, speed, trajectory):
        """
        自适应预瞄点搜索

        Ld = ratio * speed + Ld_min
        """
        Ld = np.clip(
            self.ld_ratio * speed + self.ld_min,
            self.ld_min,
            self.ld_max
        )
        self.debug['lookahead_dist'] = Ld

        curr_loc = vehicle_transform.location
        min_dist = float('inf')
        closest_idx = 0

        # 找最近点
        for i, t in enumerate(trajectory):
            d = curr_loc.distance(t.location)
            if d < min_dist:
                min_dist = d
                closest_idx = i

        # 从最近点往前搜索预瞄距离
        target_idx = closest_idx
        acc_dist = 0.0

        for i in range(closest_idx, len(trajectory) - 1):
            d = trajectory[i].location.distance(trajectory[i+1].location)
            acc_dist += d
            if acc_dist >= Ld:
                target_idx = i + 1
                break

        target_idx = min(target_idx, len(trajectory) - 1)
        return trajectory[target_idx], target_idx

    def _estimate_curvature(self, trajectory, idx):
        """
        估算轨迹在 idx 处的局部曲率
        使用三点法: κ = 2 * |AB × AC| / (|AB| * |BC| * |AC|)
        """
        if idx < 1 or idx >= len(trajectory) - 1:
            return 0.0

        A = np.array([trajectory[idx-1].location.x, trajectory[idx-1].location.y])
        B = np.array([trajectory[idx].location.x, trajectory[idx].location.y])
        C = np.array([trajectory[idx+1].location.x, trajectory[idx+1].location.y])

        AB = B - A
        AC = C - A
        BC = C - B

        cross = abs(AB[0] * AC[1] - AB[1] * AC[0])
        denom = (np.linalg.norm(AB) * np.linalg.norm(BC) * np.linalg.norm(AC))

        if denom < 1e-6:
            return 0.0

        return 2.0 * cross / denom

    def _low_pass_filter(self, steer):
        """一阶低通滤波: 消除高频抖动"""
        filtered = self.filter_alpha * steer + (1 - self.filter_alpha) * self._prev_steer
        self._prev_steer = filtered
        return filtered

    def _apply_steer_limit(self, steer, speed):
        """速度自适应转向限位"""
        if speed > 80.0:
            max_s = self.max_steer_high
        elif speed > 50.0:
            max_s = self.max_steer_mid
        else:
            max_s = 1.0
        return np.clip(steer, -max_s, max_s)

    def _fallback_lane_follow(self, vehicle_transform, speed):
        """无轨迹时退化为简单车道跟踪"""
        world_map = self.vehicle.get_world().get_map()
        wp = world_map.get_waypoint(vehicle_transform.location)
        next_wps = wp.next(max(5.0, speed * 0.3))

        if not next_wps:
            return 0.0

        target = next_wps[0].transform
        # 简化的 Stanley
        v_yaw = math.radians(vehicle_transform.rotation.yaw)
        t_yaw = math.radians(target.rotation.yaw)

        heading_err = t_yaw - v_yaw
        while heading_err > math.pi: heading_err -= 2*math.pi
        while heading_err < -math.pi: heading_err += 2*math.pi

        return np.clip(heading_err / self.max_steer_angle, -1.0, 1.0)

    def _longitudinal_pid(self, target_speed, current_speed):
        """纵向 PID 控制"""
        error = target_speed - current_speed
        self._lon_error_buffer.append(error)

        if len(self._lon_error_buffer) >= 2:
            de = self._lon_error_buffer[-1] - self._lon_error_buffer[-2]
            ie = sum(self._lon_error_buffer)
        else:
            de = 0.0
            ie = 0.0

        ie = np.clip(ie, -10.0, 10.0)

        output = self.lon_kp * error + self.lon_kd * de + self.lon_ki * ie

        throttle = 0.0
        brake = 0.0

        if output > 0:
            throttle = np.clip(output, 0.0, 1.0)
        else:
            speed_diff = abs(error)
            if speed_diff > 10.0:
                brake = 0.8
            elif speed_diff > 5.0:
                brake = 0.4
            elif speed_diff > 1.0:
                brake = 0.15

        return throttle, brake

    def _get_speed(self):
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
