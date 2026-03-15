"""
controller.py — 多预瞄增强 Stanley 横向控制 + 曲率前馈

核心改进（相对原版）：
  1. 多预瞄点加权融合：同时看近/中/远三个预瞄点，
     近处负责精确修正，远处负责弯道预判
  2. 曲率前馈：δ_ff = L·κ（Ackermann 几何），
     在弯道中提供前馈转向量，不再完全依赖误差反馈
  3. 曲率自适应增益：弯道中自动增大 K 以更积极跟踪
  4. 速度自适应增益：高速降 K（防过敏），低速升 K（提响应）
  5. 非对称低通滤波 + 速度转向限位
"""

import math
import numpy as np
import carla
from collections import deque

class VehicleController:
    """车辆纵横向控制器"""

    def __init__(self, vehicle, config=None):
        self.vehicle = vehicle
        cfg = config or {}
        ctrl = cfg.get('controller', {})

        # ======== 横向参数 ========
        self.k_min = ctrl.get('k_stanley_min', 0.3)
        self.k_max = ctrl.get('k_stanley_max', 1.8)
        self.k_soft = ctrl.get('k_soft', 1.0)
        self.max_steer_angle = math.radians(70)

        # 多预瞄距离 (m)
        self.preview_distances = ctrl.get('preview_distances', [5.0, 15.0, 35.0])
        # 对应权重（近处大→精确修正，远处小→方向预判）
        self.preview_weights = ctrl.get('preview_weights', [0.50, 0.35, 0.15])

        # 曲率前馈：车辆轴距 L (m)
        self.wheelbase = ctrl.get('wheelbase', 2.875)

        # 低通滤波
        self.steer_filter_alpha = ctrl.get('steer_filter_alpha', 0.35)
        self._prev_steer = 0.0

        # 速度转向限位
        self.max_steer_high = ctrl.get('max_steer_high_speed', 0.30)
        self.max_steer_mid  = ctrl.get('max_steer_mid_speed', 0.50)

        # ======== 纵向 PID ========
        self.lon_kp = ctrl.get('lon_kp', 0.8)
        self.lon_ki = ctrl.get('lon_ki', 0.03)
        self.lon_kd = ctrl.get('lon_kd', 0.1)
        self._lon_err_buf = deque(maxlen=20)

        # ======== 调试 ========
        self.debug = {
            'cte': 0.0, 'heading_error': 0.0, 'k_adaptive': 0.0,
            'curvature_ff': 0.0, 'steer_raw': 0.0, 'steer_filtered': 0.0,
        }

    # ==============================================================
    #                       主接口
    # ==============================================================
    def run_step(self, target_speed, trajectory=None,
                 emergency_stop=False):
        """
        参数：
            target_speed  : float — 目标速度 (km/h)
            trajectory    : list[carla.Transform] 或 None
            emergency_stop: bool
        返回：
            carla.VehicleControl
        """
        if emergency_stop:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        speed = self._get_speed()
        transform = self.vehicle.get_transform()

        # ---- 纵向 ----
        throttle, brake = self._longitudinal_pid(target_speed, speed)

        # ---- 横向 ----
        if trajectory and len(trajectory) >= 3:
            steer = self._multi_preview_stanley(transform, speed, trajectory)
        else:
            steer = self._waypoint_fallback(transform, speed)

        # ---- 低通滤波 ----
        steer = self._filter_steer(steer)
        # ---- 速度限位 ----
        steer = self._limit_steer(steer, speed)

        self.debug['steer_filtered'] = round(steer, 4)

        return carla.VehicleControl(
            throttle=float(np.clip(throttle, 0, 1)),
            steer=float(np.clip(steer, -1, 1)),
            brake=float(np.clip(brake, 0, 1)),
        )

    # ==============================================================
    #              多预瞄增强 Stanley + 曲率前馈
    # ==============================================================
    def _multi_preview_stanley(self, veh_tf, speed_kmh, trajectory):
        """
        多预瞄点加权融合 + 曲率前馈

        对每个预瞄距离 d_i：
          1. 沿轨迹找到距当前最近点前方 d_i 处的目标点
          2. 计算该目标点的 Stanley 转向量 δ_i
        最终转向 = Σ w_i δ_i + δ_feedforward
        """
        v_ms = max(speed_kmh / 3.6, 0.5)
        v_yaw = math.radians(veh_tf.rotation.yaw)
        loc = veh_tf.location

        # ---------- 找轨迹最近点索引 ----------
        closest_idx, closest_dist = 0, float('inf')
        for i, t in enumerate(trajectory):
            d = loc.distance(t.location)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i

        # 记录最近点的 CTE
        cte_0 = self._cross_track_error(loc, trajectory[closest_idx], v_yaw)
        self.debug['cte'] = round(cte_0, 3)

        # ---------- 速度自适应 K ----------
        v_ratio = min(speed_kmh / 120.0, 1.0)
        k_speed = self.k_max - (self.k_max - self.k_min) * math.sqrt(v_ratio)

        # ---------- 曲率估计（最近点附近三点法）----------
        kappa = self._estimate_curvature_at(trajectory, closest_idx)

        # 曲率自适应增益提升：弯道中 K 更大以更积极跟踪
        k_curv_boost = 1.0 + 5.0 * abs(kappa)
        k_adaptive = k_speed * k_curv_boost
        k_adaptive = min(k_adaptive, self.k_max * 2.0)
        self.debug['k_adaptive'] = round(k_adaptive, 3)

        # ---------- 多预瞄点融合 ----------
        delta_sum = 0.0
        weight_sum = 0.0

        for dist_i, w_i in zip(self.preview_distances, self.preview_weights):
            # 速度缩放预瞄距离
            scale = 0.6 + speed_kmh / 150.0
            scaled_dist = dist_i * scale

            target_tf = self._find_point_at_dist(
                trajectory, closest_idx, scaled_dist
            )
            if target_tf is None:
                continue

            t_yaw = math.radians(target_tf.rotation.yaw)

            # 航向误差
            he = t_yaw - v_yaw
            while he > math.pi:  he -= 2 * math.pi
            while he < -math.pi: he += 2 * math.pi

            # 横向误差
            cte = self._cross_track_error(loc, target_tf, t_yaw)

            # Stanley 公式
            cte_term = math.atan2(k_adaptive * cte, v_ms + self.k_soft)
            delta_i = he + cte_term

            delta_sum += w_i * delta_i
            weight_sum += w_i

        if weight_sum < 1e-6:
            return 0.0

        delta_feedback = delta_sum / weight_sum

        # ---------- 曲率前馈 ----------
        delta_ff = self.wheelbase * kappa
        self.debug['curvature_ff'] = round(delta_ff, 4)
        self.debug['heading_error'] = round(delta_feedback, 4)

        # 总转向 = 反馈 + 前馈
        delta_total = delta_feedback + delta_ff

        # 归一化
        steer = delta_total / self.max_steer_angle
        steer = np.clip(steer, -1.0, 1.0)
        self.debug['steer_raw'] = round(steer, 4)

        return steer

    # ==============================================================
    #                  辅助函数
    # ==============================================================
    def _cross_track_error(self, ego_loc, target_tf, target_yaw):
        """计算横向误差（带符号）"""
        dx = ego_loc.x - target_tf.location.x
        dy = ego_loc.y - target_tf.location.y
        return dx * math.sin(target_yaw) - dy * math.cos(target_yaw)

    def _find_point_at_dist(self, trajectory, start_idx, target_dist):
        """沿轨迹从 start_idx 向前搜索距离 target_dist 处的点"""
        acc = 0.0
        for i in range(start_idx, len(trajectory) - 1):
            seg = trajectory[i].location.distance(trajectory[i + 1].location)
            acc += seg
            if acc >= target_dist:
                return trajectory[i + 1]
        return trajectory[-1] if trajectory else None

    def _estimate_curvature_at(self, trajectory, idx):
        """三点法估计轨迹在 idx 处的曲率"""
        n = len(trajectory)
        # 取 idx 前后各 3~5 个点，跨度更大以抗噪
        step = min(5, max(1, n // 20))
        i_prev = max(0, idx - step)
        i_next = min(n - 1, idx + step)
        if i_prev == idx or i_next == idx:
            return 0.0

        A = trajectory[i_prev].location
        B = trajectory[idx].location
        C = trajectory[i_next].location

        ABx, ABy = B.x - A.x, B.y - A.y
        ACx, ACy = C.x - A.x, C.y - A.y
        BCx, BCy = C.x - B.x, C.y - B.y

        cross = abs(ABx * ACy - ABy * ACx)
        ab = math.sqrt(ABx ** 2 + ABy ** 2)
        bc = math.sqrt(BCx ** 2 + BCy ** 2)
        ac = math.sqrt(ACx ** 2 + ACy ** 2)
        denom = ab * bc * ac

        return 2.0 * cross / denom if denom > 1e-8 else 0.0

    def _waypoint_fallback(self, veh_tf, speed_kmh):
        """
        无轨迹时的航点跟踪（KEEP_LANE 模式由 main.py 提供航点轨迹，
        这里仅作最后兜底）
        """
        world = self.vehicle.get_world()
        wp = world.get_map().get_waypoint(veh_tf.location)
        dist = max(8.0, speed_kmh * 0.15)
        nxt = wp.next(dist)
        if not nxt:
            return 0.0
        target = nxt[0].transform
        v_yaw = math.radians(veh_tf.rotation.yaw)
        t_yaw = math.radians(target.rotation.yaw)
        he = t_yaw - v_yaw
        while he > math.pi:  he -= 2 * math.pi
        while he < -math.pi: he += 2 * math.pi
        return np.clip(he / self.max_steer_angle, -1, 1)

    def _filter_steer(self, steer):
        a = self.steer_filter_alpha
        out = a * steer + (1.0 - a) * self._prev_steer
        self._prev_steer = out
        return out

    def _limit_steer(self, steer, speed_kmh):
        if speed_kmh > 80:
            mx = self.max_steer_high
        elif speed_kmh > 50:
            mx = self.max_steer_mid
        else:
            mx = 1.0
        return np.clip(steer, -mx, mx)

    # ---------- 纵向 PID ----------
    def _longitudinal_pid(self, target, current):
        err = target - current
        self._lon_err_buf.append(err)
        de = (self._lon_err_buf[-1] - self._lon_err_buf[-2]) if len(self._lon_err_buf) >= 2 else 0
        ie = np.clip(sum(self._lon_err_buf), -15, 15)
        out = self.lon_kp * err + self.lon_kd * de + self.lon_ki * ie
        if out >= 0:
            return min(out, 1.0), 0.0
        else:
            diff = abs(err)
            if diff > 15:
                return 0.0, 0.9
            elif diff > 8:
                return 0.0, 0.5
            elif diff > 3:
                return 0.0, 0.25
            else:
                return 0.0, 0.1

    def _get_speed(self):
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
