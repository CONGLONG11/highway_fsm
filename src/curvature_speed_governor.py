"""
curvature_speed_governor.py — 曲率积分能量速度调控器（构想①）

核心思想：
  不在弯道当前点才减速（已来不及），而是沿前方道路计算
  一段曲率能量积分，提前感知未来路段的弯曲程度，
  在进入弯道前就开始平滑减速。

  E_κ = ∫₀^Lf κ(s)² ds

  v_max = v_target / (1 + α·E_κ)

直接解决：自车100km/h进弯冲出车道的致命BUG
"""

import math
import numpy as np

class CurvatureSpeedGovernor:
    """
    曲率积分能量速度调控器

    每帧工作流程：
    1. 从当前位置沿车道中心线向前采样 N 个路点
    2. 三点法计算每个采样点的局部曲率 κ
    3. 曲率能量积分 E = Σκ²·Δs
    4. 根据 E 和最大曲率计算安全速度上限
    5. 低通滤波平滑输出（减速早、加速慢）
    """

    def __init__(self, config=None):
        cfg = config or {}
        self.lookahead_distance = cfg.get('lookahead_distance', 150.0)
        self.sample_spacing = cfg.get('sample_spacing', 3.0)
        self.alpha = cfg.get('alpha', 600.0)
        self.a_lat_max = cfg.get('a_lat_max', 3.0)
        self.min_speed = cfg.get('min_speed', 30.0)
        self.filter_alpha_down = cfg.get('filter_alpha_down', 0.06)
        self.filter_alpha_up = cfg.get('filter_alpha_up', 0.15)
        self._prev_output = None

        self.debug = {
            'curvature_energy': 0.0,
            'max_curvature': 0.0,
            'energy_speed_limit': 0.0,
            'point_speed_limit': 0.0,
            'final_speed_limit': 0.0,
            'n_samples': 0,
        }

    def compute_safe_speed(self, current_waypoint, target_speed_kmh,
                           current_speed_kmh):
        """
        计算考虑前方曲率的安全目标速度

        参数：
            current_waypoint : carla.Waypoint
            target_speed_kmh : float — 巡航目标速度 (km/h)
            current_speed_kmh: float — 当前车速 (km/h)

        返回：
            safe_speed_kmh : float — 安全目标速度 (km/h)
        """
        # ---- 1. 前方路点采样 ----
        points = self._sample_forward_waypoints(current_waypoint)
        self.debug['n_samples'] = len(points)

        if len(points) < 3:
            return self._apply_filter(target_speed_kmh * 0.6)

        # ---- 2. 局部曲率计算 ----
        curvatures = self._compute_curvatures(points)
        if not curvatures:
            return self._apply_filter(target_speed_kmh)

        # ---- 3. 曲率能量积分 ----
        curvature_energy = sum(k ** 2 for k in curvatures) * self.sample_spacing
        max_curvature = max(curvatures)

        self.debug['curvature_energy'] = round(curvature_energy, 6)
        self.debug['max_curvature'] = round(max_curvature, 5)

        # ---- 4a. 能量限速 ----
        energy_speed = target_speed_kmh / (1.0 + self.alpha * curvature_energy)
        self.debug['energy_speed_limit'] = round(energy_speed, 1)

        # ---- 4b. 单点曲率限速（安全兜底）----
        if max_curvature > 1e-5:
            point_speed = math.sqrt(self.a_lat_max / max_curvature) * 3.6
        else:
            point_speed = target_speed_kmh
        self.debug['point_speed_limit'] = round(point_speed, 1)

        # ---- 5. 取最保守值 ----
        safe_speed = min(energy_speed, point_speed, target_speed_kmh)
        safe_speed = max(safe_speed, self.min_speed)
        self.debug['final_speed_limit'] = round(safe_speed, 1)

        # ---- 6. 非对称低通滤波 ----
        return self._apply_filter(safe_speed)

    # ------------------------------------------------------------------
    def _sample_forward_waypoints(self, start_wp):
        """沿车道中心线向前采样路点 → [(x, y), ...]"""
        points = []
        wp = start_wp
        total = 0.0
        while total < self.lookahead_distance:
            loc = wp.transform.location
            points.append((loc.x, loc.y))
            nxt = wp.next(self.sample_spacing)
            if not nxt:
                break
            wp = nxt[0]
            total += self.sample_spacing
        return points

    def _compute_curvatures(self, points):
        """三点外接圆曲率"""
        curvatures = []
        for i in range(1, len(points) - 1):
            A = np.array(points[i - 1])
            B = np.array(points[i])
            C = np.array(points[i + 1])
            AB, AC, BC = B - A, C - A, C - B
            cross = abs(AB[0] * AC[1] - AB[1] * AC[0])
            ab, bc, ac = np.linalg.norm(AB), np.linalg.norm(BC), np.linalg.norm(AC)
            denom = ab * bc * ac
            curvatures.append(2.0 * cross / denom if denom > 1e-10 else 0.0)
        return curvatures

    def _apply_filter(self, value):
        """非对称低通：减速快响应（α小），加速慢响应（α大）"""
        if self._prev_output is None:
            self._prev_output = value
            return value
        alpha = self.filter_alpha_down if value < self._prev_output else self.filter_alpha_up
        out = alpha * value + (1.0 - alpha) * self._prev_output
        self._prev_output = out
        return out

    def reset(self):
        self._prev_output = None
