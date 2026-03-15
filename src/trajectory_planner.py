"""
trajectory_planner.py — 变道轨迹规划器
================================================
基于五次多项式的平滑变道轨迹生成。

五次多项式保证：
  - 起点/终点 位置连续
  - 起点/终点 速度连续  (一阶导 = 0)
  - 起点/终点 加速度连续 (二阶导 = 0)

输出格式：
  [carla.Transform, carla.Transform, ...]
  与 controller.py 的 run_step(trajectory=...) 接口兼容。
"""

import math
import numpy as np
import carla

class LaneChangeTrajectoryPlanner:
    """
    变道轨迹规划器

    规划流程：
      1. 沿目标车道采样前方 waypoint 序列（纵向参考线）
      2. 用五次多项式计算从当前车道到目标车道的横向过渡
      3. 将纵向参考线 + 横向偏移 合成为世界坐标轨迹
      4. 输出 list[carla.Transform]
    """

    def __init__(self, config=None):
        cfg = config or {}
        planner_cfg = cfg.get('planner', {})

        self.T_lc = planner_cfg.get('lane_change_time', 3.5)       # 变道时间 (s)
        self.lane_width = planner_cfg.get('lane_width', 3.5)       # 车道宽度 (m)
        self.n_points = planner_cfg.get('sample_points', 80)       # 轨迹采样点数
        self.post_distance = planner_cfg.get('post_distance', 40.0) # 变道后直行段 (m)
        self.min_speed = 5.0  # km/h，低速兜底

    def plan(self, ego_transform, ego_speed_kmh, target_waypoint):
        """
        规划变道轨迹。

        参数：
            ego_transform    : carla.Transform — 自车当前位姿
            ego_speed_kmh    : float — 自车当前速度 (km/h)
            target_waypoint  : carla.Waypoint — 目标车道的某个 waypoint

        返回：
            list[carla.Transform] — 轨迹点序列
        """
        speed_ms = max(ego_speed_kmh / 3.6, self.min_speed / 3.6)

        # ============================================
        # 1. 计算纵向距离
        # ============================================
        lc_longitudinal = speed_ms * self.T_lc
        total_longitudinal = lc_longitudinal + self.post_distance

        # ============================================
        # 2. 沿目标车道采样参考路点
        # ============================================
        target_wps = self._sample_target_lane(target_waypoint, 
                                               total_longitudinal)
        if len(target_wps) < 2:
            return self._fallback_trajectory(ego_transform, target_waypoint)

        # ============================================
        # 3. 沿当前车道采样参考路点（用于计算横向偏移方向）
        # ============================================
        ego_wp = target_waypoint  # target_wp 其实就在目标车道上
        # 我们需要从自车当前位置出发的当前车道参考线
        map_ref = target_waypoint  # 用目标车道作为终点参考

        # ============================================
        # 4. 构建Frenet坐标系下的五次多项式轨迹
        # ============================================
        trajectory = self._build_quintic_trajectory(
            ego_transform, speed_ms, target_wps, lc_longitudinal
        )

        return trajectory

    def _sample_target_lane(self, start_wp, distance, spacing=2.0):
        """沿目标车道向前采样 waypoint"""
        wps = []
        wp = start_wp
        total = 0.0

        while total < distance:
            wps.append(wp)
            nxt = wp.next(spacing)
            if not nxt:
                break
            wp = nxt[0]
            total += spacing

        return wps

    def _build_quintic_trajectory(self, ego_tf, speed_ms, 
                                   target_wps, lc_dist):
        """
        构建五次多项式变道轨迹

        思路：
        - 纵向(s): 沿目标车道参考线的弧长，均匀采样
        - 横向(d): 五次多项式从 d0 过渡到 0
          d(s) = a0 + a1*τ + a2*τ² + a3*τ³ + a4*τ⁴ + a5*τ⁵
          其中 τ = s / s_lc ∈ [0, 1]

        边界条件：
          d(0) = d0 (当前横向偏移), d'(0) = 0, d''(0) = 0
          d(1) = 0  (到达目标车道),  d'(1) = 0, d''(1) = 0
        """
        if len(target_wps) < 2:
            return []

        ego_loc = ego_tf.location

        # 计算自车相对目标车道的横向偏移 d0
        ref_tf = target_wps[0].transform
        ref_yaw = math.radians(ref_tf.rotation.yaw)

        dx = ego_loc.x - ref_tf.location.x
        dy = ego_loc.y - ref_tf.location.y

        # 横向偏移 (Frenet d 坐标)
        d0 = -dx * math.sin(ref_yaw) + dy * math.cos(ref_yaw)

        # 五次多项式系数
        # d(τ) = d0·(1 - 10τ³ + 15τ⁴ - 6τ⁵)
        # 这是满足上述6个边界条件的标准形式

        # 总纵向距离（沿参考线的弧长）
        total_s = 0.0
        ref_s = [0.0]
        for i in range(1, len(target_wps)):
            seg = target_wps[i - 1].transform.location.distance(
                target_wps[i].transform.location
            )
            total_s += seg
            ref_s.append(total_s)

        # 均匀采样
        trajectory = []
        s_lc = min(lc_dist, total_s * 0.8)  # 变道段不超过总长的80%

        for i in range(self.n_points):
            s = total_s * i / (self.n_points - 1)

            # 找到 s 对应的参考线插值位置
            ref_idx = 0
            for j in range(len(ref_s) - 1):
                if ref_s[j] <= s <= ref_s[j + 1]:
                    ref_idx = j
                    break
            else:
                ref_idx = len(ref_s) - 2

            # 线性插值
            seg_len = ref_s[ref_idx + 1] - ref_s[ref_idx]
            if seg_len < 1e-6:
                ratio = 0.0
            else:
                ratio = (s - ref_s[ref_idx]) / seg_len

            wp_a = target_wps[ref_idx].transform
            wp_b = target_wps[min(ref_idx + 1, len(target_wps) - 1)].transform

            ref_x = wp_a.location.x + ratio * (wp_b.location.x - wp_a.location.x)
            ref_y = wp_a.location.y + ratio * (wp_b.location.y - wp_a.location.y)
            ref_z = wp_a.location.z + ratio * (wp_b.location.z - wp_a.location.z)

            # 插值 yaw
            yaw_a = math.radians(wp_a.rotation.yaw)
            yaw_b = math.radians(wp_b.rotation.yaw)
            dyaw = yaw_b - yaw_a
            while dyaw > math.pi: dyaw -= 2 * math.pi
            while dyaw < -math.pi: dyaw += 2 * math.pi
            ref_yaw = yaw_a + ratio * dyaw

            # 计算横向偏移
            if s < s_lc and s_lc > 0:
                tau = s / s_lc
                # 五次多项式: 从 d0 平滑过渡到 0
                d = d0 * (1.0 - 10.0 * tau**3 + 15.0 * tau**4 - 6.0 * tau**5)
            else:
                d = 0.0  # 变道完成，沿目标车道中心线

            # 横向偏移方向 (参考线法线方向)
            normal_x = -math.sin(ref_yaw)
            normal_y = math.cos(ref_yaw)

            world_x = ref_x + d * normal_x
            world_y = ref_y + d * normal_y
            world_z = ref_z

            # 计算轨迹点的航向
            point_yaw = math.degrees(ref_yaw)

            location = carla.Location(x=world_x, y=world_y, z=world_z)
            rotation = carla.Rotation(yaw=point_yaw)
            trajectory.append(carla.Transform(location, rotation))

        # 修正轨迹点的航向角（用前后两点计算）
        trajectory = self._smooth_yaw(trajectory)

        return trajectory

    def _smooth_yaw(self, trajectory):
        """用前后两点的方向修正每个轨迹点的航向"""
        if len(trajectory) < 2:
            return trajectory

        for i in range(len(trajectory)):
            if i < len(trajectory) - 1:
                loc_a = trajectory[i].location
                loc_b = trajectory[i + 1].location
            else:
                loc_a = trajectory[i - 1].location
                loc_b = trajectory[i].location

            dx = loc_b.x - loc_a.x
            dy = loc_b.y - loc_a.y
            yaw = math.degrees(math.atan2(dy, dx))

            trajectory[i] = carla.Transform(
                trajectory[i].location,
                carla.Rotation(yaw=yaw)
            )

        return trajectory

    def _fallback_trajectory(self, ego_tf, target_wp):
        """兜底：直接用目标车道 waypoint 生成轨迹"""
        wps = self._sample_target_lane(target_wp, 80.0, spacing=2.0)
        return [wp.transform for wp in wps]
