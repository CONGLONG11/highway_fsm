"""
phase_space_safety.py — 驾驶行为相空间安全包络（构想②）

在（相对距离 d, 相对速度 v）的二维相空间中定义安全边界：

  d_safe(v_approach) = v² / (2·a_max) + v·t_react + d_buf

边界内侧安全，外侧危险。比单一 TTC 更精确。
"""

import math
import numpy as np
from collections import deque

class PhaseSpaceSafetyEnvelope:
    """相空间安全包络计算器"""

    def __init__(self, config=None):
        cfg = config or {}
        self.a_ego_max = cfg.get('a_ego_max', 6.0)
        self.t_reaction = cfg.get('t_reaction', 0.3)
        self.d_buffer = cfg.get('d_buffer', 3.0)
        self.margin_ratio = cfg.get('margin_ratio', 1.2)

    # ----------------------------------------------------------
    def safety_boundary(self, approach_speed: float) -> float:
        """给定接近速度，返回最小安全距离"""
        if approach_speed <= 0:
            return self.d_buffer
        v = approach_speed
        return v * self.t_reaction + v ** 2 / (2.0 * self.a_ego_max) + self.d_buffer

    def evaluate(self, rel_distance: float,
                 rel_velocity: float) -> dict:
        """
        评估相空间状态安全性。

        参数：
            rel_distance : 与目标车辆的距离 (m), > 0
            rel_velocity : 相对速度 (m/s)
                           正 = 对方比我快（远离，安全）
                           负 = 我比对方快（接近，可能危险）

        返回：
            {'level': str, 'margin': float, 'min_safe_dist': float,
             'urgency': float}
        """
        approach = max(-rel_velocity, 0.0)
        d_safe = self.safety_boundary(approach)
        margin = rel_distance / d_safe if d_safe > 0.1 else 10.0

        if margin > self.margin_ratio:
            level, urgency = 'safe', 0.0
        elif margin > 1.0:
            level = 'warning'
            urgency = 1.0 - (margin - 1.0) / (self.margin_ratio - 1.0)
        elif margin > 0.5:
            level = 'danger'
            urgency = 0.7 + 0.3 * (1.0 - margin)
        else:
            level, urgency = 'critical', 1.0

        if rel_velocity > 2.0 and level in ('warning', 'danger'):
            level, urgency = 'safe', max(0.0, urgency - 0.5)

        return {
            'level': level,
            'margin': round(margin, 3),
            'min_safe_dist': round(d_safe, 2),
            'urgency': round(urgency, 3),
        }

    def check_lane_change(self, front_dist, front_speed_kmh,
                          rear_dist, rear_speed_kmh,
                          ego_speed_kmh) -> dict:
        """
        变道相空间安全检查。

        参数均为正值。front_dist/rear_dist 为到前/后车的距离 (m)。
        """
        ego_v = ego_speed_kmh / 3.6
        fv = front_speed_kmh / 3.6
        rv = rear_speed_kmh / 3.6

        front_eval = self.evaluate(front_dist, fv - ego_v)
        rear_eval = self.evaluate(rear_dist, ego_v - rv)

        f_ok = front_eval['level'] in ('safe', 'warning')
        r_ok = rear_eval['level'] in ('safe', 'warning')

        return {
            'front': front_eval,
            'rear': rear_eval,
            'overall_safe': f_ok and r_ok,
        }
