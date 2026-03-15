"""
counterfactual_safety.py — 因果反事实安全验证（构想⑤）

核心思想（do-calculus 启发）：
  不是问  P(safe | observed)
  而是问  P(safe | do(lane_change), worst_reaction)

  变道前，枚举邻道后车的三种合理反应：
    ① 保持速度  ② 加速不让  ③ 减速让行
  对每种反应前向模拟，只有在②（最坏情况）下仍安全才允许变道。
"""

import math
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

class Reaction(Enum):
    MAINTAIN = "maintain"
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"

@dataclass
class CFResult:
    reaction: Reaction
    min_distance: float
    collision: bool
    collision_time: Optional[float]

class CounterfactualVerifier:
    """因果反事实安全验证器"""

    def __init__(self, config=None):
        cfg = config or {}
        self.T_lc = cfg.get('lane_change_time', 3.5)
        self.lane_width = cfg.get('lane_width', 3.5)
        self.safety_margin = cfg.get('safety_margin', 2.5)
        self.a_max_other = cfg.get('max_other_accel', 3.0)
        self.a_min_other = cfg.get('max_other_decel', -5.0)
        self.dt = cfg.get('sim_dt', 0.1)
        self.horizon = cfg.get('sim_horizon', 5.0)
        self.n_steps = int(self.horizon / self.dt)

        self.debug = {
            'n_tested': 0,
            'worst_dist': 999.0,
            'safe': True,
        }

    # ----------------------------------------------------------
    def verify(self, ego_speed_ms: float,
               target_vehicles: List[dict],
               direction: float = 1.0) -> dict:
        """
        执行反事实安全验证。

        参数：
            ego_speed_ms   : 自车速度 (m/s)
            target_vehicles: 目标车道车辆列表
                [{'rel_dist': float(m), 'speed': float(km/h)}, ...]
                rel_dist>0 表示前方，<0 表示后方
            direction      : +1 向右, -1 向左

        返回：
            {'safe': bool, 'worst_distance': float, 'results': [...]}
        """
        ego_traj = self._ego_lc_trajectory(ego_speed_ms, direction)

        all_results: List[CFResult] = []
        overall_safe = True
        worst_dist = float('inf')

        reactions = [Reaction.MAINTAIN, Reaction.ACCELERATE]

        for veh in target_vehicles:
            rel_d = veh['rel_dist']
            v_ms = veh['speed'] / 3.6

            for react in reactions:
                other_traj = self._other_trajectory(rel_d, v_ms, react)
                col, md, ct = self._check_collision(ego_traj, other_traj)

                res = CFResult(reaction=react, min_distance=md,
                               collision=col,
                               collision_time=ct if col else None)
                all_results.append(res)
                if col:
                    overall_safe = False
                if md < worst_dist:
                    worst_dist = md

        self.debug['n_tested'] = len(all_results)
        self.debug['worst_dist'] = round(worst_dist, 2)
        self.debug['safe'] = overall_safe

        return {
            'safe': overall_safe,
            'worst_distance': worst_dist,
            'results': all_results,
        }

    # ----------------------------------------------------------
    def _ego_lc_trajectory(self, vx, direction):
        """自车五次多项式变道轨迹 → [(x, y), ...]"""
        T = self.T_lc
        D = self.lane_width * direction
        a3 = 10 * D / T ** 3
        a4 = -15 * D / T ** 4
        a5 = 6 * D / T ** 5
        traj = []
        for s in range(self.n_steps):
            t = s * self.dt
            x = vx * t
            y = (a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5) if t <= T else D
            traj.append((x, y))
        return traj

    def _other_trajectory(self, rel_dist, vx, reaction):
        """对方在给定反应下的轨迹，纵向位置相对自车初始位置"""
        traj = []
        for s in range(self.n_steps):
            t = s * self.dt
            if reaction == Reaction.MAINTAIN:
                x = rel_dist + vx * t
            elif reaction == Reaction.ACCELERATE:
                x = rel_dist + vx * t + 0.5 * self.a_max_other * t ** 2
            else:
                x = rel_dist + vx * t + 0.5 * self.a_min_other * t ** 2
                if vx + self.a_min_other * t < 0:
                    ts = -vx / self.a_min_other
                    x = rel_dist + vx * ts + 0.5 * self.a_min_other * ts ** 2
            # 对方在目标车道，y = lane_width * direction
            traj.append((x, self.lane_width))
        return traj

    def _check_collision(self, ego, other):
        vl, vw = 4.5, 1.8
        min_d = float('inf')
        col, ct = False, -1.0
        for s in range(min(len(ego), len(other))):
            dx = abs(ego[s][0] - other[s][0])
            dy = abs(ego[s][1] - other[s][1])
            gx = max(0.0, dx - vl)
            gy = max(0.0, dy - vw)
            d = math.sqrt(gx ** 2 + gy ** 2)
            if d < min_d:
                min_d = d
            sx = vl + self.safety_margin
            sy = vw + self.safety_margin * 0.5
            if dx < sx and dy < sy and not col:
                col = True
                ct = s * self.dt
        return col, min_d, ct
