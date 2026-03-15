"""
entropy_decision.py — 信息熵驱动的变道时机调制器（构想③）

核心思想：
  Shannon 信息熵衡量"不确定性"。
  周围车辆行为越混乱（频繁加减速），熵越高 → 压制变道意愿。
  周围车辆行为越稳定，熵越低 → 允许变道。

  effective_desire = fuzzy_desire × entropy_confidence

与模糊推理正交互补：
  模糊回答"想不想变道"   → desire
  熵回答  "现在适不适合"  → modulation factor
"""

import numpy as np
from collections import deque
from typing import List, Dict

class TrafficEntropyAnalyzer:
    """交通环境信息熵分析器"""

    def __init__(self, config=None):
        cfg = config or {}
        self.window_size = cfg.get('window_size', 60)       # 帧, 3s@20Hz
        self.H_threshold = cfg.get('entropy_threshold', 0.8)
        self.accel_lo = cfg.get('accel_lo', -1.5)            # m/s²
        self.accel_hi = cfg.get('accel_hi',  1.5)

        self.speed_history: Dict[int, deque] = {}
        self.behavior_window = deque(maxlen=self.window_size)
        self.entropy_history = deque(maxlen=30)

        self.debug = {
            'current_entropy': 0.0,
            'confidence': 1.0,
            'behavior_dist': [0.0, 0.0, 0.0],
        }

    # ----------------------------------------------------------
    def update(self, surrounding_vehicles: List[dict],
               dt: float = 0.05) -> float:
        """
        每帧调用，更新交通信息熵。

        参数：
            surrounding_vehicles : [{'actor_id': int, 'speed': float(km/h)}, ...]
            dt : 帧间隔

        返回：
            H ∈ [0, log₂3 ≈ 1.585]
        """
        frame_behaviors = []

        for veh in surrounding_vehicles:
            vid = veh['actor_id']
            spd = veh['speed'] / 3.6  # → m/s

            if vid not in self.speed_history:
                self.speed_history[vid] = deque(maxlen=self.window_size)
                self.speed_history[vid].append(spd)
                continue

            prev = self.speed_history[vid][-1]
            accel = (spd - prev) / dt if dt > 0 else 0.0
            self.speed_history[vid].append(spd)

            if accel < self.accel_lo:
                frame_behaviors.append(0)   # 减速
            elif accel > self.accel_hi:
                frame_behaviors.append(2)   # 加速
            else:
                frame_behaviors.append(1)   # 匀速

        counts = [0, 0, 0]
        for b in frame_behaviors:
            counts[b] += 1
        self.behavior_window.append(counts)

        if not frame_behaviors:
            self.entropy_history.append(0.0)
            self._update_debug(0.0)
            return 0.0

        total_counts = np.zeros(3)
        for fc in self.behavior_window:
            total_counts += np.array(fc, dtype=float)
        total = total_counts.sum()
        if total < 1:
            self._update_debug(0.0)
            return 0.0

        probs = total_counts / total
        H = -sum(p * np.log2(p) for p in probs if p > 1e-10)

        if self.entropy_history:
            H = 0.7 * H + 0.3 * self.entropy_history[-1]
        self.entropy_history.append(H)

        self._update_debug(H, probs)
        return H

    # ----------------------------------------------------------
    def get_modulation_factor(self) -> float:
        """
        返回变道意愿调制因子 ∈ [0, 1]。
        熵低→1，熵高→0。
        """
        return self.debug['confidence']

    def is_predictable(self) -> bool:
        return self.debug['confidence'] > 0.5

    def cleanup_stale(self, active_ids: set):
        stale = [v for v in self.speed_history if v not in active_ids]
        for v in stale:
            del self.speed_history[v]

    # ----------------------------------------------------------
    def _update_debug(self, H, probs=None):
        H_max = np.log2(3)
        conf = max(0.0, 1.0 - H / H_max) if H_max > 0 else 1.0
        self.debug['current_entropy'] = round(H, 4)
        self.debug['confidence'] = round(conf, 4)
        if probs is not None:
            self.debug['behavior_dist'] = [round(p, 3) for p in probs]
