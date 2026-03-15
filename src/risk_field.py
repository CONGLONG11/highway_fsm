"""
risk_field.py — 人工势场轨迹修正器
================================================
对规划出的变道轨迹进行势场修正：
  - 障碍物产生斥力
  - 轨迹点在斥力作用下横向微调
  - 输出修正后的轨迹
"""

import math
import numpy as np
import carla

class RiskFieldCorrector:
    """人工势场轨迹修正"""

    def __init__(self, config=None):
        cfg = config or {}
        rf = cfg.get('risk_field', {})

        self.eta = rf.get('eta', 50.0)              # 斥力增益
        self.d0 = rf.get('d0', 25.0)                # 斥力影响半径 (m)
        self.lambda_w = rf.get('lambda_weight', 0.3) # 修正权重
        self.kv = rf.get('kv', 0.05)                 # 速度感知系数

    def correct(self, trajectory, obstacles, ego_transform, ego_speed_kmh):
        """
        对轨迹进行势场修正。

        参数：
            trajectory     : list[carla.Transform]
            obstacles      : list[dict] — {'location': carla.Location, 
                                            'velocity': carla.Vector3D,
                                            'speed': float}
            ego_transform  : carla.Transform
            ego_speed_kmh  : float

        返回：
            corrected_traj : list[carla.Transform]
        """
        if not trajectory or not obstacles:
            return trajectory

        corrected = []

        for tf in trajectory:
            px, py = tf.location.x, tf.location.y
            total_fx, total_fy = 0.0, 0.0

            for obs in obstacles:
                ox = obs['location'].x
                oy = obs['location'].y
                dist = math.sqrt((px - ox) ** 2 + (py - oy) ** 2)

                if dist < 0.5:
                    dist = 0.5  # 防除零

                if dist > self.d0:
                    continue

                # 斥力大小
                magnitude = self.eta * (1.0 / dist - 1.0 / self.d0) / (dist ** 2)

                # 斥力方向（从障碍物指向轨迹点）
                dx = (px - ox) / dist
                dy = (py - oy) / dist

                total_fx += magnitude * dx
                total_fy += magnitude * dy

            # 修正量
            new_x = px + self.lambda_w * total_fx
            new_y = py + self.lambda_w * total_fy

            corrected.append(carla.Transform(
                carla.Location(x=new_x, y=new_y, z=tf.location.z),
                tf.rotation
            ))

        return corrected
