"""
人工势场计算模块 (Artificial Potential Field)
================================================
创新点 2 的核心实现。

将周围车辆建模为"排斥力源"，目标车道中心线建模为"引力源"，
通过势场梯度修正五次多项式生成的基准轨迹，使变道路径能
主动避开高风险区域。

势场模型:
  - 引力场: U_att = 0.5 * k_att * d_goal^2
  - 斥力场: U_rep = 0.5 * k_rep * (1/d_obs - 1/d0)^2  (当 d_obs < d0)
  - 总势场: U = U_att + Σ U_rep
  - 修正力: F = -∇U
"""
import numpy as np
import math

class RiskField:
    """人工势场风险评估器"""

    def __init__(self, config=None):
        if config is None:
            config = {}

        self.k_attract = config.get('k_attract', 1.0)
        self.k_repulse = config.get('k_repulse', 80.0)
        self.repulse_range = config.get('repulse_range', 25.0)      # d0
        self.safe_dist = config.get('repulse_safe_dist', 5.0)
        self.correction_gain = config.get('apf_correction_gain', 0.3)

    def compute_attractive_force(self, current_pos, goal_pos):
        """
        计算引力 (指向目标车道中心)
        F_att = -k_att * (current - goal)

        :param current_pos: np.array([x, y])
        :param goal_pos: np.array([x, y])
        :return: force np.array([fx, fy])
        """
        diff = current_pos - goal_pos
        dist = np.linalg.norm(diff) + 1e-6
        # 限制引力大小，避免距离过远时引力过大
        force = -self.k_attract * diff
        max_force = 10.0
        force_mag = np.linalg.norm(force)
        if force_mag > max_force:
            force = force / force_mag * max_force
        return force

    def compute_repulsive_force(self, current_pos, obstacle_pos, obstacle_vel=None):
        """
        计算单个障碍物的斥力

        标准 APF:
          F_rep = k_rep * (1/d - 1/d0) * (1/d^2) * (pos - obs) / |pos - obs|
          当 d < d0 时有效

        改进: 考虑障碍物速度方向，对迎面来的车辆增加权重

        :param current_pos: np.array([x, y])
        :param obstacle_pos: np.array([x, y])
        :param obstacle_vel: np.array([vx, vy]) 或 None
        :return: force np.array([fx, fy])
        """
        diff = current_pos - obstacle_pos
        dist = np.linalg.norm(diff)

        if dist > self.repulse_range or dist < 0.1:
            return np.array([0.0, 0.0])

        d = max(dist, self.safe_dist)
        d0 = self.repulse_range

        # 标准斥力公式
        magnitude = self.k_repulse * (1.0 / d - 1.0 / d0) * (1.0 / (d * d))

        # 方向: 从障碍物指向当前点
        direction = diff / (dist + 1e-6)

        force = magnitude * direction

        # === 改进: 速度感知的斥力增强 ===
        # 如果障碍物正朝着我运动，增加斥力
        if obstacle_vel is not None:
            obs_speed = np.linalg.norm(obstacle_vel)
            if obs_speed > 0.5:
                # 计算障碍物速度方向与 "障碍物→自车" 方向的夹角余弦
                vel_dir = obstacle_vel / (obs_speed + 1e-6)
                approach_cos = np.dot(vel_dir, direction)

                # 如果 cos > 0，说明障碍物正在靠近，增强斥力
                if approach_cos > 0:
                    # 速度越快、角度越正对，增强越大
                    speed_factor = 1.0 + 0.5 * obs_speed * approach_cos
                    force *= speed_factor

        return force

    def compute_total_field(self, current_pos, goal_pos, obstacles):
        """
        计算总合力

        :param current_pos: np.array([x, y])
        :param goal_pos: np.array([x, y]) 目标车道位置
        :param obstacles: list of dict:
            [{'pos': np.array([x,y]), 'vel': np.array([vx,vy])}, ...]
        :return: total_force np.array([fx, fy]), risk_score float
        """
        # 引力
        f_att = self.compute_attractive_force(current_pos, goal_pos)

        # 斥力叠加
        f_rep_total = np.array([0.0, 0.0])
        risk_score = 0.0

        for obs in obstacles:
            f_rep = self.compute_repulsive_force(
                current_pos, obs['pos'], obs.get('vel', None)
            )
            f_rep_total += f_rep

            # 计算风险评分 (用于评估车道安全性)
            dist = np.linalg.norm(current_pos - obs['pos'])
            if dist < self.repulse_range:
                risk_score += (self.repulse_range - dist) / self.repulse_range

        total_force = f_att + f_rep_total
        return total_force, risk_score

    def correct_trajectory_point(self, traj_point, goal_pos, obstacles):
        """
        利用势场修正单个轨迹点

        这是创新点2的核心接口: 对五次多项式生成的每个轨迹点，
        施加势场修正，使其远离障碍物。

        :param traj_point: np.array([x, y]) 原始轨迹点
        :param goal_pos: np.array([x, y])
        :param obstacles: 障碍物列表
        :return: corrected_point np.array([x, y])
        """
        force, _ = self.compute_total_field(traj_point, goal_pos, obstacles)

        # 修正: 沿合力方向微调轨迹点
        # 只取横向分量 (垂直于行驶方向的分量) 修正
        correction = self.correction_gain * force

        # 限制修正幅度，防止轨迹变形过大
        max_correction = 2.0  # 最大修正 2m
        correction_mag = np.linalg.norm(correction)
        if correction_mag > max_correction:
            correction = correction / correction_mag * max_correction

        return traj_point + correction

    def evaluate_lane_risk(self, lane_center_pos, obstacles):
        """
        评估特定车道的风险分数 (用于决策层的代价函数)

        :param lane_center_pos: np.array([x, y])
        :param obstacles: 该车道上的障碍物列表
        :return: risk_score float (越大越危险)
        """
        total_risk = 0.0
        for obs in obstacles:
            dist = np.linalg.norm(lane_center_pos - obs['pos'])
            if dist < self.repulse_range:
                # 距离越近，风险越高（指数衰减）
                total_risk += np.exp(-dist / 10.0) * 10.0

                # 高速接近加倍
                if obs.get('vel') is not None:
                    rel_vel = obs['vel']
                    approach_speed = np.linalg.norm(rel_vel)
                    if approach_speed > 5.0:
                        total_risk += approach_speed * 0.2

        return total_risk
