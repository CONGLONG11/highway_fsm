"""
模糊增强有限状态机决策模块 (Fuzzy-Enhanced FSM Decision)
============================================================
创新点 1: 将模糊逻辑与有限状态机、代价函数三者结合。

工作流程:
  1. 模糊引擎计算"变道意愿度" (连续值 0~1)
  2. 若意愿度超过阈值，触发代价函数评估
  3. 代价函数综合 效率/安全/舒适 三维代价
  4. FSM 根据最低代价选择最优行为

状态转换图:
  ┌───────────┐
  │ KEEP_LANE │ ←─────────────────────────┐
  └─────┬─────┘                           │
        │ 意愿度 > 阈值                    │ 变道完成 / 冷却结束
        │ 且代价评估通过                    │
        ▼                                 │
  ┌──────────────┐    ┌──────────────┐    │
  │ CHANGE_LEFT  │    │ CHANGE_RIGHT │ ───┘
  └──────────────┘    └──────────────┘
"""
from enum import Enum
import numpy as np
import math

from fuzzy_engine import FuzzyLaneChangeEngine
from risk_field import RiskField

class State(Enum):
    KEEP_LANE = 0
    PREPARE_LANE_CHANGE_LEFT = 1    # 准备变道 (触发规划)
    LANE_CHANGE_LEFT = 2             # 正在变道 (跟踪轨迹)
    PREPARE_LANE_CHANGE_RIGHT = 3
    LANE_CHANGE_RIGHT = 4

class FSMDecision:
    def __init__(self, config=None):
        config = config or {}

        self.state = State.KEEP_LANE
        self.target_speed = config.get('target_speed', 100.0)

        # 模糊引擎
        fuzzy_config = config.get('fuzzy', {})
        self.fuzzy_engine = FuzzyLaneChangeEngine(fuzzy_config)
        self.desire_threshold = fuzzy_config.get('desire_threshold', 0.6)

        # 势场风险评估
        apf_config = config.get('apf', {})
        self.risk_field = RiskField(apf_config)

        # 代价权重
        cost_config = config.get('cost', {})
        self.w_efficiency = cost_config.get('w_efficiency', 10.0)
        self.w_safety = cost_config.get('w_safety', 20.0)
        self.w_comfort = cost_config.get('w_comfort', 8.0)
        self.w_lane_bias = cost_config.get('w_lane_bias', 2.0)

        # 安全参数
        safety_config = config.get('safety', {})
        self.min_front_dist = safety_config.get('min_front_dist', 15.0)
        self.min_side_dist = safety_config.get('min_side_dist', 8.0)
        self.min_rear_dist = safety_config.get('min_rear_dist', 12.0)
        self.ttc_threshold = safety_config.get('ttc_threshold', 3.0)

        # 冷却计时
        self.cooldown_max = safety_config.get('lane_change_cooldown', 150)
        self.cooldown_timer = 0

        # 调试信息
        self.debug_info = {}

    def decide(self, perception_data):
        """
        主决策函数

        :param perception_data: 感知模块的输出
        :return: State 枚举值
        """
        ego = perception_data['ego']
        surrounding = perception_data['surrounding']
        lanes = perception_data['lanes']

        # 冷却倒计时
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        # ============================================================
        # 状态机逻辑
        # ============================================================

        if self.state == State.KEEP_LANE:
            return self._handle_keep_lane(ego, surrounding, lanes)

        elif self.state in (State.PREPARE_LANE_CHANGE_LEFT,
                            State.PREPARE_LANE_CHANGE_RIGHT):
            # 准备状态: 等待规划器生成轨迹, 自动跳转到执行
            # (由 main loop 控制)
            return self.state

        elif self.state in (State.LANE_CHANGE_LEFT, State.LANE_CHANGE_RIGHT):
            # 执行中: 由 main loop 监控轨迹完成度来切回 KEEP_LANE
            return self.state

        return State.KEEP_LANE

    def _handle_keep_lane(self, ego, surrounding, lanes):
        """保持车道状态下的决策逻辑"""

        # 如果冷却未结束，不做决策
        if self.cooldown_timer > 0:
            return State.KEEP_LANE

        current_lane = ego['lane_id']

        # ---- Step 1: 模糊推理计算变道意愿度 ----
        front_vehicle = self._get_front_vehicle(current_lane, surrounding)

        if front_vehicle is None:
            front_dist = 999.0
            delta_speed = 0.0
        else:
            front_dist = front_vehicle['rel_dist']
            delta_speed = ego['speed'] - front_vehicle['speed']  # 正值=我更快

        desire = self.fuzzy_engine.evaluate(front_dist, delta_speed)

        self.debug_info['fuzzy_desire'] = desire
        self.debug_info['front_dist'] = front_dist
        self.debug_info['delta_speed'] = delta_speed

        # ---- Step 2: 意愿度检查 ----
        if desire < self.desire_threshold:
            # 不够想变道，保持车道
            return State.KEEP_LANE

        # ---- Step 3: 代价函数评估所有可行方案 ----
        costs = {}

        # 当前车道代价
        costs['keep'] = self._compute_lane_cost(
            current_lane, current_lane, ego, surrounding, is_change=False
        )

        # 左变道代价
        if lanes['left_available']:
            left_lane = lanes['left_wp'].lane_id if lanes['left_wp'] else None
            if left_lane and self._hard_safety_check(left_lane, surrounding, ego):
                costs['left'] = self._compute_lane_cost(
                    left_lane, current_lane, ego, surrounding, is_change=True
                )
            else:
                costs['left'] = float('inf')
        else:
            costs['left'] = float('inf')

        # 右变道代价
        if lanes['right_available']:
            right_lane = lanes['right_wp'].lane_id if lanes['right_wp'] else None
            if right_lane and self._hard_safety_check(right_lane, surrounding, ego):
                costs['right'] = self._compute_lane_cost(
                    right_lane, current_lane, ego, surrounding, is_change=True
                )
            else:
                costs['right'] = float('inf')
        else:
            costs['right'] = float('inf')

        self.debug_info['costs'] = costs

        # ---- Step 4: 选择最优方案 ----
        best = min(costs, key=costs.get)

        if best == 'left' and costs['left'] < costs['keep']:
            self.state = State.PREPARE_LANE_CHANGE_LEFT
            self.cooldown_timer = self.cooldown_max
            print(f"[Decision] >>> 左变道 | desire={desire:.2f} | costs={costs}")
            return self.state

        elif best == 'right' and costs['right'] < costs['keep']:
            self.state = State.PREPARE_LANE_CHANGE_RIGHT
            self.cooldown_timer = self.cooldown_max
            print(f"[Decision] >>> 右变道 | desire={desire:.2f} | costs={costs}")
            return self.state

        return State.KEEP_LANE

    def _compute_lane_cost(self, target_lane, current_lane, ego, surrounding,
                           is_change=False):
        """
        综合代价函数

        Cost = w_eff * C_efficiency + w_safe * C_safety + w_com * C_comfort
        """
        cost = 0.0

        # --- A. 效率代价: 期望速度损失 ---
        front = self._get_front_vehicle(target_lane, surrounding)
        expected_speed = self.target_speed

        if front is not None:
            fd = front['rel_dist']
            if fd < 80.0:
                # 期望速度受限于前车
                expected_speed = min(self.target_speed, front['speed'])
                # 距离越近，限速越严
                if fd < 30.0:
                    expected_speed *= (fd / 30.0)

        speed_loss = (self.target_speed - expected_speed) / self.target_speed
        cost += self.w_efficiency * max(0.0, speed_loss)

        # --- B. 安全代价: 势场风险评分 ---
        ego_pos = np.array([ego['location'].x, ego['location'].y])
        obs_list = []
        if target_lane in surrounding:
            for v in surrounding[target_lane]:
                obs_list.append({
                    'pos': np.array([v['location'].x, v['location'].y]),
                    'vel': np.array([v['velocity'].x, v['velocity'].y]),
                })
        risk = self.risk_field.evaluate_lane_risk(ego_pos, obs_list)
        cost += self.w_safety * risk * 0.1

        # --- C. 舒适代价: 变道惩罚 ---
        if is_change:
            cost += self.w_comfort

        return round(cost, 3)

    def _hard_safety_check(self, target_lane, surrounding, ego):
        """硬性安全检查 (一票否决)"""
        if target_lane not in surrounding:
            return True

        for v in surrounding[target_lane]:
            rd = v['rel_dist']
            rs = v['rel_speed']

            # 盲区
            if abs(rd) < self.min_side_dist:
                return False

            # 前方安全距离
            if 0 < rd < self.min_front_dist:
                return False

            # 后方
            if rd < 0:
                if abs(rd) < self.min_rear_dist:
                    return False
                # TTC 检查
                if rs > 3.0:
                    ttc = abs(rd) / (rs / 3.6)
                    if ttc < self.ttc_threshold:
                        return False

        return True

    def _get_front_vehicle(self, lane_id, surrounding):
        if lane_id not in surrounding:
            return None
        for v in surrounding[lane_id]:
            if v['rel_dist'] > 3.0:
                return v
        return None

    def notify_lane_change_complete(self):
        """由 main loop 调用，通知变道完成"""
        self.state = State.KEEP_LANE
        print("[Decision] 变道完成，回归 KEEP_LANE")

    def notify_lane_change_start(self):
        """由 main loop 调用，通知轨迹已生成，开始执行"""
        if self.state == State.PREPARE_LANE_CHANGE_LEFT:
            self.state = State.LANE_CHANGE_LEFT
        elif self.state == State.PREPARE_LANE_CHANGE_RIGHT:
            self.state = State.LANE_CHANGE_RIGHT
