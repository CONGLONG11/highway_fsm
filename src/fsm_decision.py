"""
fsm_decision.py — 有限状态机决策模块

集成创新：
  1. 模糊意愿度（原有）
  2. 信息熵调制（构想③）     — 混乱环境压制变道
  3. 反事实安全验证（构想⑤） — 模拟最坏反应
  4. 相空间安全包络（构想②） — 替代简单距离阈值
"""

import math
import time
from enum import Enum, auto

class State(Enum):
    KEEP_LANE = auto()
    PREP_LEFT = auto()
    PREP_RIGHT = auto()
    CHANGE_LEFT = auto()
    CHANGE_RIGHT = auto()
    ABORT = auto()

class FSMDecision:
    """分层决策状态机"""

    def __init__(self, config=None):
        cfg = config or {}
        dec = cfg.get('decision', {})

        self.state = State.KEEP_LANE
        self.target_lane_id = None
        self._state_entry_time = time.time()

        # ---- 阈值 ----
        self.desire_threshold = dec.get('desire_threshold', 0.55)
        self.prep_duration = dec.get('prep_duration', 1.0)
        self.abort_duration = dec.get('abort_duration', 1.5)
        self.lc_timeout = dec.get('lane_change_timeout', 6.0)
        self.safe_front_dist = dec.get('safe_front_dist', 15.0)
        self.safe_rear_dist = dec.get('safe_rear_dist', 12.0)

        # ---- 模糊引擎（外部注入）----
        self.fuzzy_engine = None

        # ---- 信息熵分析器（构想③）----
        from entropy_decision import TrafficEntropyAnalyzer
        entropy_cfg = cfg.get('entropy', {})
        self.entropy_analyzer = TrafficEntropyAnalyzer(entropy_cfg)
        self.use_entropy = dec.get('use_entropy', True)

        # ---- 反事实验证器（构想⑤）----
        from counterfactual_safety import CounterfactualVerifier
        cf_cfg = cfg.get('counterfactual', {})
        self.cf_verifier = CounterfactualVerifier(cf_cfg)
        self.use_counterfactual = dec.get('use_counterfactual', True)

        # ---- 相空间安全包络（构想②）----
        from phase_space_safety import PhaseSpaceSafetyEnvelope
        ps_cfg = cfg.get('phase_space', {})
        self.phase_safety = PhaseSpaceSafetyEnvelope(ps_cfg)
        self.use_phase_space = dec.get('use_phase_space', True)

        # ---- 调试 ----
        self.debug_info = {
            'state': 'KEEP_LANE',
            'desire_raw': 0.0,
            'desire_effective': 0.0,
            'entropy': 0.0,
            'entropy_factor': 1.0,
            'cf_safe': True,
            'cf_worst_dist': 999.0,
            'target_lane': None,
        }

    def set_fuzzy_engine(self, engine):
        self.fuzzy_engine = engine

    # ==============================================================
    #                       主更新入口
    # ==============================================================
    def update(self, ego_data, surrounding):
        """
        每帧调用。

        参数：
            ego_data    : dict — 自车信息
                {'speed': float(km/h), 'waypoint': carla.Waypoint, ...}
            surrounding : dict — 按车道偏移分组的周围车辆
                {0: [当前车道车辆], -1: [左车道], 1: [右车道]}
                每辆车 {'actor_id': int, 'speed': float(km/h),
                        'rel_dist': float(m), ...}

        返回：
            dict — {'state': State, 'target_lane_id': int or None}
        """
        # ---- 更新信息熵 ----
        if self.use_entropy:
            all_vehs = []
            for lane_vehs in surrounding.values():
                all_vehs.extend(lane_vehs)
            self.entropy_analyzer.update(all_vehs)
            self.debug_info['entropy'] = self.entropy_analyzer.debug['current_entropy']
            self.debug_info['entropy_factor'] = self.entropy_analyzer.get_modulation_factor()

        # ---- 状态机分发 ----
        handlers = {
            State.KEEP_LANE: self._handle_keep_lane,
            State.PREP_LEFT: self._handle_prep,
            State.PREP_RIGHT: self._handle_prep,
            State.CHANGE_LEFT: self._handle_change,
            State.CHANGE_RIGHT: self._handle_change,
            State.ABORT: self._handle_abort,
        }
        handler = handlers.get(self.state, self._handle_keep_lane)
        handler(ego_data, surrounding)

        self.debug_info['state'] = self.state.name
        self.debug_info['target_lane'] = self.target_lane_id

        return {
            'state': self.state,
            'target_lane_id': self.target_lane_id,
        }

    # ==============================================================
    #                     KEEP_LANE 状态
    # ==============================================================
    def _handle_keep_lane(self, ego, surr):
        front = self._get_front_vehicle(0, surr)

        if front is None:
            self.debug_info['desire_raw'] = 0.0
            self.debug_info['desire_effective'] = 0.0
            return  # 前方无车，保持车道

        fd = front['rel_dist']
        ds = ego['speed'] - front['speed']

        # ---- Step 1: 模糊意愿度 ----
        if self.fuzzy_engine:
            desire = self.fuzzy_engine.compute_desire(fd, ds)
        else:
            desire = 0.0
        self.debug_info['desire_raw'] = round(desire, 3)

        # ---- Step 2: 信息熵调制（构想③）----
        if self.use_entropy:
            factor = self.entropy_analyzer.get_modulation_factor()
            desire *= factor
        self.debug_info['desire_effective'] = round(desire, 3)

        # ---- Step 3: 意愿度检查 ----
        if desire < self.desire_threshold:
            return

        # ---- Step 4: 选择目标车道 ----
        left_ok = self._evaluate_lane(-1, surr, ego)
        right_ok = self._evaluate_lane(1, surr, ego)

        if left_ok and right_ok:
            left_q = self._lane_quality(-1, surr, ego)
            right_q = self._lane_quality(1, surr, ego)
            chosen = -1 if left_q >= right_q else 1
        elif left_ok:
            chosen = -1
        elif right_ok:
            chosen = 1
        else:
            return  # 两侧都不可行

        self._transition(State.PREP_LEFT if chosen == -1 else State.PREP_RIGHT)
        wp = ego['waypoint']
        target_wp = wp.get_left_lane() if chosen == -1 else wp.get_right_lane()
        self.target_lane_id = target_wp.lane_id if target_wp else None

    # ==============================================================
    #                     PREP 状态
    # ==============================================================
    def _handle_prep(self, ego, surr):
        elapsed = time.time() - self._state_entry_time
        direction = -1 if self.state == State.PREP_LEFT else 1

        # 二次安全确认
        if not self._evaluate_lane(direction, surr, ego):
            self._transition(State.ABORT)
            return

        if elapsed >= self.prep_duration:
            new_state = State.CHANGE_LEFT if direction == -1 else State.CHANGE_RIGHT
            self._transition(new_state)

    # ==============================================================
    #                    CHANGE 状态
    # ==============================================================
    def _handle_change(self, ego, surr):
        elapsed = time.time() - self._state_entry_time
        direction = -1 if self.state == State.CHANGE_LEFT else 1

        # 超时检查
        if elapsed > self.lc_timeout:
            self._transition(State.KEEP_LANE)
            self.target_lane_id = None
            return

        # 到达目标车道？
        wp = ego['waypoint']
        if self.target_lane_id is not None and wp.lane_id == self.target_lane_id:
            self._transition(State.KEEP_LANE)
            self.target_lane_id = None

    # ==============================================================
    #                     ABORT 状态
    # ==============================================================
    def _handle_abort(self, ego, surr):
        elapsed = time.time() - self._state_entry_time
        if elapsed >= self.abort_duration:
            self._transition(State.KEEP_LANE)
            self.target_lane_id = None

    # ==============================================================
    #                   车道评估（三层安全）
    # ==============================================================
    def _evaluate_lane(self, direction, surr, ego):
        """
        综合三层安全检查：
          Layer 1: 距离硬约束（原有）
          Layer 2: 相空间安全包络（构想②）
          Layer 3: 反事实安全验证（构想⑤）

        全部通过才允许变道。
        """
        # --- 车道存在性检查 ---
        wp = ego['waypoint']
        target_wp = wp.get_left_lane() if direction == -1 else wp.get_right_lane()
        if target_wp is None:
            return False
        # 检查是否是同方向车道
        if target_wp.lane_type != wp.lane_type:
            return False

        target_vehs = surr.get(direction, [])

        front = self._get_front_vehicle(direction, surr)
        rear = self._get_rear_vehicle(direction, surr)

        # ---- Layer 1: 距离硬约束 ----
        if front and front['rel_dist'] < self.safe_front_dist:
            return False
        if rear and abs(rear['rel_dist']) < self.safe_rear_dist:
            return False

        # ---- Layer 2: 相空间安全包络（构想②）----
        if self.use_phase_space:
            fd = front['rel_dist'] if front else 200.0
            fs = front['speed'] if front else ego['speed']
            rd = abs(rear['rel_dist']) if rear else 200.0
            rs = rear['speed'] if rear else ego['speed']

            ps_result = self.phase_safety.check_lane_change(
                fd, fs, rd, rs, ego['speed']
            )
            if not ps_result['overall_safe']:
                return False

        # ---- Layer 3: 反事实安全验证（构想⑤）----
        if self.use_counterfactual and target_vehs:
            cf_result = self.cf_verifier.verify(
                ego_speed_ms=ego['speed'] / 3.6,
                target_vehicles=target_vehs,
                direction=float(direction),
            )
            self.debug_info['cf_safe'] = cf_result['safe']
            self.debug_info['cf_worst_dist'] = cf_result['worst_distance']
            if not cf_result['safe']:
                return False

        return True

    # ==============================================================
    #                       辅助函数
    # ==============================================================
    def _lane_quality(self, direction, surr, ego):
        """车道质量评分（用于左右二选一）"""
        front = self._get_front_vehicle(direction, surr)
        if front is None:
            return 1.0
        dist_score = min(front['rel_dist'] / 100.0, 1.0)
        speed_score = min(front['speed'] / ego['speed'], 1.0) if ego['speed'] > 0 else 1.0
        return 0.5 * dist_score + 0.5 * speed_score

    def _get_front_vehicle(self, lane_offset, surr):
        """获取指定车道偏移中的最近前车"""
        vehs = surr.get(lane_offset, [])
        fronts = [v for v in vehs if v['rel_dist'] > 0]
        return min(fronts, key=lambda v: v['rel_dist']) if fronts else None

    def _get_rear_vehicle(self, lane_offset, surr):
        """获取指定车道偏移中的最近后车"""
        vehs = surr.get(lane_offset, [])
        rears = [v for v in vehs if v['rel_dist'] < 0]
        return max(rears, key=lambda v: v['rel_dist']) if rears else None

    def _transition(self, new_state):
        self.state = new_state
        self._state_entry_time = time.time()
