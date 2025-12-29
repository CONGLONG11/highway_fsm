# 文件: ~/Highway_FSM/src/fsm_decision.py
from enum import Enum

class State(Enum):
    KEEP_LANE = 0
    LANE_CHANGE_LEFT = 1
    LANE_CHANGE_RIGHT = 2

class FSMDecision:
    def __init__(self, target_speed=100.0, safety_dist=20.0):
        self.current_state = State.KEEP_LANE
        self.target_speed = target_speed
        self.safety_dist = safety_dist
        self.change_lane_cooldown = 0 

    def decide(self, perception_data):
        ego = perception_data['ego']
        surrounding = perception_data['surrounding']
        lanes = perception_data['lanes']
        current_lane_id = ego['lane_id']
        
        if self.change_lane_cooldown > 0: self.change_lane_cooldown -= 1
        
        next_state = self.current_state
        
        if self.current_state == State.KEEP_LANE:
            # 1. 检查是否需要变道
            front_vehicle = self._get_front_vehicle(current_lane_id, surrounding)
            needs_lane_change = False
            
            if front_vehicle:
                # 触发条件：前车慢且近
                if (front_vehicle['rel_dist'] < 40.0 and front_vehicle['speed'] < self.target_speed * 0.8) or \
                   (front_vehicle['rel_dist'] < 60.0 and front_vehicle['speed'] < 50.0):
                    needs_lane_change = True

            if needs_lane_change and self.change_lane_cooldown == 0:
                # Town04/OpenDRIVE: 左侧 ID+1, 右侧 ID-1
                left_lane_id = current_lane_id + 1
                right_lane_id = current_lane_id - 1
                
                # 2. 优先尝试向左
                if lanes['left_available'] and self._check_safety(left_lane_id, surrounding, ego['speed']):
                    next_state = State.LANE_CHANGE_LEFT
                    self.change_lane_cooldown = 150 # 冷却
                    print(f"    >>> 决策: 向左换道 (ID {left_lane_id})")
                
                # 3. 其次尝试向右
                elif next_state == State.KEEP_LANE and lanes['right_available']:
                    if self._check_safety(right_lane_id, surrounding, ego['speed']):
                        next_state = State.LANE_CHANGE_RIGHT
                        self.change_lane_cooldown = 150
                        print(f"    >>> 决策: 向右换道 (ID {right_lane_id})")

        if next_state != self.current_state:
            self.current_state = next_state
            
        return self.current_state

    def _get_front_vehicle(self, current_lane, surrounding):
        if current_lane not in surrounding: return None
        min_dist = 999.0
        closest_front = None
        for obj in surrounding[current_lane]:
            if 0 < obj['rel_dist'] < min_dist:
                min_dist = obj['rel_dist']
                closest_front = obj
        return closest_front

    def _check_safety(self, target_lane_id, surrounding, ego_speed):
        if target_lane_id not in surrounding: return True
        
        for obj in surrounding[target_lane_id]:
            rel_dist = obj['rel_dist']
            lat_dist = obj.get('lat_dist', 10.0)
            rel_speed = obj['rel_speed'] 

            # 1. 盲区检测 (最重要的防撞逻辑)
            # 无论纵向多远，只要横向贴着 (< 2.5m 压线) 或纵向并排 (< 10m)，就不能变
            if abs(rel_dist) < 12.0: return False

            # 2. 前车安全
            required_dist = self.safety_dist
            if rel_speed < -10.0: required_dist += 15.0 # 我比前车快很多
            if rel_dist > 0 and rel_dist < required_dist: return False
            
            # 3. 后车安全 (TTC)
            if rel_dist < 0:
                required_rear = 15.0
                if rel_speed > 2.0: # 后车比我快
                    ttc = abs(rel_dist) / (rel_speed / 3.6)
                    if ttc < 3.5: return False # TTC < 3.5s 危险
                    required_rear = 30.0
                if abs(rel_dist) < required_rear: return False
                
        return True