# 文件: ~/Highway_FSM/src/controller.py
import carla
import math
import numpy as np
from collections import deque

class VehicleController:
    def __init__(self, vehicle, args_lateral=None, args_longitudinal=None):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        
        # === 修复震荡的核心参数 ===
        # 原参数: K_P=1.2, K_D=0.2 (导致回正过猛，阻尼不足)
        # 新参数: 
        # K_P = 0.8  -> 降低回正的暴躁程度
        # K_D = 0.5  -> 大幅增加阻尼，这是抑制"画龙"的关键！
        # K_I = 0.05 -> 保持低积分，减少稳态误差
        self.args_lat = args_lateral if args_lateral else {'K_P': 1.0, 'K_D': 0.8, 'K_I': 0.05}
        
        # 纵向参数 (油门/刹车) 保持不变，目前工作良好
        self.args_lon = args_longitudinal if args_longitudinal else {'K_P': 1.0, 'K_D': 0.05, 'K_I': 0.05}
        
        self._lon_error_buffer = deque(maxlen=10)
        self._lat_error_buffer = deque(maxlen=10)
        
    def run_step(self, target_speed, target_waypoint, emergency_stop=False):
        """计算控制指令"""
        if emergency_stop:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        current_speed = self._get_speed()
        
        # 1. 纵向控制 (线性刹车优化)
        throttle = self._pid_control(target_speed, current_speed, self.args_lon, self._lon_error_buffer)
        brake = 0.0
        
        speed_diff = current_speed - target_speed
        if speed_diff > 0:
            throttle = 0.0
            # 线性刹车逻辑：超速越多刹得越狠，避免顿挫
            if speed_diff > 10.0: brake = 1.0
            elif speed_diff > 2.0: brake = (speed_diff - 2.0) / 8.0 
        
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        # 2. 横向控制
        current_transform = self.vehicle.get_transform()
        steer = self._lateral_pid_control(target_waypoint, current_transform)
        
        # 3. 动态方向盘限位 (防止高速过敏)
        # 速度越快，允许的方向盘角度越小
        max_steer = 1.0
        if current_speed > 80.0: max_steer = 0.25 # 90km/h时只允许打25%方向
        elif current_speed > 50.0: max_steer = 0.4
        elif current_speed > 30.0: max_steer = 0.6
        
        steer = np.clip(steer, -max_steer, max_steer)
        
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

    def _get_speed(self):
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def _pid_control(self, target, current, args, buffer):
        error = target - current
        buffer.append(error)
        
        if len(buffer) >= 2:
            _de = (buffer[-1] - buffer[-2])
            _ie = sum(buffer)
        else:
            _de = 0.0
            _ie = 0.0
            
        # === 关键修复：积分限幅 (Anti-Windup) ===
        # 防止弯道或变道时积分误差积累过大，导致回到直道后车辆甩尾
        _ie = np.clip(_ie, -5.0, 5.0) 
        
        return (args['K_P'] * error) + (args['K_D'] * _de) + (args['K_I'] * _ie)

    def _lateral_pid_control(self, target_waypoint, vehicle_transform):
        v_begin = vehicle_transform.location
        v_end = target_waypoint.transform.location
        
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        v_vec_norm = np.linalg.norm(v_vec)
        
        # 保护：防止除以零
        if v_vec_norm < 0.1: return 0.0
        
        forward_vec = np.array([
            math.cos(math.radians(vehicle_transform.rotation.yaw)), 
            math.sin(math.radians(vehicle_transform.rotation.yaw)), 0.0])
        
        dot_value = np.dot(v_vec, forward_vec) / v_vec_norm
        dot_value = np.clip(dot_value, -1.0, 1.0)
        alpha = math.acos(dot_value)
        
        cross_value = np.cross(forward_vec, v_vec)
        if cross_value[2] < 0: alpha = -alpha
            
        return self._pid_control(0.0, -alpha, self.args_lat, self._lat_error_buffer)