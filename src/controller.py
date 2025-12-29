# 文件: ~/Highway_FSM/src/controller.py
import carla
import math
import numpy as np
from collections import deque

class VehicleController:
    def __init__(self, vehicle, args_lateral=None, args_longitudinal=None):
        self.vehicle = vehicle
        
        # === 针对转弯优化的 PID ===
        # K_P 1.5: 提高入弯响应
        # K_D 0.3: 提高出弯稳定性 (防抖)
        self.args_lat = args_lateral if args_lateral else {'K_P': 1.5, 'K_D': 0.3, 'K_I': 0.1}
        self.args_lon = args_longitudinal if args_longitudinal else {'K_P': 1.0, 'K_D': 0.05, 'K_I': 0.05}
        
        self._lon_error_buffer = deque(maxlen=10)
        self._lat_error_buffer = deque(maxlen=10)
        
    def run_step(self, target_speed, target_waypoint, emergency_stop=False):
        if emergency_stop:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        current_speed = self._get_speed()
        
        # 1. 线性刹车控制
        throttle = self._pid_control(target_speed, current_speed, self.args_lon, self._lon_error_buffer)
        brake = 0.0
        
        speed_diff = current_speed - target_speed
        if speed_diff > 0:
            throttle = 0.0
            if speed_diff > 10.0: brake = 1.0   # 重刹
            elif speed_diff > 1.0: brake = (speed_diff - 1.0) / 9.0 # 线性轻刹
        
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        # 2. 横向控制
        current_transform = self.vehicle.get_transform()
        steer = self._lateral_pid_control(target_waypoint, current_transform)
        
        # 3. 智能方向盘限制
        # 速度越快，限制越严；但如果真的要转大弯 (目标偏差极大)，适当放宽限制
        max_steer = 1.0
        if current_speed > 80.0: max_steer = 0.35
        elif current_speed > 50.0: max_steer = 0.5 
        
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
        else: _de = 0.0; _ie = 0.0
        _ie = np.clip(_ie, -10.0, 10.0) # 积分限幅
        return (args['K_P'] * error) + (args['K_D'] * _de) + (args['K_I'] * _ie)

    def _lateral_pid_control(self, target_waypoint, vehicle_transform):
        v_begin = vehicle_transform.location
        v_end = target_waypoint.transform.location
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        v_vec_norm = np.linalg.norm(v_vec)
        
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