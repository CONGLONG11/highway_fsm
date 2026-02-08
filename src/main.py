# 文件: ~/Highway_FSM/src/main.py
import carla
import time
import random
import math
import numpy as np
import cv2
import os
import datetime
import threading
import queue
import csv  
from perception import PerceptionModule
from fsm_decision import FSMDecision, State
from controller import VehicleController

# ============================
# 0. 数据日志 (保持不变)
# ============================
class DataLogger:
    def __init__(self, filename="simulation_log.csv"):
        self.filename = filename
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'Frame', 'Time', 'Speed_kmh', 'Target_Speed', 
            'Steer', 'Throttle', 'Brake', 
            'State', 'Lane_ID', 'Front_Dist', 'L_Free', 'R_Free', 'Ego_X', 'Ego_Y'
        ])

    def log(self, frame, sim_time, speed, tgt_speed, steer, throttle, brake, state, lane_id, f_dist, l_free, r_free, x, y):
        self.writer.writerow([
            frame, f"{sim_time:.2f}", f"{speed:.2f}", f"{tgt_speed:.2f}",
            f"{steer:.4f}", f"{throttle:.2f}", f"{brake:.2f}",
            state, lane_id, f"{f_dist:.2f}", l_free, r_free, f"{x:.2f}", f"{y:.2f}"
        ])

    def close(self):
        self.file.close()

# ============================
# 1. 录像模块 (保持不变)
# ============================
class SyncVideoRecorder:
    def __init__(self, vehicle, world, width=800, height=600, fps=20):
        self.vehicle = vehicle
        self.world = world
        self.width = width
        self.height = height
        self.fps = fps
        self.queue = queue.Queue()
        self.recording = True
        self.thread = None
        
        if not os.path.exists("videos"): os.makedirs("videos")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = f"videos/Highway_Fixed_{timestamp}.mp4"
        
        bp_lib = world.get_blueprint_library()
        self.camera_bp = bp_lib.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(width))
        self.camera_bp.set_attribute('image_size_y', str(height))
        self.camera_bp.set_attribute('sensor_tick', str(1.0 / fps))
        # 录像机位也调整得更好看一点
        self.transform = carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-15))

    def start(self):
        self.sensor = self.world.spawn_actor(self.camera_bp, self.transform, attach_to=self.vehicle)
        self.sensor.listen(lambda image: self.queue.put(image))
        self.thread = threading.Thread(target=self._write_loop)
        self.thread.start()

    def _write_loop(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.filepath, fourcc, float(self.fps), (self.width, self.height))
        while self.recording or not self.queue.empty():
            try:
                image = self.queue.get(timeout=1.0)
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                array = array[:, :, :3].copy()
                cv2.putText(array, "FSM AutoPilot", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                writer.write(array)
            except queue.Empty: continue
        writer.release()

    def stop(self):
        self.recording = False
        if self.sensor: self.sensor.stop(); self.sensor.destroy()
        if self.thread: self.thread.join()

# ============================
# 2. 辅助函数
# ============================
def get_front_obstacle_info(lane_id, perception_data, check_dist=100.0):
    min_dist = 999.0
    obs_speed = 0.0
    has_obs = False
    
    if lane_id in perception_data['surrounding']:
        for obj in perception_data['surrounding'][lane_id]:
            if 0 < obj['rel_dist'] < check_dist:
                if abs(obj.get('lat_dist', 0)) < 2.5: 
                    if obj['rel_dist'] < min_dist:
                        min_dist = obj['rel_dist']
                        obs_speed = obj['speed']
                        has_obs = True
    return has_obs, obs_speed, min_dist

def check_lane_safety(target_lane_id, perception_data, ego_speed):
    if target_lane_id not in perception_data['surrounding']: return True
    
    for obj in perception_data['surrounding'][target_lane_id]:
        d = obj['rel_dist']
        rel_speed = obj['rel_speed']
        # 激进一点的安全距离，方便触发换道
        if abs(d) < 8.0: return False # 盲区
        if d > 0 and d < 12.0: return False # 前方太近
        if d < 0:
            if rel_speed > 5.0: 
                ttc = abs(d) / (rel_speed / 3.6)
                if ttc < 2.5: return False
            if abs(d) < 12.0: return False
    return True

# ============================
# 3. 交通流生成 (任务三重点修改)
# ============================
def spawn_traffic(client, world, ego_spawn_transform, num_vehicles=100):
    """
    任务三修复：基于自车位置生成密集车流
    """
    print(f"正在自车附近生成 {num_vehicles} 辆背景车辆...")
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(int(time.time()))
    
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter('vehicle.*')
    vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
    
    all_spawn_points = world.get_map().get_spawn_points()
    
    # === 空间过滤算法 ===
    nearby_points = []
    ego_loc = ego_spawn_transform.location
    
    for sp in all_spawn_points:
        # 计算距离
        dist = sp.location.distance(ego_loc)
        
        # 筛选逻辑：
        # 1. 距离在 5米 到 300米之间 (太近会撞，太远没意义)
        # 2. 尽量筛选同向车道 (简单判断：Yaw 角度差小于 90度)
        if 5.0 < dist < 300.0:
            ego_yaw = ego_spawn_transform.rotation.yaw
            sp_yaw = sp.rotation.yaw
            angle_diff = abs(ego_yaw - sp_yaw)
            while angle_diff > 180: angle_diff -= 360
            while angle_diff < -180: angle_diff += 360
            
            if abs(angle_diff) < 90: # 同向
                nearby_points.append(sp)
    
    print(f"在范围内找到 {len(nearby_points)} 个可用生成点。")
    random.shuffle(nearby_points)
    
    npc_list = []
    # 如果点不够，就尽量生成，不超过 num_vehicles
    count = min(num_vehicles, len(nearby_points))

    for i in range(count):
        transform = nearby_points[i]
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        try:
            actor = world.try_spawn_actor(bp, transform)
            if actor:
                actor.set_autopilot(True)
                # 让背景车辆速度慢一点 (-30% ~ -10%)，迫使自车换道
                tm.vehicle_percentage_speed_difference(actor, random.uniform(10, 30))
                # 忽略红绿灯和停车标志 (高速公路模式)
                tm.ignore_lights_percentage(actor, 100)
                tm.ignore_signs_percentage(actor, 100)
                npc_list.append(actor)
        except: pass
    
    print(f"成功生成 {len(npc_list)} 辆密集背景车。")
    return npc_list

# ============================
# 4. 主程序
# ============================
def main():
    host_ip = '127.0.0.1' 
    client = carla.Client(host_ip, 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    
    if 'Town04' not in world.get_map().name:
        client.load_world('Town04')
        world = client.get_world()

    world.set_weather(carla.WeatherParameters.ClearNoon)

    FPS = 20
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)
    
    ego_vehicle = None; recorder = None; npc_list = []; data_logger = None

    try:
        data_logger = DataLogger("simulation_data_final.csv")
        
        # === 任务二：保持生成点位置不变 ===
        print("正在定位目标生成点 (-515.25, 240.96)...")
        spawn_points = world.get_map().get_spawn_points()
        start_transform = None
        min_dist = 999.0
        target_loc = carla.Location(x=-515.25, y=240.96, z=0.5)
        
        for sp in spawn_points:
            dist = sp.location.distance(target_loc)
            if dist < min_dist:
                min_dist = dist
                start_transform = sp
        
        if min_dist > 5.0:
            print("警告：未找到匹配的SpawnPoint，使用强制坐标。")
            wp = world.get_map().get_waypoint(target_loc)
            start_transform = carla.Transform(target_loc, wp.transform.rotation)
            start_transform.location.z += 0.5
        else:
            print(f"找到匹配点，距离误差: {min_dist:.2f}米")

        ego_bp = world.get_blueprint_library().filter('model3')[0]
        ego_bp.set_attribute('color', '0,220,0')
        ego_vehicle = world.spawn_actor(ego_bp, start_transform)
        
        for _ in range(20): world.tick()

        # === 任务三：传入 ego_transform 进行密集生成 ===
        npc_list = spawn_traffic(client, world, start_transform, num_vehicles=400)

        perception = PerceptionModule(ego_vehicle, world)
        decision = FSMDecision(target_speed=95.0, safety_dist=20.0) 
        controller = VehicleController(ego_vehicle)
        
        recorder = SyncVideoRecorder(ego_vehicle, world, fps=FPS)
        recorder.start()
        
        spectator = world.get_spectator()
        
        print(f"\n=== 仿真开始: 压线修正 & 视角跟随版 ===")
        print(f"修正策略: 极短预瞄 + 密集车流 + 实时追尾视角")
        print("-" * 80)

        is_changing_lane = False; target_lane_id = None    
        frame_count = 0
        total_distance = 0.0; last_print_dist = 0.0
        last_location = ego_vehicle.get_location()

        while True:
            world.tick()
            frame_count += 1
            
            # === 任务二核心修复：视角每帧跟随 (Chase View) ===
            # 获取自车当前的变换矩阵
            ego_trans = ego_vehicle.get_transform()
            # 计算摄像机位置：自车后方 8米，高度 3.5米
            # 使用 get_forward_vector() 的反方向
            camera_loc = ego_trans.location - 8.0 * ego_trans.get_forward_vector() + carla.Location(z=3.5)
            # 摄像机朝向：与车头一致，但在 Pitch 上稍微向下俯视 (-10度)
            camera_rot = ego_trans.rotation
            camera_rot.pitch = -10.0 
            spectator.set_transform(carla.Transform(camera_loc, camera_rot))

            # 里程统计
            current_location = ego_vehicle.get_location()
            total_distance += current_location.distance(last_location)
            last_location = current_location
            
            if total_distance - last_print_dist > 200.0:
                print(f"\n[INFO] 里程: {total_distance:.1f}m | 坐标: ({current_location.x:.1f}, {current_location.y:.1f})")
                last_print_dist = total_distance

            # 感知 & 决策
            perception_data = perception.get_perception_data()
            ego = perception_data['ego']
            ego_wp = ego['waypoint']
            ego_spd = ego['speed']
            current_lane_id = ego['lane_id']
            
            target_state = decision.current_state
            if not is_changing_lane:
                target_state = decision.decide(perception_data)
                
                if target_state == State.LANE_CHANGE_LEFT:
                    left_wp = ego_wp.get_left_lane()
                    if left_wp and left_wp.lane_type == carla.LaneType.Driving:
                        if check_lane_safety(left_wp.lane_id, perception_data, ego_spd):
                             is_changing_lane = True; target_lane_id = left_wp.lane_id; decision.change_lane_cooldown = 100
                    else: decision.current_state = State.KEEP_LANE 

                elif target_state == State.LANE_CHANGE_RIGHT:
                    right_wp = ego_wp.get_right_lane()
                    if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                         if check_lane_safety(right_wp.lane_id, perception_data, ego_spd):
                             is_changing_lane = True; target_lane_id = right_wp.lane_id; decision.change_lane_cooldown = 100
                    else: decision.current_state = State.KEEP_LANE

            if is_changing_lane:
                if current_lane_id == target_lane_id:
                    is_changing_lane = False; target_lane_id = None
                    decision.current_state = State.KEEP_LANE
                    print(f"    >>> 换道完成.")

            # === 任务一核心修复：修正弯道压线 (解决 Chord Error) ===
            # 旧逻辑: lookahead = ego_spd * 0.25 (90km/h -> 22m), 在弯道依然会切内线。
            # 新逻辑: 强制将上限压低至 15m。
            # 即使在 100km/h，也只看 15m。这会让方向盘响应更灵敏，从而紧贴车道线。
            # 下限设为 4.5m 保证低速不抖动。
            lookahead_dist = np.clip(ego_spd * 0.18, 4.5, 15.0)
            
            follow_wp = ego_wp
            if is_changing_lane and target_lane_id:
                if target_lane_id == current_lane_id + 1: 
                    maybe_left = ego_wp.get_left_lane()
                    if maybe_left: follow_wp = maybe_left
                elif target_lane_id == current_lane_id - 1:
                    maybe_right = ego_wp.get_right_lane()
                    if maybe_right: follow_wp = maybe_right
            
            next_wps = follow_wp.next(lookahead_dist)
            aim_wp = next_wps[0] if next_wps else follow_wp

            # ACC 速度规划
            target_speed = 95.0
            check_lanes = [current_lane_id]
            if is_changing_lane and target_lane_id: check_lanes.append(target_lane_id)
            
            min_obs_dist = 999.0; front_car_spd = 0.0
            for lid in check_lanes:
                has_o, o_spd, o_dist = get_front_obstacle_info(lid, perception_data)
                if has_o and o_dist < min_obs_dist: min_obs_dist = o_dist; front_car_spd = o_spd
            
            if min_obs_dist < 60.0:
                if min_obs_dist < 15.0: target_speed = 0.0
                elif min_obs_dist < 35.0: target_speed = max(0, front_car_spd - 5)
                else: target_speed = min(95.0, front_car_spd + 10) 
            
            control = controller.run_step(target_speed, aim_wp, emergency_stop=(target_speed==0))
            ego_vehicle.apply_control(control)

            l_free = perception_data['lanes']['left_available']
            r_free = perception_data['lanes']['right_available']
            current_time = frame_count / FPS
            
            data_logger.log(
                frame_count, current_time, ego_spd, target_speed, 
                control.steer, control.throttle, control.brake,
                decision.current_state.name, current_lane_id, min_obs_dist, l_free, r_free,
                current_location.x, current_location.y
            )
            
            if frame_count % 10 == 0:
                s_str = decision.current_state.name
                if is_changing_lane: s_str = f"CHG -> {target_lane_id}"
                # 打印信息中增加 'Lane' 方便观察是否在切弯
                print(f"\rTime:{current_time:5.1f}s | Spd:{ego_spd:4.1f} | Lookahead:{lookahead_dist:.1f}m | Obs:{min_obs_dist:4.1f}m | {s_str:<12} ", end="")

    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if recorder: recorder.stop()
        if ego_vehicle: ego_vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_list])
        if data_logger: data_logger.close() 

if __name__ == '__main__':
    main()