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
# 0. 数据日志
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
# 1. 录像模块
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
        self.transform = carla.Transform(carla.Location(x=-6, z=4), carla.Rotation(pitch=-15))

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
                cv2.putText(array, "FSM AutoPilot (Fixed)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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
        if abs(d) < 10.0: return False # 盲区
        if d > 0 and d < 15.0: return False # 前方太近
        if d < 0:
            if rel_speed > 5.0: 
                ttc = abs(d) / (rel_speed / 3.6)
                if ttc < 3.0: return False
            if abs(d) < 15.0: return False
    return True

# ============================
# 3. 交通流生成
# ============================
def spawn_traffic(client, world, num_vehicles=30):
    print(f"正在生成 {num_vehicles} 辆背景车辆...")
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(0)
    
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter('vehicle.*')
    vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
    spawn_points = world.get_map().get_spawn_points()
    
    npc_list = []
    random.shuffle(spawn_points)

    for n, transform in enumerate(spawn_points):
        if n >= num_vehicles: break
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        try:
            actor = world.try_spawn_actor(bp, transform)
            if actor:
                actor.set_autopilot(True)
                tm.vehicle_percentage_speed_difference(actor, random.choice([-20, -10, 0, 10, 20]))
                npc_list.append(actor)
        except: pass
    
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
        
        # === 任务二：保持目前的自车生成点位置不变 ===
        # 根据日志，原起点坐标为 X:-515.25, Y:240.96
        # 我们遍历所有 SpawnPoints，找到最接近这个坐标的点
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
        
        # 如果找不到特别近的，就强制使用坐标生成（可能在半空中，需小心）
        if min_dist > 5.0:
            print("警告：未找到匹配的SpawnPoint，使用强制坐标。")
            start_transform = carla.Transform(target_loc, carla.Rotation(yaw=0)) # Yaw需根据道路调整
            # 修正 Yaw: 获取最近的路点朝向
            wp = world.get_map().get_waypoint(target_loc)
            start_transform.rotation = wp.transform.rotation
            start_transform.location.z += 0.5
        else:
            print(f"找到匹配点，距离误差: {min_dist:.2f}米")

        ego_bp = world.get_blueprint_library().filter('model3')[0]
        ego_bp.set_attribute('color', '0,220,0')
        ego_vehicle = world.spawn_actor(ego_bp, start_transform)
        
        for _ in range(20): world.tick()

        npc_list = spawn_traffic(client, world, num_vehicles=50)

        perception = PerceptionModule(ego_vehicle, world)
        decision = FSMDecision(target_speed=95.0, safety_dist=20.0) 
        controller = VehicleController(ego_vehicle)
        
        recorder = SyncVideoRecorder(ego_vehicle, world, fps=FPS)
        recorder.start()
        
        print(f"\n=== 仿真开始: 修复切弯撞墙Bug版 ===")
        print(f"修正策略: 缩短横向控制预瞄距离")
        print("-" * 80)

        is_changing_lane = False; target_lane_id = None    
        frame_count = 0
        total_distance = 0.0; last_print_dist = 0.0
        last_location = ego_vehicle.get_location()

        while True:
            world.tick()
            frame_count += 1
            
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
                    else: decision.current_state = State.KEEP_LANE # 物理不可达

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

            # === 任务一核心修复：路径规划 (Lookahead) ===
            # 之前是 ego_spd * 0.5 (90km/h时=45m)。这在内侧车道会导致切过护栏。
            # 修复：大幅缩短横向控制的预瞄距离，强制车辆贴合当前弯道几何。
            # 新逻辑：20km/h -> 6m, 90km/h -> 22m。
            lookahead_dist = np.clip(ego_spd * 0.25, 6.0, 25.0)
            
            follow_wp = ego_wp
            if is_changing_lane and target_lane_id:
                if target_lane_id == current_lane_id + 1: # OpenDRIVE negative IDs
                    maybe_left = ego_wp.get_left_lane()
                    if maybe_left: follow_wp = maybe_left
                elif target_lane_id == current_lane_id - 1:
                    maybe_right = ego_wp.get_right_lane()
                    if maybe_right: follow_wp = maybe_right
            
            # 获取预瞄点
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

            # 日志
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
                print(f"\rTime:{current_time:5.1f}s | Spd:{ego_spd:4.1f} | Lookahead:{lookahead_dist:.1f}m | Dist:{min_obs_dist:4.1f}m | {s_str:<12} | Loc:({current_location.x:.0f},{current_location.y:.0f})", end="")

    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if recorder: recorder.stop()
        if ego_vehicle: ego_vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_list])
        if data_logger: data_logger.close() 

if __name__ == '__main__':
    main()