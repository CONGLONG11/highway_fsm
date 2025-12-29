# 文件: ~/Highway_FSM/src/main.py
import carla
import time
import random
import math
import numpy as np
from perception import PerceptionModule
from fsm_decision import FSMDecision, State
from controller import VehicleController

def spawn_global_traffic(world, client, ego_spawn_point, num_vehicles=60):
    """全图交通流生成"""
    print(f"正在生成 {num_vehicles} 辆背景车...")
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [x for x in bp_lib.filter('vehicle.*') if int(x.get_attribute('number_of_wheels')) == 4]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.global_percentage_speed_difference(0.0) 
    
    npc_list = []
    count = 0
    for transform in spawn_points:
        if count >= num_vehicles: break
        # 避免在自车出生点附近生成
        if transform.location.distance(ego_spawn_point.location) < 30.0: continue
        
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        
        # 使用 try_spawn_actor 防止崩溃
        npc = world.try_spawn_actor(bp, transform)
        if npc:
            npc.set_autopilot(True)
            r = random.random()
            if r < 0.2: tm.vehicle_percentage_speed_difference(npc, 40)
            elif r < 0.8: tm.vehicle_percentage_speed_difference(npc, random.uniform(-10, 10))
            else: 
                tm.vehicle_percentage_speed_difference(npc, -30)
                tm.auto_lane_change(npc, True)
            npc_list.append(npc)
            count += 1
    return npc_list

def get_front_obstacle_speed(lane_id, perception_data, check_dist=45.0):
    """ACC 检测"""
    min_dist = 999.0
    obs_speed = 0.0
    has_obs = False
    
    if lane_id in perception_data['surrounding']:
        for obj in perception_data['surrounding'][lane_id]:
            if 0 < obj['rel_dist'] < check_dist:
                if obj['rel_dist'] < min_dist:
                    min_dist = obj['rel_dist']
                    obs_speed = obj['speed']
                    has_obs = True
    return has_obs, obs_speed, min_dist

def scan_road_ahead(ego_wp, ego_yaw, max_dist=100.0, step=5.0):
    """
    【核心修复】向前扫描 100米，寻找最大的曲率
    返回: (max_curvature_angle, distance_to_curve)
    """
    max_curve = 0.0
    dist_to_max = 0.0
    
    current_wp = ego_wp
    
    # 向前迭代采样
    for dist in range(0, int(max_dist), int(step)):
        next_wps = current_wp.next(step)
        if not next_wps: 
            break
        current_wp = next_wps[0]
        
        # 计算该点的航向角差异
        road_yaw = current_wp.transform.rotation.yaw
        diff = abs(road_yaw - ego_yaw)
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        diff = abs(diff)
        
        if diff > max_curve:
            max_curve = diff
            dist_to_max = dist + step
            
    return max_curve, dist_to_max

def main():
    host_ip = '127.0.0.1' 
    client = carla.Client(host_ip, 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # 确保加载地图
    if 'Town04' not in world.get_map().name:
        client.load_world('Town04')
        world = client.get_world()
    world.tick()

    npc_list = []
    ego_vehicle = None

    try:
        # --- 车辆生成 ---
        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        
        ego_loc = spawn_points[0] # 这是一个 Transform 对象
        ego_bp = bp_lib.filter('model3')[0]
        ego_bp.set_attribute('color', '255,0,0')
        
        # 尝试生成自车
        ego_vehicle = world.try_spawn_actor(ego_bp, ego_loc)
        if ego_vehicle is None:
            # 如果点0被占用了，换一个点试试
            ego_loc = spawn_points[10]
            ego_vehicle = world.spawn_actor(ego_bp, ego_loc)
        
        # --- 修复点：更安全的障碍车生成逻辑 ---
        # 1. 获取自车所在的 Waypoint
        start_wp = world.get_map().get_waypoint(ego_loc.location)
        # 2. 沿着道路向前找 50 米的点 (保证在路面上)
        front_wps = start_wp.next(50.0)
        
        if front_wps:
            front_trans = front_wps[0].transform
            # 3. 稍微抬高一点点，防止轮子陷进地里导致碰撞
            front_trans.location.z += 1.0 
            
            npc_bp = bp_lib.filter('vehicle.audi.tt')[0]
            # 4. 使用 try_spawn_actor 替代 spawn_actor，避免报错崩溃
            npc_vehicle = world.try_spawn_actor(npc_bp, front_trans)
            
            if npc_vehicle:
                npc_vehicle.set_autopilot(True)
                client.get_trafficmanager().vehicle_percentage_speed_difference(npc_vehicle, 60.0)
                client.get_trafficmanager().auto_lane_change(npc_vehicle, False)
                npc_list.append(npc_vehicle)
                print("前方障碍车生成成功！")
            else:
                print("前方障碍车生成失败 (位置冲突)，跳过。")
        else:
            print("前方无路，无法生成障碍车。")
        
        # 生成背景交通流
        npc_list.extend(spawn_global_traffic(world, client, ego_loc, num_vehicles=70))

        # --- 模块初始化 ---
        perception = PerceptionModule(ego_vehicle, world)
        decision = FSMDecision(target_speed=95.0, safety_dist=20.0) 
        controller = VehicleController(ego_vehicle)
        spectator = world.get_spectator()
        
        print("\n=== 最终防撞版自动驾驶系统 ===")
        print("优化: 远距离曲率预测(100m) + 预测性重刹 + 直角弯通过能力")

        is_changing_lane = False 
        target_lane_id = None    
        
        while True:
            # 1. 感知
            perception_data = perception.get_perception_data()
            ego_wp = perception_data['ego']['waypoint']
            ego_spd = perception_data['ego']['speed']
            current_lane_id = perception_data['ego']['lane_id']
            ego_yaw = ego_vehicle.get_transform().rotation.yaw
            
            # --- 核心修复：远距离路况预测 ---
            # 提前看 100 米
            max_curve, dist_to_curve = scan_road_ahead(ego_wp, ego_yaw, max_dist=100.0)
            
            # 定义弯道等级
            is_sharp_turn = max_curve > 30.0 # 直角弯通常 > 45度
            is_normal_curve = max_curve > 10.0
            
            # 2. 决策 & 状态机
            # 如果是急弯，禁止变道
            can_change_lane = not is_changing_lane and not is_sharp_turn and (dist_to_curve > 40)
            
            if can_change_lane:
                target_state = decision.decide(perception_data)
                
                if target_state == State.LANE_CHANGE_LEFT:
                    left_wp = ego_wp.get_left_lane()
                    safe = decision._check_safety(current_lane_id-1, perception_data['surrounding'], ego_spd)
                    if left_wp and left_wp.lane_type == carla.LaneType.Driving and safe:
                        is_changing_lane = True; target_lane_id = left_wp.lane_id
                
                elif target_state == State.LANE_CHANGE_RIGHT:
                    right_wp = ego_wp.get_right_lane()
                    safe = decision._check_safety(current_lane_id+1, perception_data['surrounding'], ego_spd)
                    if right_wp and right_wp.lane_type == carla.LaneType.Driving and safe:
                        is_changing_lane = True; target_lane_id = right_wp.lane_id
            else:
                if current_lane_id == target_lane_id:
                    is_changing_lane = False
                    target_lane_id = None
                    decision.current_state = State.KEEP_LANE
                    decision.change_lane_cooldown = 50

            # 3. 规划 (Lookahead 策略)
            
            # 基础预瞄
            lookahead_dist = np.clip(ego_spd * 0.3, 6.0, 20.0)
            
            # 弯道修正：如果是急弯，强制看近点 (贴线走)
            if is_sharp_turn and dist_to_curve < 20.0:
                lookahead_dist = 5.0 # 极短视距，为了贴死内线
            elif is_normal_curve:
                lookahead_dist = 8.0

            # 确定目标车道
            follow_lane_wp = ego_wp
            if is_changing_lane and target_lane_id is not None:
                if ego_wp.get_left_lane() and ego_wp.get_left_lane().lane_id == target_lane_id:
                    follow_lane_wp = ego_wp.get_left_lane()
                elif ego_wp.get_right_lane() and ego_wp.get_right_lane().lane_id == target_lane_id:
                    follow_lane_wp = ego_wp.get_right_lane()
                if current_lane_id == target_lane_id: follow_lane_wp = ego_wp
            
            next_wps = follow_lane_wp.next(lookahead_dist)
            aim_wp = next_wps[0] if next_wps else follow_lane_wp
            
            # 4. 速度规划 (关键：预测性限速)
            
            target_speed = 95.0 # 默认高速
            
            # === 弯道限速逻辑 ===
            if is_sharp_turn:
                # 发现前方有直角弯
                if dist_to_curve > 60:
                    target_speed = 60.0 # 提前减速
                elif dist_to_curve > 30:
                    target_speed = 40.0 # 临近重刹
                else:
                    target_speed = 30.0 # 入弯龟速 (保证能转过来)
                    
            elif is_normal_curve:
                target_speed = 70.0 # 普通高速弯
            
            # === ACC 防追尾逻辑 ===
            check_lane = target_lane_id if is_changing_lane and target_lane_id else current_lane_id
            has_obs, obs_spd, obs_dist = get_front_obstacle_speed(check_lane, perception_data)
            
            emergency_brake = False
            if has_obs:
                if obs_dist < 10.0:
                    emergency_brake = True
                    target_speed = 0.0
                elif obs_dist < 40.0:
                    target_speed = min(target_speed, max(0, obs_spd - 5.0))
                else:
                    target_speed = min(target_speed, obs_spd + 10.0)

            # 发送控制
            control = controller.run_step(target_speed, aim_wp, emergency_stop=emergency_brake)
            ego_vehicle.apply_control(control)
            
            # 5. 可视化
            world.debug.draw_point(aim_wp.transform.location, size=0.1, color=carla.Color(0, 255, 0), life_time=0.1)
            spectator.set_transform(carla.Transform(ego_vehicle.get_transform().location + carla.Location(z=40), carla.Rotation(pitch=-90)))
            
            # 状态打印
            status = "Str"
            if is_sharp_turn: status = f"SHARP({dist_to_curve:.0f}m)"
            elif is_normal_curve: status = "Curve"
            if is_changing_lane: status = "Change"
            
            print(f"\rSpd:{ego_spd:.0f}|{status}|Angle:{max_curve:.0f}|Tgt:{target_speed:.0f}|Steer:{control.steer:.2f}", end="   ")
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n停止测试...")
    finally:
        if ego_vehicle: ego_vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_list])

if __name__ == '__main__':
    main()