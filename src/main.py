"""
Highway FSM 自动驾驶系统 - 主入口
=====================================
整合所有模块的主控制循环
"""
import carla
import yaml
import time
import random
import sys
import os
import math
import traceback

# 将 src 目录添加到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perception import PerceptionModule
from fsm_decision import FSMDecision, State
from planner import MotionPlanner
from controller import AdaptiveStanleyController
from visualization import Visualizer

def load_config(config_path='../config/config.yaml'):
    """加载配置文件"""
    abs_path = os.path.join(os.path.dirname(__file__), config_path)
    if os.path.exists(abs_path):
        with open(abs_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        print(f"[Warning] Config file not found: {abs_path}, using defaults")
        return {}

def setup_carla(config):
    """连接 CARLA 并初始化世界"""
    carla_cfg = config.get('carla', {})
    # host = carla_cfg.get('host', 'localhost')
    host = '127.0.0.1'
    port = carla_cfg.get('port', 2000)
    timeout = carla_cfg.get('timeout', 10.0)

    client = carla.Client(host, port)
    client.set_timeout(timeout)

    # 加载地图
    map_name = carla_cfg.get('map', 'Town04')
    world = client.load_world(map_name)

    # 设置同步模式
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    # 设置天气 (白天晴天，论文截图好看)
    weather = carla.WeatherParameters.ClearNoon
    world.set_weather(weather)

    return client, world

def spawn_ego_vehicle(world, config):
    """生成自车"""
    ego_cfg = config.get('ego', {})
    model = ego_cfg.get('vehicle_model', 'vehicle.tesla.model3')

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find(model)
    vehicle_bp.set_attribute('role_name', 'hero')

    # 选择出生点
    spawn_points = world.get_map().get_spawn_points()
    spawn_idx = config.get('carla', {}).get('spawn_point_index', 50)
    spawn_idx = min(spawn_idx, len(spawn_points) - 1)

    ego = world.spawn_actor(vehicle_bp, spawn_points[spawn_idx])
    print(f"[Setup] 自车已生成: {model} at spawn point {spawn_idx}")
    return ego

def spawn_npc_vehicles(world, config):
    """生成 NPC 交通车辆"""
    carla_cfg = config.get('carla', {})
    num_npc = carla_cfg.get('num_npc', 30)

    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()

    npcs = []
    random.shuffle(spawn_points)

    for i in range(min(num_npc, len(spawn_points) - 1)):
        bp = random.choice(vehicle_bps)
        try:
            npc = world.spawn_actor(bp, spawn_points[i + 1])
            npc.set_autopilot(True)
            npcs.append(npc)
        except Exception:
            continue

    print(f"[Setup] 已生成 {len(npcs)} 辆 NPC 车辆")
    return npcs

def setup_spectator(world, vehicle):
    """设置俯视观察视角"""
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(z=50),
        carla.Rotation(pitch=-90)
    ))

def update_spectator(world, vehicle):
    """跟随自车更新观察视角"""
    spectator = world.get_spectator()
    transform = vehicle.get_transform()

    # 第三人称跟随视角
    forward = transform.get_forward_vector()
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(
            x=-10 * forward.x,
            y=-10 * forward.y,
            z=8
        ),
        carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)
    ))

def main():
    """主函数"""
    # ============================================================
    # 初始化
    # ============================================================
    config = load_config()

    client, world = setup_carla(config)
    ego_vehicle = spawn_ego_vehicle(world, config)
    npc_vehicles = spawn_npc_vehicles(world, config)

    # Tick 一次让所有车辆就位
    world.tick()
    time.sleep(1.0)

    # 初始化各模块
    perception = PerceptionModule(ego_vehicle, config.get('perception', {}))
    decision = FSMDecision(config)
    planner = MotionPlanner(ego_vehicle, config)
    controller = AdaptiveStanleyController(ego_vehicle, config)
    visualizer = Visualizer(world, config)

    target_speed = config.get('ego', {}).get('target_speed', 100.0)

    # 轨迹状态
    active_trajectory = []   # 当前正在执行的轨迹
    trajectory_index = 0     # 轨迹进度

    print("=" * 60)
    print("  Highway FSM 自动驾驶系统启动")
    print("  按 Ctrl+C 停止")
    print("=" * 60)

    frame_count = 0

    try:
        while True:
            # ============================================================
            # Step 0: 仿真前进
            # ============================================================
            world.tick()
            frame_count += 1

            # ============================================================
            # Step 1: 感知
            # ============================================================
            percept_data = perception.update()
            ego_data = percept_data['ego']
            surrounding = percept_data['surrounding']
            lanes = percept_data['lanes']

            # ============================================================
            # Step 2: 决策
            # ============================================================
            state = decision.decide(percept_data)

            # ============================================================
            # Step 3: 规划
            # ============================================================

            # 准备变道 → 生成轨迹
            if state == State.PREPARE_LANE_CHANGE_LEFT and not active_trajectory:
                if lanes['left_wp'] is not None:
                    obstacles = perception.get_obstacles_for_planner(surrounding)
                    active_trajectory = planner.generate_lane_change_trajectory(
                        start_wp=ego_data['waypoint'],
                        target_wp=lanes['left_wp'],
                        ego_speed=ego_data['speed'],
                        obstacles=obstacles
                    )
                    if active_trajectory:
                        decision.notify_lane_change_start()
                        print(f"[Planner] 左变道轨迹已生成: {len(active_trajectory)} 点")
                    else:
                        decision.notify_lane_change_complete()

            elif state == State.PREPARE_LANE_CHANGE_RIGHT and not active_trajectory:
                if lanes['right_wp'] is not None:
                    obstacles = perception.get_obstacles_for_planner(surrounding)
                    active_trajectory = planner.generate_lane_change_trajectory(
                        start_wp=ego_data['waypoint'],
                        target_wp=lanes['right_wp'],
                        ego_speed=ego_data['speed'],
                        obstacles=obstacles
                    )
                    if active_trajectory:
                        decision.notify_lane_change_start()
                        print(f"[Planner] 右变道轨迹已生成: {len(active_trajectory)} 点")
                    else:
                        decision.notify_lane_change_complete()

            # 车道保持时生成参考轨迹
            elif state == State.KEEP_LANE and not active_trajectory:
                keep_traj = planner.generate_keep_lane_trajectory(
                    ego_data['waypoint'], length=40.0
                )
                # 车道保持轨迹不存入 active_trajectory (每帧刷新)
                pass

            # ============================================================
            # Step 4: 轨迹完成检查
            # ============================================================
            if active_trajectory:
                # 检查是否到达轨迹终点
                ego_loc = ego_data['location']
                end_loc = active_trajectory[-1].location
                dist_to_end = ego_loc.distance(end_loc)

                if dist_to_end < 3.0:
                    print("[Planner] 轨迹执行完毕")
                    active_trajectory = []
                    decision.notify_lane_change_complete()

            # ============================================================
            # Step 5: 控制
            # ============================================================

            # 计算目标速度 (跟车时需减速)
            current_lane = ego_data['lane_id']
            front = perception.get_front_vehicle(current_lane, surrounding)
            control_speed = target_speed

            if front is not None:
                fd = front['rel_dist']
                if fd < 40.0:
                    # 线性减速
                    ratio = max(0.3, fd / 40.0)
                    control_speed = min(target_speed, front['speed'] + 5.0)
                    control_speed = max(control_speed, 30.0)

            # 紧急制动判断
            emergency = False
            if front is not None and front['rel_dist'] < 8.0:
                emergency = True

            # 选择控制输入
            if active_trajectory:
                control = controller.run_step(
                    control_speed, trajectory=active_trajectory,
                    emergency_stop=emergency
                )
            else:
                # 车道保持: 使用地图路点
                keep_traj = planner.generate_keep_lane_trajectory(
                    ego_data['waypoint'], length=40.0
                )
                control = controller.run_step(
                    control_speed, trajectory=keep_traj,
                    emergency_stop=emergency
                )

            ego_vehicle.apply_control(control)

            # ============================================================
            # Step 6: 可视化
            # ============================================================
            if active_trajectory:
                visualizer.draw_trajectory(
                    active_trajectory,
                    color=carla.Color(0, 255, 0),
                    life_time=0.15
                )

            visualizer.draw_vehicle_state(
                ego_vehicle, decision.state, decision.debug_info
            )

            # 更新观察视角
            if frame_count % 5 == 0:
                update_spectator(world, ego_vehicle)

            # 打印状态 (每秒一次)
            if frame_count % 20 == 0:
                desire = decision.debug_info.get('fuzzy_desire', 0)
                cte = controller.debug.get('cte', 0)
                k_adp = controller.debug.get('k_adaptive', 0)

                print(
                    f"[Frame {frame_count:5d}] "
                    f"State={decision.state.name:25s} | "
                    f"Speed={ego_data['speed']:5.1f} km/h | "
                    f"Desire={desire:.2f} | "
                    f"CTE={cte:+.3f} | "
                    f"K={k_adp:.3f} | "
                    f"Traj={'Active' if active_trajectory else 'None':6s}"
                )

    except KeyboardInterrupt:
        print("\n[System] 用户中断，正在清理...")

    except Exception as e:
        print(f"\n[Error] {e}")
        traceback.print_exc()

    finally:
        # ============================================================
        # 清理
        # ============================================================
        print("[Cleanup] 销毁车辆...")

        # 恢复异步模式
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        # 销毁 NPC
        for npc in npc_vehicles:
            try:
                npc.destroy()
            except Exception:
                pass

        # 销毁自车
        try:
            ego_vehicle.destroy()
        except Exception:
            pass

        print("[Cleanup] 完成")

if __name__ == '__main__':
    main()
