"""
main.py — Highway FSM 自动驾驶系统主程序
================================================
"""

import sys
import os
import time
import math
import traceback

import carla
import numpy as np

# ---- 项目模块 ----
from setup import setup_carla, cleanup
from perception import PerceptionModule
from fsm_decision import FSMDecision, State
from fuzzy_engine import FuzzyLaneChangeEngine
from trajectory_planner import LaneChangeTrajectoryPlanner
from risk_field import RiskFieldCorrector
from controller import VehicleController
from curvature_speed_governor import CurvatureSpeedGovernor

import yaml

def load_config():
    """加载 YAML 配置文件"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'default_config.yaml'
    )
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            print(f"[Config] 已加载配置文件: {config_path}")
            return cfg
    print(f"[Config] 未找到 {config_path}，使用默认参数")
    return {}

# ==============================================================
#              天气设置（解决问题②）
# ==============================================================
def set_clear_weather(world):
    """设置晴朗天气"""
    weather = carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_azimuth_angle=45.0,
        sun_altitude_angle=70.0,    # 太阳高度角，70度≈正午偏早
        fog_density=0.0,
        fog_distance=0.0,
        wetness=0.0,
    )
    world.set_weather(weather)
    print("[Weather] 已设置为晴朗天气 ☀️")

# ==============================================================
#             NPC 速度差异化（解决问题③ 辅助）
# ==============================================================
def configure_npc_traffic(npc_vehicles, client):
    """
    让 NPC 速度差异化：部分慢车、部分快车。
    这样自车更容易遇到前方慢车，触发换道。
    
    参数：
        npc_vehicles : list of carla.Actor
        client : carla.Client （用于获取 TrafficManager）
    """
    import random
    
    try:
        tm = client.get_trafficmanager()
        tm.set_global_distance_to_leading_vehicle(2.5)

        for i, npc in enumerate(npc_vehicles):
            if npc is None or not npc.is_alive:
                continue
            npc.set_autopilot(True, tm.get_port())

            # 30% 的车设为慢车（限速的60-75%），增加换道机会
            # 40% 正常速度
            # 30% 略快
            r = random.random()
            if r < 0.30:
                # 慢车：percentage 为正值 → 比限速慢
                tm.vehicle_percentage_speed_difference(npc, random.uniform(25.0, 40.0))
            elif r < 0.70:
                # 正常
                tm.vehicle_percentage_speed_difference(npc, random.uniform(0.0, 15.0))
            else:
                # 快车：percentage 为负值 → 比限速快
                tm.vehicle_percentage_speed_difference(npc, random.uniform(-15.0, -5.0))

            # 随机换道行为，让交通更自然
            tm.random_left_lanechange_percentage(npc, 30)
            tm.random_right_lanechange_percentage(npc, 30)
            tm.auto_lane_change(npc, True)

        print(f"[Traffic] 已配置 {len(npc_vehicles)} 辆 NPC 差异化速度")
    except Exception as e:
        print(f"[Traffic] TrafficManager 配置失败: {e}，使用默认 autopilot")
        for npc in npc_vehicles:
            if npc is not None and npc.is_alive:
                npc.set_autopilot(True)

# ==============================================================
#             感知数据转换（关键适配层）
# ==============================================================
def convert_surrounding_to_offsets(surrounding_by_lane_id, ego_lane_id,
                                   lanes_info):
    """
    将 perception.py 输出的按真实 lane_id 分组的 surrounding
    转换为 fsm_decision.py 期望的按偏移量分组的格式。
    """
    offset_surr = {0: [], -1: [], 1: []}

    left_wp = lanes_info.get('left_wp')
    right_wp = lanes_info.get('right_wp')

    left_lane_id = left_wp.lane_id if left_wp else None
    right_lane_id = right_wp.lane_id if right_wp else None

    for lane_id, vehicles in surrounding_by_lane_id.items():
        for v in vehicles:
            if 'actor_id' not in v:
                v['actor_id'] = v.get('id', 0)

        if lane_id == ego_lane_id:
            offset = 0
        elif left_lane_id is not None and lane_id == left_lane_id:
            offset = -1
        elif right_lane_id is not None and lane_id == right_lane_id:
            offset = 1
        else:
            continue

        offset_surr[offset].extend(vehicles)

    for offset in offset_surr:
        offset_surr[offset].sort(key=lambda v: v['rel_dist'])

    return offset_surr

def generate_waypoint_trajectory(waypoint, distance=150.0, spacing=2.0):
    """从当前 waypoint 向前采样轨迹"""
    transforms = []
    wp = waypoint
    total = 0.0
    while total < distance:
        transforms.append(wp.transform)
        nxt = wp.next(spacing)
        if not nxt:
            break
        wp = nxt[0]
        total += spacing
    return transforms

def get_front_vehicle(offset_surrounding):
    """获取当前车道最近前车"""
    vehs = offset_surrounding.get(0, [])
    fronts = [v for v in vehs if v['rel_dist'] > 2.0]
    return min(fronts, key=lambda v: v['rel_dist']) if fronts else None

def collect_obstacles(offset_surrounding):
    """收集所有周围车辆作为势场障碍物"""
    obstacles = []
    for lane_vehs in offset_surrounding.values():
        for v in lane_vehs:
            if 'location' in v and 'velocity' in v:
                obstacles.append({
                    'location': v['location'],
                    'velocity': v['velocity'],
                    'speed': v['speed'],
                })
    return obstacles

def _update_spectator(spectator, ego_vehicle):
    """第三人称追尾视角"""
    tf = ego_vehicle.get_transform()
    yaw_rad = math.radians(tf.rotation.yaw)

    offset_back = 12.0
    offset_up = 8.0
    pitch = -20.0

    cam_x = tf.location.x - offset_back * math.cos(yaw_rad)
    cam_y = tf.location.y - offset_back * math.sin(yaw_rad)
    cam_z = tf.location.z + offset_up

    spectator.set_transform(carla.Transform(
        carla.Location(x=cam_x, y=cam_y, z=cam_z),
        carla.Rotation(pitch=pitch, yaw=tf.rotation.yaw, roll=0.0)
    ))

# ==============================================================
#                         主函数
# ==============================================================
def main():
    config = load_config()
    target_speed = config.get('target_speed', 100.0)

    # ============================================
    # 初始化 CARLA
    # ============================================
    world, ego_vehicle, npc_vehicles = setup_carla(config)

    # 获取 client 对象用于 TrafficManager
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
    except Exception as e:
        print(f"[Main] 无法连接 CARLA client: {e}")
        client = None

    # ============================================
    # 设置晴朗天气（问题②）
    # ============================================
    set_clear_weather(world)

    # ============================================
    # 配置 NPC 差异化速度（问题③ 辅助）
    # ============================================
    if client:
        configure_npc_traffic(npc_vehicles, client)
    else:
        print("[Traffic] 跳过 TrafficManager 配置")
        for npc in npc_vehicles:
            if npc is not None and npc.is_alive:
                npc.set_autopilot(True)

    # ============================================
    # 初始化各模块
    # ============================================
    perception = PerceptionModule(ego_vehicle, config.get('perception', {}))

    fsm = FSMDecision(config)
    fuzzy = FuzzyLaneChangeEngine(config)
    fsm.set_fuzzy_engine(fuzzy)

    planner = LaneChangeTrajectoryPlanner(config)
    risk_field = RiskFieldCorrector(config)
    controller = VehicleController(ego_vehicle, config)
    speed_governor = CurvatureSpeedGovernor(
        config.get('curvature_governor', {}))

    # 变道轨迹缓存
    active_trajectory = None

    # 观察者摄像机
    spectator = world.get_spectator()

    # 统计
    lane_change_count = 0
    last_state = State.KEEP_LANE

    print("=" * 60)
    print("  Highway FSM 自动驾驶系统启动")
    print("  创新: 曲率调速 | 信息熵 | 反事实 | 相空间 | 多预瞄")
    print("  天气: 晴朗 ☀️  | NPC: {} 辆 (差异化速度)".format(
        len(npc_vehicles)))
    print("  按 Ctrl+C 停止")
    print("=" * 60)

    frame_count = 0

    try:
        while True:
            # ---- Tick ----
            world.tick()
            frame_count += 1

            # ---- 更新观察者视角 ----
            _update_spectator(spectator, ego_vehicle)

            # ============================================
            # 1. 感知
            # ============================================
            perception_data = perception.update()

            ego_data = perception_data['ego']
            raw_surrounding = perception_data['surrounding']
            lanes_info = perception_data['lanes']

            current_speed = ego_data['speed']
            current_wp = ego_data['waypoint']
            ego_lane_id = ego_data['lane_id']

            surrounding = convert_surrounding_to_offsets(
                raw_surrounding, ego_lane_id, lanes_info
            )

            # ============================================
            # 2. 曲率积分速度调控（构想①）
            # ============================================
            curvature_safe_speed = speed_governor.compute_safe_speed(
                current_wp, target_speed, current_speed
            )

            # ============================================
            # 3. 决策（FSM + 模糊 + 熵 + 反事实 + 相空间）
            # ============================================
            decision = fsm.update(ego_data, surrounding)
            state = decision['state']
            target_lane_id = decision['target_lane_id']

            # 统计换道次数
            if (state in (State.CHANGE_LEFT, State.CHANGE_RIGHT) and
                    last_state not in (State.CHANGE_LEFT, State.CHANGE_RIGHT)):
                lane_change_count += 1
                direction_str = "← 左" if state == State.CHANGE_LEFT else "右 →"
                print(f"\n{'='*40}")
                print(f"  🚗 第 {lane_change_count} 次换道: {direction_str}")
                print(f"  速度={current_speed:.1f} km/h  "
                      f"意愿={fsm.debug_info['desire_effective']:.2f}")
                print(f"{'='*40}\n")
            last_state = state

            # ============================================
            # 4. 规划
            # ============================================
            trajectory = None

            if state in (State.CHANGE_LEFT, State.CHANGE_RIGHT):
                if active_trajectory is None:
                    direction = -1 if state == State.CHANGE_LEFT else 1
                    target_wp = (lanes_info.get('left_wp')
                                 if direction == -1
                                 else lanes_info.get('right_wp'))

                    if target_wp is not None:
                        try:
                            raw_traj = planner.plan(
                                ego_data['transform'],
                                current_speed,
                                target_wp,
                            )
                            obstacles = collect_obstacles(surrounding)
                            if obstacles and raw_traj:
                                try:
                                    active_trajectory = risk_field.correct(
                                        raw_traj, obstacles,
                                        ego_data['transform'],
                                        current_speed
                                    )
                                except Exception:
                                    active_trajectory = raw_traj
                            else:
                                active_trajectory = raw_traj
                        except Exception as e:
                            print(f"[Planner] 轨迹规划失败: {e}")
                            active_trajectory = None

                trajectory = active_trajectory

            elif state in (State.KEEP_LANE, State.PREP_LEFT, State.PREP_RIGHT):
                active_trajectory = None
                trajectory = generate_waypoint_trajectory(
                    current_wp, distance=150.0, spacing=2.0
                )

            elif state == State.ABORT:
                active_trajectory = None
                trajectory = generate_waypoint_trajectory(
                    current_wp, distance=100.0, spacing=2.0
                )

            # ============================================
            # 5. 速度计算
            # ============================================
            follow_speed = target_speed
            front = get_front_vehicle(surrounding)
            if front is not None:
                fd = front['rel_dist']
                if fd < 50.0:
                    ratio = max(0.0, (fd - 5.0) / 45.0)
                    follow_speed = (front['speed']
                                    + ratio * (target_speed - front['speed']))
                    follow_speed = max(follow_speed, 30.0)

            control_speed = min(target_speed, curvature_safe_speed,
                                follow_speed)

            if state in (State.CHANGE_LEFT, State.CHANGE_RIGHT):
                control_speed = min(control_speed, 90.0)  # 换道限速略提高

            # ============================================
            # 6. 控制
            # ============================================
            emergency = False
            if front is not None and front['rel_dist'] < 5.0:
                emergency = True

            control = controller.run_step(
                target_speed=control_speed,
                trajectory=trajectory,
                emergency_stop=emergency,
            )
            ego_vehicle.apply_control(control)

            # ============================================
            # 7. 日志输出
            # ============================================
            if frame_count % 20 == 0:
                gov = speed_governor.debug
                ctrl = controller.debug
                di = fsm.debug_info
                traj_str = (f"{len(trajectory)}pts"
                            if trajectory else "None")

                front_str = (f"d={front['rel_dist']:.1f}m "
                             f"v={front['speed']:.1f}"
                             if front else "None")

                print(
                    f"[F{frame_count:5d}] "
                    f"{di['state']:14s} | "
                    f"Spd={current_speed:5.1f} → {control_speed:5.1f} | "
                    f"Curv={gov['final_speed_limit']:5.1f} | "
                    f"Des={di['desire_effective']:.2f} | "
                    f"H={di.get('entropy', 0):.2f} | "
                    f"CTE={ctrl['cte']:+.2f} | "
                    f"Front=[{front_str}] | "
                    f"Traj={traj_str} | "
                    f"LC={lane_change_count}"
                )

    except KeyboardInterrupt:
        print(f"\n[Main] 用户中断 | 总换道次数: {lane_change_count}")
    except Exception as e:
        print(f"\n[Main] 运行异常: {e}")
        traceback.print_exc()
    finally:
        cleanup(world, ego_vehicle, npc_vehicles)
        print("[Main] 程序结束")

# ==============================================================
if __name__ == '__main__':
    main()

