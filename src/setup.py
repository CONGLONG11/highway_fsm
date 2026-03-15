"""
setup.py — CARLA 环境初始化与清理
================================================
功能：
  1. 连接 CARLA 服务器
  2. 设置同步模式
  3. 生成自车（带碰撞重试）
  4. 生成 NPC 车辆（带自动驾驶）
  5. 清理函数
"""

import carla
import random
import time
import sys

def setup_carla(config=None):
    """
    初始化 CARLA 世界、自车和 NPC。

    参数：
        config : dict，可选配置

    返回：
        (world, ego_vehicle, npc_vehicles)
    """
    config = config or {}
    carla_cfg = config.get('carla', {})

    host = carla_cfg.get('host', 'localhost')
    port = carla_cfg.get('port', 2000)
    timeout = carla_cfg.get('timeout', 10.0)
    target_map = carla_cfg.get('map', 'Town04')
    sync_mode = carla_cfg.get('sync_mode', True)
    delta = carla_cfg.get('delta_seconds', 0.05)

    spawn_index = config.get('spawn_point_index', 50)
    num_npc = config.get('num_npc', 30)

    # ========================================
    # 1. 连接服务器
    # ========================================
    print(f"[Setup] 连接 CARLA 服务器 {host}:{port} ...")
    client = carla.Client(host, port)
    client.set_timeout(timeout)

    # 加载地图（如果当前地图不匹配）
    current_map = client.get_world().get_map().name
    if target_map not in current_map:
        print(f"[Setup] 正在加载地图 {target_map} ...")
        client.load_world(target_map)
        time.sleep(5.0)  # 等待地图加载完成
    else:
        print(f"[Setup] 当前地图已是 {target_map}")

    world = client.get_world()

    # ========================================
    # 2. 清理旧车辆（防残留导致碰撞）
    # ========================================
    _destroy_all_vehicles(world, client)

    # ========================================
    # 3. 设置同步模式
    # ========================================
    settings = world.get_settings()
    if sync_mode:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta
    else:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    # 设置交通管理器同步
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(sync_mode)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.global_percentage_speed_difference(10.0)

    # Tick 几帧让世界稳定
    if sync_mode:
        for _ in range(5):
            world.tick()

    # ========================================
    # 4. 生成自车（带碰撞重试）
    # ========================================
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        print("[Setup] 错误：地图没有生成点！")
        sys.exit(1)

    ego_bp = blueprint_library.find('vehicle.tesla.model3')
    ego_bp.set_attribute('role_name', 'hero')

    ego_vehicle = None
    ego_spawn = None

    # 尝试指定的 spawn point
    if spawn_index < len(spawn_points):
        try:
            ego_spawn = spawn_points[spawn_index]
            ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn)
            if ego_vehicle is not None:
                print(f"[Setup] 自车生成成功: spawn point {spawn_index}")
        except Exception as e:
            print(f"[Setup] spawn point {spawn_index} 失败: {e}")

    # 如果指定点失败，遍历其他 spawn point
    if ego_vehicle is None:
        print("[Setup] 指定生成点不可用，尝试其他位置...")
        # 优先尝试附近的点
        candidates = list(range(len(spawn_points)))
        # 把指定索引附近的点排在前面
        candidates.sort(key=lambda i: abs(i - spawn_index))

        for idx in candidates:
            try:
                ego_spawn = spawn_points[idx]
                ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn)
                if ego_vehicle is not None:
                    print(f"[Setup] 自车生成成功: spawn point {idx}"
                          f"（原始请求: {spawn_index}）")
                    break
            except Exception:
                continue

    if ego_vehicle is None:
        print("[Setup] 致命错误：所有生成点都不可用！")
        print("[Setup] 请确保 CARLA 服务器正在运行且地图已正确加载")
        sys.exit(1)

    # 等待自车稳定
    if sync_mode:
        world.tick()
        world.tick()

    # ========================================
    # 5. 生成 NPC 车辆
    # ========================================
    npc_vehicles = []

    vehicle_bps = blueprint_library.filter('vehicle.*')
    vehicle_bps = [bp for bp in vehicle_bps
                   if int(bp.get_attribute('number_of_wheels')) >= 4]

    # 排除自车附近的生成点（防碰撞）
    ego_loc = ego_spawn.location
    available_spawns = [sp for sp in spawn_points
                        if sp.location.distance(ego_loc) > 15.0]
    random.shuffle(available_spawns)

    num_to_spawn = min(num_npc, len(available_spawns))
    print(f"[Setup] 准备生成 {num_to_spawn} 辆 NPC ...")

    batch = []
    for i in range(num_to_spawn):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        bp.set_attribute('role_name', 'autopilot')
        batch.append(
            carla.command.SpawnActor(bp, available_spawns[i])
            .then(carla.command.SetAutopilot(
                carla.command.FutureActor, True, 8000))
        )

    results = client.apply_batch_sync(batch, True)
    for result in results:
        if not result.error:
            actor = world.get_actor(result.actor_id)
            if actor is not None:
                npc_vehicles.append(actor)

    print(f"[Setup] 成功生成 {len(npc_vehicles)} 辆 NPC 车辆")

    # NPC 行为微调
    for npc in npc_vehicles:
        try:
            traffic_manager.vehicle_percentage_speed_difference(
                npc, random.uniform(-10, 30)
            )
            traffic_manager.distance_to_leading_vehicle(
                npc, random.uniform(3, 8)
            )
        except Exception:
            pass

    # 多 tick 几帧让物理稳定
    if sync_mode:
        for _ in range(10):
            world.tick()

    print(f"[Setup] 初始化完成！自车速度目标: {config.get('target_speed', 100)} km/h")
    return world, ego_vehicle, npc_vehicles

def _destroy_all_vehicles(world, client):
    """清理地图上所有已有车辆（防止残留导致 spawn 碰撞）"""
    actors = world.get_actors().filter('vehicle.*')
    if len(actors) > 0:
        print(f"[Setup] 清理 {len(actors)} 辆残留车辆...")
        batch = [carla.command.DestroyActor(a.id) for a in actors]
        client.apply_batch_sync(batch, True)
        time.sleep(1.0)

def cleanup(world, ego_vehicle, npc_vehicles):
    """
    清理所有生成的 Actor，恢复世界设置。
    """
    print("[Cleanup] 正在清理...")

    # 获取 client
    client = None
    try:
        # 从 world 获取 client（CARLA 0.9.13+ 支持）
        actors = world.get_actors()
        # 通过 ego_vehicle 获取
        if ego_vehicle is not None:
            client = ego_vehicle.get_world()  # 实际上需要 client
    except Exception:
        pass

    # 停止 NPC 自动驾驶 & 销毁
    destroyed = 0
    for v in npc_vehicles:
        try:
            if v is not None and v.is_alive:
                v.set_autopilot(False, 8000)
                v.destroy()
                destroyed += 1
        except Exception:
            pass
    print(f"[Cleanup] 销毁 {destroyed} 辆 NPC")

    # 销毁自车
    if ego_vehicle is not None:
        try:
            ego_vehicle.destroy()
            print("[Cleanup] 自车已销毁")
        except Exception:
            pass

    # 恢复异步模式
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        print("[Cleanup] 世界设置已恢复（异步模式）")
    except Exception:
        pass

    print("[Cleanup] 清理完成")
