import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 加载 Town06
world = client.load_world('Town06')
print(f"已加载: {world.get_map().name}")

# 获取所有 spawn point
spawn_points = world.get_map().get_spawn_points()
print(f"\n总共有 {len(spawn_points)} 个 spawn point\n")

# 打印前 20 个
for i, sp in enumerate(spawn_points[:20]):
    print(f"Spawn {i:2d}: Location=({sp.location.x:8.1f}, {sp.location.y:8.1f}, "
          f"{sp.location.z:5.1f}), Yaw={sp.rotation.yaw:7.1f}°")

if len(spawn_points) > 20:
    print(f"... 还有 {len(spawn_points) - 20} 个 spawn point")
    print(f"\n最后一个 Spawn {len(spawn_points)-1}: "
          f"Location=({spawn_points[-1].location.x:.1f}, "
          f"{spawn_points[-1].location.y:.1f})")