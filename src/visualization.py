"""
可视化工具模块
================
在 CARLA 世界中绘制调试信息：轨迹、航点、风险等
"""
import carla
import math

class Visualizer:
    def __init__(self, world, config=None):
        self.world = world
        self.debug = world.debug
        config = config or {}
        vis_cfg = config.get('visualization', {})
        self.enabled = vis_cfg.get('enabled', True)
        self.draw_traj = vis_cfg.get('draw_trajectory', True)
        self.draw_wp = vis_cfg.get('draw_waypoints', True)

    def draw_trajectory(self, trajectory, color=None, life_time=0.1):
        """绘制规划的轨迹"""
        if not self.enabled or not self.draw_traj:
            return
        if not trajectory or len(trajectory) < 2:
            return

        if color is None:
            color = carla.Color(0, 255, 0)  # 绿色

        for i in range(len(trajectory) - 1):
            loc1 = trajectory[i].location
            loc2 = trajectory[i+1].location

            # 抬高一点避免被地面遮挡
            loc1.z += 0.5
            loc2.z += 0.5

            self.debug.draw_line(
                loc1, loc2,
                thickness=0.08,
                color=color,
                life_time=life_time
            )

    def draw_waypoints(self, waypoints, color=None, life_time=0.1):
        """绘制航点"""
        if not self.enabled or not self.draw_wp:
            return
        if color is None:
            color = carla.Color(0, 0, 255)

        for wp in waypoints:
            loc = wp.transform.location
            loc.z += 0.5
            self.debug.draw_point(
                loc, size=0.1, color=color, life_time=life_time
            )

    def draw_info_text(self, location, text, color=None, life_time=0.1):
        """在世界中绘制文字"""
        if not self.enabled:
            return
        if color is None:
            color = carla.Color(255, 255, 255)

        loc = carla.Location(x=location.x, y=location.y, z=location.z + 2.0)
        self.debug.draw_string(loc, text, color=color, life_time=life_time)

    def draw_vehicle_state(self, vehicle, decision_state, debug_info=None):
        """在车辆上方绘制状态信息"""
        if not self.enabled:
            return

        loc = vehicle.get_transform().location
        speed = vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)

        text = f"State: {decision_state.name}\n"
        text += f"Speed: {speed_kmh:.1f} km/h\n"

        if debug_info:
            if 'fuzzy_desire' in debug_info:
                text += f"Desire: {debug_info['fuzzy_desire']:.2f}\n"
            if 'costs' in debug_info:
                costs = debug_info['costs']
                text += f"Cost K:{costs.get('keep','?')} L:{costs.get('left','?')} R:{costs.get('right','?')}"

        self.draw_info_text(loc, text, life_time=0.15)
