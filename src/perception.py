"""
感知模块 (Perception Module)
================================
负责从 CARLA 世界中提取结构化信息，供决策和规划使用。

输出格式:
{
  'ego': {
      'location': carla.Location,
      'transform': carla.Transform,
      'speed': float (km/h),
      'velocity': carla.Vector3D,
      'lane_id': int,
      'waypoint': carla.Waypoint,
  },
  'surrounding': {
      lane_id: [
          {
              'id': int,
              'location': carla.Location,
              'speed': float,
              'velocity': carla.Vector3D,
              'rel_dist': float (有符号, 正=前方),
              'rel_speed': float (正=对方比我快),
              'lane_id': int,
          }, ...
      ], ...
  },
  'lanes': {
      'current_id': int,
      'left_available': bool,
      'right_available': bool,
      'left_wp': carla.Waypoint or None,
      'right_wp': carla.Waypoint or None,
  }
}
"""
import carla
import math
import numpy as np

class PerceptionModule:
    def __init__(self, vehicle, config=None):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()

        config = config or {}
        self.detection_range = config.get('detection_range', 100.0)

    def update(self):
        """主更新函数，返回完整的感知数据"""
        ego_data = self._get_ego_data()
        surrounding = self._get_surrounding_vehicles(ego_data)
        lanes = self._get_lane_info(ego_data)

        return {
            'ego': ego_data,
            'surrounding': surrounding,
            'lanes': lanes,
        }

    def _get_ego_data(self):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        waypoint = self.map.get_waypoint(transform.location)

        return {
            'location': transform.location,
            'transform': transform,
            'speed': speed,
            'velocity': velocity,
            'lane_id': waypoint.lane_id,
            'waypoint': waypoint,
            'yaw': math.radians(transform.rotation.yaw),
        }

    def _get_surrounding_vehicles(self, ego_data):
        """
        获取周围车辆信息，按车道分组
        """
        surrounding = {}
        ego_loc = ego_data['location']
        ego_wp = ego_data['waypoint']
        ego_forward = np.array([
            math.cos(ego_data['yaw']),
            math.sin(ego_data['yaw'])
        ])

        all_vehicles = self.world.get_actors().filter('vehicle.*')

        for v in all_vehicles:
            if v.id == self.vehicle.id:
                continue

            v_loc = v.get_transform().location
            dist = ego_loc.distance(v_loc)

            if dist > self.detection_range:
                continue

            v_vel = v.get_velocity()
            v_speed = 3.6 * math.sqrt(v_vel.x**2 + v_vel.y**2 + v_vel.z**2)
            v_wp = self.map.get_waypoint(v_loc)

            if v_wp is None:
                continue

            # 计算相对距离 (带符号)
            vec_to_v = np.array([v_loc.x - ego_loc.x, v_loc.y - ego_loc.y])
            rel_dist_signed = np.dot(vec_to_v, ego_forward)

            v_lane_id = v_wp.lane_id
            if v_lane_id not in surrounding:
                surrounding[v_lane_id] = []

            surrounding[v_lane_id].append({
                'id': v.id,
                'location': v_loc,
                'speed': v_speed,
                'velocity': v_vel,
                'rel_dist': rel_dist_signed,
                'rel_speed': v_speed - ego_data['speed'],
                'lane_id': v_lane_id,
            })

        # 每个车道按前后距离排序
        for lane_id in surrounding:
            surrounding[lane_id].sort(key=lambda x: x['rel_dist'])

        return surrounding

    def _get_lane_info(self, ego_data):
        """检查左右车道可用性"""
        wp = ego_data['waypoint']

        left_wp = wp.get_left_lane()
        right_wp = wp.get_right_lane()

        left_available = False
        right_available = False

        if left_wp is not None:
            # 确保是同方向车道
            if left_wp.lane_type == carla.LaneType.Driving:
                if str(left_wp.lane_change) in ['Left', 'Both']:
                    left_available = True
                # 也检查当前车道的 lane_change 属性
                if str(wp.lane_change) in ['Left', 'Both']:
                    left_available = True

        if right_wp is not None:
            if right_wp.lane_type == carla.LaneType.Driving:
                if str(right_wp.lane_change) in ['Right', 'Both']:
                    right_available = True
                if str(wp.lane_change) in ['Right', 'Both']:
                    right_available = True

        return {
            'current_id': ego_data['lane_id'],
            'left_available': left_available,
            'right_available': right_available,
            'left_wp': left_wp,
            'right_wp': right_wp,
        }

    def get_front_vehicle(self, lane_id, surrounding):
        """获取指定车道上最近的前车"""
        if lane_id not in surrounding:
            return None

        for v in surrounding[lane_id]:
            if v['rel_dist'] > 2.0:  # 前方 (忽略并排的)
                return v
        return None

    def get_obstacles_for_planner(self, surrounding):
        """
        将感知数据转换为规划器需要的格式
        :return: list of {'pos': np.array, 'vel': np.array}
        """
        obstacles = []
        for lane_id, vehicles in surrounding.items():
            for v in vehicles:
                loc = v['location']
                vel = v['velocity']
                obstacles.append({
                    'pos': np.array([loc.x, loc.y]),
                    'vel': np.array([vel.x, vel.y]),
                })
        return obstacles
