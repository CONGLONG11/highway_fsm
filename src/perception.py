# 文件: ~/Highway_FSM/src/perception.py
import carla
import math
import numpy as np

class PerceptionModule:
    def __init__(self, vehicle, world):
        self.ego_vehicle = vehicle
        self.world = world
        self.map = world.get_map()
        
    def get_perception_data(self):
        """主接口：获取当前帧的所有环境信息"""
        ego_transform = self.ego_vehicle.get_transform()
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)
        
        # 获取当前位置 Waypoint
        # project_to_road=True 防止车辆轻微腾空导致找不到路
        ego_waypoint = self.map.get_waypoint(ego_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        # 获取周边车辆
        surrounding_objects = self._detect_surrounding_vehicles(ego_waypoint, ego_transform)

        return {
            'ego': {
                'id': self.ego_vehicle.id,
                'speed': ego_speed,
                'lane_id': ego_waypoint.lane_id,
                'road_id': ego_waypoint.road_id,
                's': ego_waypoint.s, 
                'waypoint': ego_waypoint,
                'transform': ego_transform
            },
            'surrounding': surrounding_objects,
            'lanes': self._analyze_lane_availability(ego_waypoint)
        }

    def _detect_surrounding_vehicles(self, ego_wp, ego_transform, radius=120.0):
        """检测周边车辆，计算纵向和横向距离"""
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_speed_scalar = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
        
        ego_loc = ego_transform.location
        ego_fwd = ego_transform.get_forward_vector()
        ego_right = ego_transform.get_right_vector() # 用于计算横向距离

        vehicle_list = self.world.get_actors().filter('vehicle.*')
        objects_by_lane = {} 
        
        for npc in vehicle_list:
            if npc.id == self.ego_vehicle.id: continue
                
            npc_loc = npc.get_transform().location
            dist_euclidean = npc_loc.distance(ego_loc)
            if dist_euclidean > radius: continue

            npc_wp = self.map.get_waypoint(npc_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if not npc_wp: continue 
            
            lane_id = npc_wp.lane_id
            
            # 向量法计算相对位置
            vec_npc_ego = carla.Location(npc_loc.x - ego_loc.x, npc_loc.y - ego_loc.y, npc_loc.z - ego_loc.z)
            
            # 纵向距离 (Longitudinal)
            longitudinal_dist = (vec_npc_ego.x * ego_fwd.x) + (vec_npc_ego.y * ego_fwd.y)
            # 横向距离 (Lateral) >0为右, <0为左
            lateral_dist = (vec_npc_ego.x * ego_right.x) + (vec_npc_ego.y * ego_right.y)
            
            npc_speed = 3.6 * math.sqrt(npc.get_velocity().x**2 + npc.get_velocity().y**2)

            obj_info = {
                'id': npc.id,
                'lane_id': lane_id,
                'rel_dist': longitudinal_dist, 
                'lat_dist': lateral_dist,
                'speed': npc_speed,
                'rel_speed': npc_speed - ego_speed_scalar 
            }
            
            if lane_id not in objects_by_lane: objects_by_lane[lane_id] = []
            objects_by_lane[lane_id].append(obj_info)
            
        return objects_by_lane

    def _analyze_lane_availability(self, waypoint):
        """分析左右车道是否可变道 (检查虚线)"""
        left_wp = waypoint.get_left_lane()
        can_change_left = False
        if left_wp and left_wp.lane_type == carla.LaneType.Driving:
            if waypoint.left_lane_marking.type == carla.LaneMarkingType.Broken:
                can_change_left = True
                
        right_wp = waypoint.get_right_lane()
        can_change_right = False
        if right_wp and right_wp.lane_type == carla.LaneType.Driving:
            if waypoint.right_lane_marking.type == carla.LaneMarkingType.Broken:
                can_change_right = True
                
        return {
            'left_available': can_change_left,
            'right_available': can_change_right,
            'current_lane_width': waypoint.lane_width
        }