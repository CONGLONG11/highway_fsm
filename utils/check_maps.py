import glob
import os
import sys
 
# 这部分是Carla Python API的标准导入路径设置，确保能找到carla模块
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
 
import carla
 
# 连接到Carla服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
 
# 获取所有可用的地图并打印
available_maps = client.get_available_maps()
print(f"当前共有 {len(available_maps)} 张地图可用：")
for map in available_maps:
    print(map)