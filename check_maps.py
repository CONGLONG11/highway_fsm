# 文件: ~/Highway_FSM/src/check_maps.py
import carla

def main():
    try:
        # 连接到你的 CARLA 服务器
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(5.0)

        # 获取所有可用的地图列表
        available_maps = client.get_available_maps()
        
        print("="*30)
        print("当前 CARLA 模拟器可用的地图列表：")
        print("="*30)
        
        # 格式化输出，只显示地图名
        for map_path in available_maps:
            map_name = map_path.split('/')[-1]
            print(f"- {map_name}")
            
        print("\n请从以上列表中选择一个地图，并修改 main.py 中的 'target_map' 变量。")
        print("推荐用于高速测试的地图：Town04, Town05, Town10HD_Opt")

    except Exception as e:
        print(f"连接 CARLA 失败: {e}")
        print("请确保你的 CARLA 模拟器正在 Windows 上运行。")

if __name__ == '__main__':
    main()