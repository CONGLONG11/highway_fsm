"""
模糊逻辑引擎 (Fuzzy Logic Engine)
======================================
创新点 1 的核心实现。

传统 FSM 使用硬阈值判断（如 if dist < 30），在阈值边界处
存在频繁的状态切换（决策抖动）。本模块通过模糊隶属度函数
将连续物理量映射为模糊语义，再通过模糊推理规则得到一个
连续的"变道意愿度"值，消除了硬阈值导致的边界问题。

原理:
  1. 模糊化 (Fuzzification): 将 distance, delta_v 映射为隶属度
  2. 模糊推理 (Inference): 应用 IF-THEN 规则
  3. 解模糊 (Defuzzification): 加权平均得到输出值
"""
import numpy as np

class TriangularMF:
    """三角形隶属度函数 (Triangular Membership Function)"""

    def __init__(self, a, b, c):
        """
        三角形隶属度: 在 b 处取最大值 1.0
              /\\
             /  \\
            /    \\
        ---a--b--c---
        :param a: 左脚
        :param b: 峰值
        :param c: 右脚
        """
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def evaluate(self, x):
        """计算输入 x 的隶属度 [0, 1]"""
        if x <= self.a or x >= self.c:
            return 0.0
        elif x <= self.b:
            return (x - self.a) / (self.b - self.a + 1e-9)
        else:
            return (self.c - x) / (self.c - self.b + 1e-9)

class TrapezoidalMF:
    """梯形隶属度函数"""

    def __init__(self, a, b, c, d):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def evaluate(self, x):
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.a < x < self.b:
            return (x - self.a) / (self.b - self.a + 1e-9)
        elif self.b <= x <= self.c:
            return 1.0
        else:
            return (self.d - x) / (self.d - self.c + 1e-9)

class FuzzyLaneChangeEngine:
    """
    模糊变道意愿推理引擎

    输入变量:
      - front_distance: 前车距离 (m)
      - delta_speed: 速度差 (km/h), 正值表示前车比我快

    输出变量:
      - lane_change_desire: 变道意愿度 [0.0, 1.0]
        0.0 = 完全不想变道
        1.0 = 非常想变道
    """

    def __init__(self, config=None):
        if config is None:
            config = {}

        # === 输入: 前车距离 ===
        dist_close = config.get('dist_close_range', [0, 12, 25])
        dist_medium = config.get('dist_medium_range', [20, 35, 50])
        dist_far = config.get('dist_far_range', [45, 60, 120])

        self.dist_mfs = {
            'close': TriangularMF(*dist_close),
            'medium': TriangularMF(*dist_medium),
            'far': TriangularMF(*dist_far),
        }

        # === 输入: 速度差 (ego_speed - front_speed) ===
        # 正值 = 我比前车快 (需要变道超车)
        # 负值 = 前车比我快 (不需要变道)
        dv_slow = config.get('dv_negative_range', [-30, -15, 0])
        dv_same = config.get('dv_zero_range', [-5, 0, 5])
        dv_fast = config.get('dv_positive_range', [0, 15, 30])

        self.dv_mfs = {
            'slower': TriangularMF(*dv_slow),     # 我比前车慢
            'same': TriangularMF(*dv_same),        # 速度相近
            'faster': TriangularMF(*dv_fast),      # 我比前车快
        }

        # === 模糊推理规则表 ===
        # 规则格式: (dist_label, dv_label) -> desire_value
        # desire 值范围: 0.0(不变道) ~ 1.0(强烈变道)
        self.rules = {
            # 距离近
            ('close', 'faster'):  0.95,   # 近+我更快 → 强烈变道
            ('close', 'same'):    0.80,   # 近+同速 → 较想变道
            ('close', 'slower'):  0.50,   # 近+我较慢 → 稍想变道(跟车不舒服)

            # 距离中
            ('medium', 'faster'): 0.70,   # 中+我更快 → 想变道超车
            ('medium', 'same'):   0.30,   # 中+同速 → 不太想
            ('medium', 'slower'): 0.10,   # 中+我较慢 → 基本不想

            # 距离远
            ('far', 'faster'):    0.20,   # 远+我更快 → 略想(提前准备)
            ('far', 'same'):      0.05,   # 远+同速 → 不想
            ('far', 'slower'):    0.00,   # 远+我较慢 → 完全不想
        }

    def evaluate(self, front_distance, delta_speed):
        """
        计算变道意愿度

        :param front_distance: 前车距离 (m), 无前车时传入 999
        :param delta_speed: ego_speed - front_speed (km/h)
        :return: desire 值 [0.0, 1.0]
        """
        if front_distance > 100.0:
            return 0.0  # 前方无车，不需要变道

        # 1. 模糊化: 计算所有隶属度
        dist_grades = {k: mf.evaluate(front_distance) for k, mf in self.dist_mfs.items()}
        dv_grades = {k: mf.evaluate(delta_speed) for k, mf in self.dv_mfs.items()}

        # 2. 模糊推理: 对每条规则取 min(前件1, 前件2) 作为激活强度
        numerator = 0.0
        denominator = 0.0

        for (dist_label, dv_label), desire_val in self.rules.items():
            # 取交集 (AND): min 操作
            activation = min(dist_grades[dist_label], dv_grades[dv_label])

            if activation > 0:
                numerator += activation * desire_val
                denominator += activation

        # 3. 解模糊: 加权平均法 (Weighted Average Defuzzification)
        if denominator < 1e-9:
            return 0.0

        desire = numerator / denominator
        return np.clip(desire, 0.0, 1.0)

    def get_debug_info(self, front_distance, delta_speed):
        """返回调试信息，用于论文中的可视化"""
        dist_grades = {k: round(mf.evaluate(front_distance), 3)
                       for k, mf in self.dist_mfs.items()}
        dv_grades = {k: round(mf.evaluate(delta_speed), 3)
                     for k, mf in self.dv_mfs.items()}
        desire = self.evaluate(front_distance, delta_speed)

        return {
            'front_distance': front_distance,
            'delta_speed': delta_speed,
            'dist_membership': dist_grades,
            'dv_membership': dv_grades,
            'desire': round(desire, 4)
        }

    def compute_desire(self, front_distance, delta_speed):
        """
        别名方法，供 fsm_decision.py 调用
        实际调用 evaluate()
        """
        return self.evaluate(front_distance, delta_speed)
