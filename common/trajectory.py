"""
相机运动轨迹平滑。

来源（4 份重复实现，函数体一致，只有移动平均的**遍数**不同）：

    passes=2  offline/basic/src/stable.py       TrajectorySmoother.smooth
    passes=2  offline/basic/src/stable1.py      TrajectorySmoother.smooth
    passes=2  offline/basic/u16/stable.py       TrajectorySmoother.smooth
    passes=3  offline/enhanced/smooth.py        TrajectorySmoother.smooth

注意：遍数是**唯一**差异，且会改变数值结果，所以做成参数而非常量。
各处默认值保持各自原值，不要"统一"。
"""

import numpy as np

DEFAULT_RADIUS = 50
DEFAULT_PASSES = 2


def moving_average(curve, radius=DEFAULT_RADIUS):
    """对一维曲线做移动平均（两端用边缘值 padding）。

    参数:
        curve:  一维数组
        radius: 平滑半径，窗口长度为 2*radius+1
    返回:
        平滑后的曲线，长度与输入相同
    """
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    return curve_smoothed[radius:-radius]


class TrajectorySmoother:
    """轨迹平滑器：对 (N, 3) 的累积相机轨迹做移动平均滤波。

    参数:
        smoothing_radius: 平滑半径，默认 50
        passes:           每个维度做几遍移动平均，默认 2
                         （offline/enhanced/smooth.py 用的是 3）
    """

    def __init__(self, smoothing_radius=DEFAULT_RADIUS, passes=DEFAULT_PASSES):
        self.smoothing_radius = smoothing_radius
        self.passes = passes

    def moving_average(self, curve):
        """对单条曲线做一遍移动平均。"""
        return moving_average(curve, self.smoothing_radius)

    def smooth(self, trajectory):
        """平滑整个轨迹（x, y, angle 三个维度）。

        参数:
            trajectory: shape=(N, 3)，每行为 (dx, dy, da)
        返回:
            平滑后的轨迹，shape=(N, 3)
        """
        smoothed_trajectory = np.copy(trajectory)
        for i in range(3):
            # 第一遍基于原始轨迹，其余遍基于上一遍结果——与原实现逐行对应
            smoothed_trajectory[:, i] = self.moving_average(trajectory[:, i])
            for _ in range(self.passes - 1):
                smoothed_trajectory[:, i] = self.moving_average(smoothed_trajectory[:, i])
        return smoothed_trajectory
