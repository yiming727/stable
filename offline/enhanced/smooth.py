import numpy as np

class TrajectorySmoother:
    def __init__(self, smoothing_radius=50):
        """
        :param smoothing_radius: 平滑半径
        """
        self.smoothing_radius = smoothing_radius

    def moving_average(self, curve):
        """
        对曲线进行移动平均滤波，以平滑曲线
        :param curve: 一维数组
        :return: 平滑后的曲线
        """
        radius = self.smoothing_radius
        window_size = 2 * radius + 1
        f = np.ones(window_size) / window_size
        curve_pad = np.pad(curve, (radius, radius), 'edge')
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        curve_smoothed = curve_smoothed[radius:-radius]
        return curve_smoothed

    def smooth(self, trajectory):
        """
        平滑整个轨迹
        :param trajectory: shape=(N,3)的轨迹数组
        :return: 平滑后的轨迹
        """
        smoothed_trajectory = np.copy(trajectory)
        for i in range(3):
            smoothed_trajectory[:, i] = self.moving_average(trajectory[:, i])
            smoothed_trajectory[:, i] = self.moving_average(smoothed_trajectory[:, i])
            smoothed_trajectory[:, i] = self.moving_average(smoothed_trajectory[:, i])
        return smoothed_trajectory