"""
滚动平均滤波
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义一个函数，用于对曲线进行移动平均滤波，以平滑曲线
def moving_average(curve, radius):
    window_size = 2 * radius + 1  # 窗口大小
    f = np.ones(window_size) / window_size  # 创建一个平均滤波器
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')  # 对曲线进行边缘填充
    curve_smoothed = np.convolve(curve_pad, f, mode='same')  # 应用卷积操作进行滤波
    curve_smoothed = curve_smoothed[radius:-radius]  # 去除填充的边缘
    return curve_smoothed

# 定义一个函数，用于平滑整个轨迹
def smooth_trajectory_with_futrue(trajectory, SmoothingRadius):
    smoothed_trajectory = np.copy(trajectory)  # 复制轨迹数组
    for i in range(3):  # 对轨迹的每个维度进行平滑处理
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=SmoothingRadius)
    return smoothed_trajectory

# 定义一个函数，用于修复由于变换导致的边界问题
def fix_border(frame):
    s = frame.shape  # 获取帧的尺寸
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)  # 创建一个旋转矩阵
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))  # 应用旋转和缩放
    return frame

# 使用队列来管理帧
class FrameQueue:
    def __init__(self, max_len):
        self.frames = deque(maxlen=max_len)  # 使用deque来缓存视频帧
        self.max_len = max_len

    def append(self, frame):
        """添加帧到队列中"""
        self.frames.append(frame)

    def pop_append(self, frame):
        """如果队列已满，则将队头帧弹出并返回，然后将新帧添加到队尾"""
        popped_frame = None
        # 如果队列已满，弹出队头元素并返回
        if len(self.frames) == self.max_len:
            popped_frame = self.frames.popleft()
        # 将新的帧添加到队尾
        self.frames.append(frame)
        return popped_frame

    def pop_appenddd(self, frame):
        """将新帧与队列中的所有帧做计算后加入队列，如果队列已满，弹出最旧的帧并返回"""
        popped_frame = None

        # 如果队列已满，弹出队头元素并返回
        if len(self.frames) == self.max_len:
            popped_frame = self.frames.popleft()

        # 将新帧添加到队尾
        self.frames.append(frame)

        # 如果队列已经有元素，进行一次计算（比如求平均值）
        if len(self.frames) > 1:
            # 这里可以根据需要计算其他的值，假设我们是做平均
            avg_frame = np.mean(np.array(self.frames), axis=0)  # 对所有帧求平均
            # 将平均值更新到队列中（你可以决定如何处理这些值）
            self.frames = deque([avg_frame] * len(self.frames), maxlen=self.max_len)  # 替换为平均后的值

        return popped_frame

    def get_all(self):
        """获取队列中的所有帧"""
        return list(self.frames)

# 打开摄像头（或者在线视频流）
cap = cv2.VideoCapture(0)  # 使用默认摄像头，或者替换为在线视频流 URL
# 检查视频是否成功打开
if not cap.isOpened():
    print("Error opening video file")
    exit()
# 获取视频的总帧数、宽、高和帧率
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 设置平滑半径
SMOOTHING_RADIUS = 50
# 读取视频的第一帧
_, prev = cap.read()
if prev is None:
    print("Error reading video file")
    cap.release()
    exit()
# 将第一帧转换为灰度图
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
# 网格大小
grid_size = [4, 4]
# 创建平滑参数队列
parameters_queue = FrameQueue(max_len=250)
# 用于存储每帧滤波前和滤波后的累计变换数据（便于后续画图）
original_dx = []
original_dy = []
original_da = []
smoothed_dx = []
smoothed_dy = []
smoothed_da = []

# 主循环，处理视频流
while True:
    success, curr = cap.read()
    if not success:
        break

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # 筛选出成功跟踪的点
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # 如果跟踪的点太少，则使用单位矩阵
    if prev_pts.shape[0] < 4:
        m = np.eye(2, 3, dtype=np.float32)
    else:
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)  # 估计仿射变换矩阵

    if m is None:
        m = np.eye(2, 3, dtype=np.float32)

    # 提取变换参数
    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])

    """
    光流重跟踪算法
    """
    # grid_w = prev_gray.shape[1] // grid_size[0]
    # grid_h = prev_gray.shape[0] // grid_size[1]
    # grid_trajectories = np.zeros(3, np.float32)  # 一维数组，存储 dx, dy, da
    #
    # grid_point = []
    # for y in range(grid_size[1]):
    #     for x in range(grid_size[0]):
    #         grid_x, grid_y = x * grid_w, y * grid_h
    #         grid_roi_prev = prev_gray[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w]
    #         harris_response = cv2.cornerHarris(grid_roi_prev, 2, 3, 0.04)
    #         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(harris_response)
    #         grid_point.append((grid_x + max_loc[0], grid_y + max_loc[1]))
    #
    # good_point = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=30, blockSize=3)
    #
    # # 将 grid_point 和 good_point 分别转换为适合光流计算的格式
    # prev_pts = np.array(grid_point, dtype=np.float32).reshape(-1, 1, 2)  # 网格角点
    # good_pts = np.array(good_point, dtype=np.float32).reshape(-1, 1, 2)  # 全局角点
    #
    # # 正向光流跟踪
    # grid_curr_pts, status_grid, err_grid = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    # good_curr_pts, status_good, err_good = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, good_pts, None)
    #
    # # 反向光流跟踪
    # grid_back_pts, _, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, grid_curr_pts, None)
    # good_back_pts, _, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, good_curr_pts, None)
    #
    # # 重跟踪误差计算，同时筛选前一帧的点和后一帧的点
    # reliable_prev_pts = []
    # reliable_curr_pts = []
    #
    # for i, (p0, p1, p2) in enumerate(zip(prev_pts, grid_curr_pts, grid_back_pts)):
    #     if status_grid[i] and np.linalg.norm(p0 - p2) < 1.0:  # 判定误差阈值
    #         reliable_prev_pts.append(p0)
    #         reliable_curr_pts.append(p1)
    #
    # for i, (p0, p1, p2) in enumerate(zip(good_pts, good_curr_pts, good_back_pts)):
    #     if status_good[i] and np.linalg.norm(p0 - p2) < 1.0:
    #         reliable_prev_pts.append(p0)
    #         reliable_curr_pts.append(p1)
    #
    # # 转换为 numpy 格式，确保匹配的点数量一致
    # reliable_prev_pts = np.array(reliable_prev_pts, dtype=np.float32).reshape(-1, 1, 2)
    # reliable_curr_pts = np.array(reliable_curr_pts, dtype=np.float32).reshape(-1, 1, 2)
    #
    # # 计算仿射变换矩阵
    # M, _ = cv2.estimateAffinePartial2D(reliable_prev_pts, reliable_curr_pts)
    # if M is None:
    #     M = np.eye(2, 3, dtype=np.float32)
    #
    # dx = M[0, 2]
    # dy = M[1, 2]
    # da = np.arctan2(M[1, 0], M[0, 0])

    # 将当前帧加入队列
    parameters_queue.append([dx, dy, da])

    # 从队列中获取已经缓存的帧（假设队列已满）
    smoothed_parameters = parameters_queue.get_all()

    # 如果队列有足够的帧，进行平滑处理
    if len(smoothed_parameters) == parameters_queue.max_len:
        # 计算累积变换轨迹
        transforms = np.array(smoothed_parameters)
        trajectory = np.cumsum(transforms, axis=0)

        # 平滑变换轨迹
        smoothed_trajectory = smooth_trajectory_with_futrue(trajectory, SmoothingRadius=25)

        # 计算平滑轨迹和原始轨迹的差异
        difference = smoothed_trajectory - trajectory

        # 更新变换数组
        transforms_smooth = transforms + difference

        # 提取最终平滑后的变换参数
        final_smooth_dx, final_smooth_dy, final_smooth_da = transforms_smooth[0]
    else:
        final_smooth_dx, final_smooth_dy, final_smooth_da = dx, dy, da

    # 记录数据（用于后续画图）
    original_dx.append(dx)
    original_dy.append(dy)
    original_da.append(da)
    smoothed_dx.append(final_smooth_dx)
    smoothed_dy.append(final_smooth_dy)
    smoothed_da.append(final_smooth_da)

    # 构造仿射变换矩阵
    m_smooth = np.zeros((2, 3), np.float32)
    m_smooth[0, 0] = np.cos(final_smooth_da)
    m_smooth[0, 1] = -np.sin(final_smooth_da)
    m_smooth[1, 0] = np.sin(final_smooth_da)
    m_smooth[1, 1] = np.cos(final_smooth_da)
    m_smooth[0, 2] = final_smooth_dx
    m_smooth[1, 2] = final_smooth_dy

    # 应用变换到当前帧
    frame_stabilized = cv2.warpAffine(curr, m_smooth, (w, h))

    # 修复变换后的边界问题
    frame_stabilized = fix_border(frame_stabilized)

    # 将原始帧和平滑帧并排放置
    frame_out = cv2.hconcat([curr, frame_stabilized])

    # 显示输出
    cv2.imshow("Original vs Stabilized", frame_out)

    # 按‘q’键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = curr_gray

# 释放所有资源
cap.release()
cv2.destroyAllWindows()

# -------------------------------
# 绘制滤波前与滤波后的累计变换曲线
# -------------------------------
frames = range(len(original_dx))
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(frames, original_dx, 'r-', label="原始 dx")
plt.plot(frames, smoothed_dx, 'b-', label="平滑 dx")
plt.ylabel("dx")
plt.legend()
plt.title("累计平移与旋转数据对比")

plt.subplot(3, 1, 2)
plt.plot(frames, original_dy, 'r-', label="原始 dy")
plt.plot(frames, smoothed_dy, 'b-', label="平滑 dy")
plt.ylabel("dy")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(frames, original_da, 'r-', label="原始 da")
plt.plot(frames, smoothed_da, 'b-', label="平滑 da")
plt.xlabel("帧数")
plt.ylabel("da")
plt.legend()

plt.tight_layout()
plt.show()

