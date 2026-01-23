"""
网格重构
"""

import numpy as np
import cv2

# from scipy.signal import savgol_filter

# 定义一个函数，用于对曲线进行移动平均滤波，以平滑曲线
def moving_average(curve, radius):
    window_size = 2 * radius + 1  # 窗口大小
    f = np.ones(window_size) / window_size  # 创建一个平均滤波器
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')  # 对曲线进行边缘填充
    curve_smoothed = np.convolve(curve_pad, f, mode='same')  # 应用卷积操作进行滤波
    curve_smoothed = curve_smoothed[radius:-radius]  # 去除填充的边缘
    return curve_smoothed

# 定义一个函数，用于平滑整个轨迹
def smooth_trajectory(trajectory):
    smoothed_trajectory = np.copy(trajectory)  # 复制轨迹数组
    for i in range(3):  # 对轨迹的每个维度进行平滑处理
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=SMOOTHING_RADIUS)
        smoothed_trajectory[:, i] = moving_average(smoothed_trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

# 定义一个函数，用于修复由于变换导致的边界问题
def fix_border(frame):
    s = frame.shape  # 获取帧的尺寸
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.21)  # 创建一个旋转矩阵
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))  # 应用旋转和缩放

    # # 创建一个掩模，标记黑色区域
    # mask = np.zeros(frame.shape, dtype=np.uint8)
    # mask[frame == 0] = 255  # 标记黑色区域
    # # 将三通道掩模转换为灰度掩模（如果每个通道值相同）
    # mask = np.mean(mask, axis=2).astype(np.uint8)  # 将RGB掩模转换为灰度图
    #
    # # 使用inpainting修复黑色区域
    # frame_inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    return frame

# 设置平滑半径
SMOOTHING_RADIUS = 50
# 设置网格大小
GRID_SIZE = (4, 4)  # 4x4网格

# 设置帧间变换的阈值
MAX_TRANSLATION_THRESHOLD = 15  # 最大平移阈值：单位像素
MAX_ROTATION_THRESHOLD = np.deg2rad(15)  # 最大旋转阈值：单位弧度（10度）

# 打开视频文件
cap = cv2.VideoCapture(r'../07.mp4')
# 检查视频是否成功打开
if not cap.isOpened():
    print("Error opening video file")
    exit()
# 获取视频的总帧数、宽、高和帧率
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# 设置视频输出格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 创建视频写入对象
out = cv2.VideoWriter('../Basic07.mp4', fourcc, fps, (w, h))
# 读取视频的第一帧
_, prev = cap.read()
if prev is None:
    print("Error reading video file")
    cap.release()
    exit()
# 将第一帧转换为灰度图
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
# 初始化每个网格的变换轨迹
grid_trajectories = np.zeros((n_frames - 1, 3), np.float32)  # 每个网格的变换轨迹

# 为每个网格计算光流和变换
grid_w = w // GRID_SIZE[0]
grid_h = h // GRID_SIZE[1]

# 初始化上一帧成功跟踪的特征点数
prev_tracked_points_count = 0  # 确保变量在开始时初始化为0
max_feature_points_count = 0  # 用于记录整个视频的最大特征点数

# 遍历视频的每一帧，计算每个网格的帧间变换
for i in range(n_frames - 2):
    grid_point = []
    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            grid_idx = y * GRID_SIZE[0] + x
            # 每个网格的左上角坐标
            grid_x, grid_y = x * grid_w, y * grid_h
            grid_roi_prev = prev_gray[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w]
            # 使用 Harris 角点检测在当前网格内查找最显著的角点
            harris_response = cv2.cornerHarris(grid_roi_prev, 2, 3, 0.04)
            # 找到响应值最大的角点
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(harris_response)
            # 将网格的角点添加到 grid_points 中
            grid_point.append((grid_x + max_loc[0], grid_y + max_loc[1]))

    good_point = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    success, curr = cap.read()
    if not success:
        break
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # 将 grid_point 和 good_point 分别转换为适合光流计算的格式
    prev_pts = np.array(grid_point, dtype=np.float32).reshape(-1, 1, 2)  # 网格角点
    good_pts = np.array(good_point, dtype=np.float32).reshape(-1, 1, 2)  # 全局角点

    # 特征点跟踪
    grid_curr_pts, status_grid, err_grid = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    good_curr_pts, status_good, err_good = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, good_pts, None)
    # # 筛选出成功跟踪的点
    # idx1 = np.where(status_grid == 1)[0]
    # prev_pts = prev_pts[idx1]
    # grid_curr_pts = grid_curr_pts[idx1]
    # idx2 = np.where(status_good == 1)[0]
    # good_pts = good_pts[idx2]
    # good_curr_pts = good_curr_pts[idx2]

    # 合并网格点和全局点
    prev_pts = np.concatenate([prev_pts, good_pts], axis=0)  # 合并网格点和全局点
    curr_pts = np.concatenate([grid_curr_pts, good_curr_pts], axis=0)

    # 计算四个角点之间的单应性矩阵
    M, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

    if M is None:
        M = np.eye(2, 3, dtype=np.float32)
    else:
        # 提取变换参数：平移、旋转等
        dx = M[0, 2]
        dy = M[1, 2]
        da = np.arctan2(M[1, 0], M[0, 0])

        # # 判断帧间平移和旋转量是否超过阈值
        # if abs(dx) > MAX_TRANSLATION_THRESHOLD or abs(dy) > MAX_TRANSLATION_THRESHOLD or abs(da) > MAX_ROTATION_THRESHOLD:
        #     print(f"Skipping frame {i+1} due to excessive motion: Translation ({dx}, {dy}), Rotation ({da})")
        #     continue  # 如果变换量超出阈值，跳过当前帧的消抖处理

        # 存储变换轨迹
        grid_trajectories[i, 0] = dx  # x 平移
        grid_trajectories[i, 1] = dy  # y 平移
        grid_trajectories[i, 2] = da  # 旋转角度
    prev_gray = curr_gray
    # print(f"Processed frame {i + 1}/{n_frames - 2}")

# 计算累积变换轨迹
trajectory = np.cumsum(grid_trajectories, axis=0)
# 平滑变换轨迹
smoothed_trajectory = smooth_trajectory(trajectory)
# 计算平滑轨迹与原始轨迹的差异
difference = smoothed_trajectory - trajectory
# 更新变换数组
grid_trajectories = grid_trajectories + difference
# 重置视频读取位置到第一帧
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in range(n_frames - 2):
    success, frame = cap.read()
    if not success:
        break
    # 获取平滑变换参数
    dx = grid_trajectories[i, 0]
    dy = grid_trajectories[i, 1]
    da = grid_trajectories[i, 2]
    # 构造仿射变换矩阵
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy
    # 应用变换到当前帧
    frame_stabilized = cv2.warpAffine(frame, m, (w, h))
    # 修复变换后的边界问题
    frame_stabilized = fix_border(frame_stabilized)
    # 将原始帧和平滑帧并排放置
    frame_out = cv2.hconcat([frame, frame_stabilized])
    #将处理后的帧写入输出视频
    out.write(frame_stabilized)

# 释放视频读取和写入对象
cap.release()
out.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()

