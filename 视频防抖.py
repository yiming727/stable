"""
光流法
"""

# 导入必要的库
import numpy as np
import cv2

# 定义一个函数，用于对曲线进行移动平均滤波，以平滑曲线
def moving_average(curve, radius):
    window_size = 2 * radius + 1  # 窗口大小
    f = np.ones(window_size) / window_size  # 创建一个平均滤波器
    curve_pad = np.pad(curve, (radius, radius), 'edge')  # 对曲线进行边缘填充
    curve_smoothed = np.convolve(curve_pad, f, mode='same')  # 应用卷积操作进行滤波
    curve_smoothed = curve_smoothed[radius:-radius]  # 去除填充的边缘
    return curve_smoothed

# 定义一个函数，用于平滑整个轨迹
def smooth_trajectory(trajectory):
    smoothed_trajectory = np.copy(trajectory)  # 复制轨迹数组
    for i in range(3):  # 对轨迹的每个维度进行平滑处理
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

# 定义一个函数，用于修复由于变换导致的边界问题
def fix_border(frame):
    s = frame.shape  # 获取帧的尺寸
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)  # 创建一个旋转矩阵
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))  # 应用旋转和缩放
    return frame

# 设置平滑半径
SMOOTHING_RADIUS = 100

# 打开视频文件
cap = cv2.VideoCapture(r'../Data/24.mp4')
# cap = cv2.VideoCapture(r'./original_output.mp4')

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
# out = cv2.VideoWriter('../Data/result1.mp4', fourcc, fps, (2 * w, h))
out = cv2.VideoWriter('./Basic.mp4', fourcc, fps, (w, h))

# 读取视频的第一帧
_, prev = cap.read()
if prev is None:
    print("Error reading video file")
    cap.release()
    exit()

# 将第一帧转换为灰度图
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# 初始化变换数组
transforms = np.zeros((n_frames - 1, 3), np.float32)

# 遍历视频的每一帧，计算帧间变换
for i in range(n_frames - 2):
    # Shi-Tomasi 角点检测方法，是 Harris 角点检测的改进版本。
    # prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    # ORB 特征点检测
    orb = cv2.ORB_create()
    keypoints = orb.detect(prev_gray, None)
    prev_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    success, curr = cap.read()
    if not success:
        break
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
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

    # 保存变换
    transforms[i] = [dx, dy, da]
    prev_gray = curr_gray
    print(f"Frame: {i}/{n_frames - 2} - Tracked points: {len(prev_pts)}")
print(transforms)
# 计算累积变换轨迹
trajectory = np.cumsum(transforms, axis=0)
# 平滑变换轨迹
smoothed_trajectory = smooth_trajectory(trajectory)
# 计算平滑轨迹与原始轨迹的差异
difference = smoothed_trajectory - trajectory
# 更新变换数组
# 将原始变换与差异相结合，以获得平滑的变换
transforms_smooth = transforms + difference
# 重置视频读取位置到第一帧
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# 遍历视频帧，应用平滑变换
for i in range(n_frames - 2):
    success, frame = cap.read()
    if not success:
        break
    # 获取平滑变换参数
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]
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
    # out.write(frame_out)
    out.write(frame_stabilized)

# 释放视频读取和写入对象
cap.release()
out.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
