"""
光流法视频稳像 - 支持 IRV 红外视频格式
"""

# 导入必要的库
import numpy as np
import cv2
from keii_data_load import keii_data_load
from ir_color import ppbyIron


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


# 将 u16 图像转换为 8 位灰度图
def u16_to_gray(img_u16):
    """将 16 位 AD 值转换为 8 位灰度图"""
    # 归一化到 0-255
    img_min = img_u16.min()
    img_max = img_u16.max()
    if img_max > img_min:
        img_normalized = ((img_u16 - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_normalized = np.zeros_like(img_u16, dtype=np.uint8)
    return img_normalized


# 将 u16 图像转换为伪彩色 BGR 图像
def u16_to_bgr(img_u16, rgb_list):
    """将 16 位 AD 值转换为伪彩色 BGR 图像"""
    # 归一化到 0-255
    img_normalized = u16_to_gray(img_u16)

    # 转换为伪彩色
    h = img_normalized.shape[0]
    w = img_normalized.shape[1]
    T = img_normalized.reshape(1, h * w)

    R = rgb_list[T, 0]
    G = rgb_list[T, 1]
    B = rgb_list[T, 2]
    R = R.reshape(h, w)
    G = G.reshape(h, w)
    B = B.reshape(h, w)

    # OpenCV 使用 BGR 格式
    img_bgr = np.zeros((h, w, 3), 'uint8')
    img_bgr[:, :, 0] = B
    img_bgr[:, :, 1] = G
    img_bgr[:, :, 2] = R

    return img_bgr


# ============================================
# 主程序
# ============================================

# 设置平滑半径
SMOOTHING_RADIUS = 100

# IRV 视频文件路径
video_path = r'./Data/20231017/20230831171237_00.IRV'

# 创建 KEII 数据加载器
loader = keii_data_load()

# 获取伪彩色表
rgb_list = ppbyIron

# 打开 IRV 视频文件，获取基本信息
print("正在读取视频信息...")
f = open(video_path, "rb")
width, height = loader.Read_IRVFrame_wh(f)
f.close()

f = open(video_path, "rb")
n_frames = loader.Read_TotalFrame_IRV(f)
f.close()

print(f"视频信息: {width}x{height}, 总帧数: {n_frames}")

# 设置输出帧率（IRV 格式没有帧率信息，默认设置为 25）
fps = 25

# 检查是否成功读取视频信息
if width == 0 or height == 0 or n_frames == 0:
    print("Error: 无法读取视频信息")
    exit()

# 设置视频输出格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 创建视频写入对象
out = cv2.VideoWriter('./Basic_IRV.mp4', fourcc, fps, (width, height))

# 读取视频的第一帧
print("读取第一帧...")
f = open(video_path, "rb")
prev_u16 = loader.Open_Frame_IRV(f, 0, width, height)
f.close()

if prev_u16 is None:
    print("Error: 无法读取第一帧")
    exit()

# 将第一帧转换为灰度图（用于光流计算）
prev_gray = u16_to_gray(prev_u16)

# 初始化变换数组
transforms = np.zeros((n_frames - 1, 3), np.float32)

# 遍历视频的每一帧，计算帧间变换
print("开始计算帧间变换...")
for i in range(n_frames - 2):
    # ORB 特征点检测
    orb = cv2.ORB_create()
    keypoints = orb.detect(prev_gray, None)

    if len(keypoints) == 0:
        print(f"警告: 帧 {i} 未检测到特征点")
        transforms[i] = [0, 0, 0]
        prev_gray = prev_gray  # 保持不变
        continue

    prev_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    # 读取下一帧
    f = open(video_path, "rb")
    curr_u16 = loader.Open_Frame_IRV(f, i + 1, width, height)
    f.close()

    if curr_u16 is None:
        print(f"警告: 无法读取帧 {i + 1}")
        break

    # 转换为灰度图
    curr_gray = u16_to_gray(curr_u16)

    # 光流跟踪
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

    if (i + 1) % 10 == 0:
        print(f"进度: {i + 1}/{n_frames - 2} - 跟踪点数: {len(prev_pts)}")

print("变换矩阵计算完成")
print(f"变换矩阵形状: {transforms.shape}")

# 计算累积变换轨迹
trajectory = np.cumsum(transforms, axis=0)

# 平滑变换轨迹
smoothed_trajectory = smooth_trajectory(trajectory)

# 计算平滑轨迹与原始轨迹的差异
difference = smoothed_trajectory - trajectory

# 更新变换数组
transforms_smooth = transforms + difference

# 应用平滑变换到每一帧
print("开始应用平滑变换...")
for i in range(n_frames - 2):
    # 读取当前帧
    f = open(video_path, "rb")
    frame_u16 = loader.Open_Frame_IRV(f, i, width, height)
    f.close()

    if frame_u16 is None:
        print(f"警告: 无法读取帧 {i}")
        break

    # 转换为伪彩色 BGR 图像
    frame_bgr = u16_to_bgr(frame_u16, rgb_list)

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
    frame_stabilized = cv2.warpAffine(frame_bgr, m, (width, height))

    # 修复变换后的边界问题
    frame_stabilized = fix_border(frame_stabilized)

    # 将处理后的帧写入输出视频
    out.write(frame_stabilized)

    if (i + 1) % 10 == 0:
        print(f"写入进度: {i + 1}/{n_frames - 2}")

# 释放视频写入对象
out.release()

print("视频稳像完成！输出文件: Basic_IRV.mp4")
