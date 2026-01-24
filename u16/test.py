"""
光流法视频稳像 - 支持 IRV 红外视频格式（16位精度版本 - 无闪烁）
"""

import numpy as np
import cv2
from keii_data_load import keii_data_load


def moving_average(curve, radius):
    """移动平均滤波"""
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed


def smooth_trajectory(trajectory):
    """平滑轨迹"""
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory


def fix_border_u16(frame_u16):
    """修复 16 位图像的边界（使用缩放）"""
    s = frame_u16.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)
    frame = cv2.warpAffine(frame_u16, T, (s[1], s[0]),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    return frame


def u16_to_gray_global(img_u16, global_min, global_max):
    """
    使用全局 min/max 转换为 8 位灰度图 - 避免闪烁

    参数:
        img_u16: 16位输入图像
        global_min: 全局最小值
        global_max: 全局最大值
    """
    if global_max > global_min:
        img_normalized = np.clip(
            (img_u16 - global_min) / (global_max - global_min) * 255,
            0,
            255
        ).astype(np.uint8)
    else:
        img_normalized = np.zeros_like(img_u16, dtype=np.uint8)
    return img_normalized


def compute_global_range(video_path, loader, width, height, n_frames, sample_rate=10):
    """
    计算视频的全局温度范围

    参数:
        sample_rate: 采样率（每隔多少帧采样一次）
    """
    print("正在计算全局温度范围...")
    global_min = np.inf
    global_max = -np.inf

    sample_indices = range(0, n_frames, sample_rate)

    for i in sample_indices:
        with open(video_path, "rb") as f:
            frame_u16 = loader.Open_Frame_IRV(f, i, width, height)

        if frame_u16 is not None:
            global_min = min(global_min, frame_u16.min())
            global_max = max(global_max, frame_u16.max())

        if (i + 1) % 100 == 0:
            print(f"  采样进度: {i + 1}/{n_frames}")

    print(f"全局温度范围: {global_min} - {global_max}")
    return global_min, global_max


SMOOTHING_RADIUS = 100
video_path = r'./20230831171237_00.IRV'

# 创建数据加载器
loader = keii_data_load()

# 读取视频信息
print("正在读取视频信息...")
with open(video_path, "rb") as f:
    width, height = loader.Read_IRVFrame_wh(f)

with open(video_path, "rb") as f:
    n_frames = loader.Read_TotalFrame_IRV(f)

print(f"视频信息: {width}x{height}, 总帧数: {n_frames}")

if width == 0 or height == 0 or n_frames == 0:
    print("Error: 无法读取视频信息")
    exit()

# 计算全局温度范围
global_min, global_max = compute_global_range(
    video_path, loader, width, height, n_frames, sample_rate=10
)

fps = 25

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./Basic_IRV_Gray.mp4', fourcc, fps, (width, height), isColor=False)

# 读取第一帧
with open(video_path, "rb") as f:
    prev_u16 = loader.Open_Frame_IRV(f, 0, width, height)

if prev_u16 is None:
    print("Error: 无法读取第一帧")
    exit()

# 使用全局归一化
prev_gray = u16_to_gray_global(prev_u16, global_min, global_max)
transforms = np.zeros((n_frames - 1, 3), np.float32)

# 计算帧间变换
print("\n正在计算帧间变换...")
for i in range(n_frames - 2):
    orb = cv2.ORB_create()
    keypoints = orb.detect(prev_gray, None)

    if len(keypoints) == 0:
        print(f"警告: 帧 {i} 未检测到特征点")
        transforms[i] = [0, 0, 0]
        continue

    prev_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    with open(video_path, "rb") as f:
        curr_u16 = loader.Open_Frame_IRV(f, i + 1, width, height)

    if curr_u16 is None:
        print(f"警告: 无法读取帧 {i + 1}")
        break

    # 使用全局归一化
    curr_gray = u16_to_gray_global(curr_u16, global_min, global_max)

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    if prev_pts.shape[0] < 4:
        m = np.eye(2, 3, dtype=np.float32)
    else:
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

    if m is None:
        m = np.eye(2, 3, dtype=np.float32)

    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])

    transforms[i] = [dx, dy, da]
    prev_gray = curr_gray

    if (i + 1) % 10 == 0:
        print(f"进度: {i + 1}/{n_frames - 2} - 跟踪点数: {len(prev_pts)}")

print("\n正在平滑轨迹...")
trajectory = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth_trajectory(trajectory)
difference = smoothed_trajectory - trajectory
transforms_smooth = transforms + difference

print("\n正在生成稳定视频...")
for i in range(n_frames - 2):
    with open(video_path, "rb") as f:
        frame_u16 = loader.Open_Frame_IRV(f, i, width, height)

    if frame_u16 is None:
        print(f"警告: 无法读取帧 {i}")
        break

    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    frame_u16_stabilized = cv2.warpAffine(
        frame_u16,
        m,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    frame_u16_stabilized = fix_border_u16(frame_u16_stabilized)

    # 使用全局归一化
    frame_gray = u16_to_gray_global(frame_u16_stabilized, global_min, global_max)

    out.write(frame_gray)

    if (i + 1) % 10 == 0:
        print(f"写入进度: {i + 1}/{n_frames - 2}")

out.release()
