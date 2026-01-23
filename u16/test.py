"""
光流法视频稳像 - 支持 IRV 红外视频格式（16位精度版本）
"""

import numpy as np
import cv2
from keii_data_load import keii_data_load
from ir_color import ppbyIron


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
    # 使用 INTER_LINEAR 插值，保持 16 位精度
    frame = cv2.warpAffine(frame_u16, T, (s[1], s[0]),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    return frame


def u16_to_gray(img_u16):
    """将 16 位图像转换为 8 位灰度图（用于特征检测）"""
    img_min = img_u16.min()
    img_max = img_u16.max()
    if img_max > img_min:
        img_normalized = ((img_u16 - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_normalized = np.zeros_like(img_u16, dtype=np.uint8)
    return img_normalized


def u16_to_bgr(img_u16, rgb_list):
    """将 16 位图像转换为伪彩色 BGR 图像"""
    img_normalized = u16_to_gray(img_u16)

    h, w = img_normalized.shape
    T = img_normalized.reshape(1, h * w)

    R = rgb_list[T, 0].reshape(h, w)
    G = rgb_list[T, 1].reshape(h, w)
    B = rgb_list[T, 2].reshape(h, w)

    img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    img_bgr[:, :, 0] = B
    img_bgr[:, :, 1] = G
    img_bgr[:, :, 2] = R

    return img_bgr


# ============================================
# 主程序
# ============================================

SMOOTHING_RADIUS = 100
video_path = r'./20230831171237_00.IRV'

# 创建数据加载器
loader = keii_data_load()
rgb_list = ppbyIron

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

fps = 25

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./Basic_IRV_16bit.mp4', fourcc, fps, (width, height))

# ============================================
# 第一步：计算变换矩阵（使用 8 位灰度图）
# ============================================
print("\n=== 第一步：计算帧间变换 ===")

# 读取第一帧
with open(video_path, "rb") as f:
    prev_u16 = loader.Open_Frame_IRV(f, 0, width, height)

if prev_u16 is None:
    print("Error: 无法读取第一帧")
    exit()

prev_gray = u16_to_gray(prev_u16)
transforms = np.zeros((n_frames - 1, 3), np.float32)

# 计算帧间变换
for i in range(n_frames - 2):
    # 检测特征点
    orb = cv2.ORB_create()
    keypoints = orb.detect(prev_gray, None)

    if len(keypoints) == 0:
        print(f"警告: 帧 {i} 未检测到特征点")
        transforms[i] = [0, 0, 0]
        continue

    prev_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    # 读取下一帧
    with open(video_path, "rb") as f:
        curr_u16 = loader.Open_Frame_IRV(f, i + 1, width, height)

    if curr_u16 is None:
        print(f"警告: 无法读取帧 {i + 1}")
        break

    curr_gray = u16_to_gray(curr_u16)

    # 光流跟踪
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # 筛选成功跟踪的点
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # 估计仿射变换
    if prev_pts.shape[0] < 4:
        m = np.eye(2, 3, dtype=np.float32)
    else:
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

    if m is None:
        m = np.eye(2, 3, dtype=np.float32)

    # 提取变换参数
    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])

    transforms[i] = [dx, dy, da]
    prev_gray = curr_gray

    if (i + 1) % 10 == 0:
        print(f"进度: {i + 1}/{n_frames - 2} - 跟踪点数: {len(prev_pts)}")

print("变换矩阵计算完成")

# ============================================
# 第二步：平滑轨迹
# ============================================
print("\n=== 第二步：平滑轨迹 ===")

trajectory = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth_trajectory(trajectory)
difference = smoothed_trajectory - trajectory
transforms_smooth = transforms + difference

print("轨迹平滑完成")

# ============================================
# 第三步：在 16 位数据上应用变换
# ============================================
print("\n=== 第三步：应用变换（16位精度）===")

for i in range(n_frames - 2):
    # 读取 16 位原始帧
    with open(video_path, "rb") as f:
        frame_u16 = loader.Open_Frame_IRV(f, i, width, height)

    if frame_u16 is None:
        print(f"警告: 无法读取帧 {i}")
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

    # ⭐ 关键：在 16 位数据上应用变换
    frame_u16_stabilized = cv2.warpAffine(
        frame_u16,
        m,
        (width, height),
        flags=cv2.INTER_LINEAR,  # 线性插值
        borderMode=cv2.BORDER_REPLICATE  # 边界复制
    )

    # 修复边界
    frame_u16_stabilized = fix_border_u16(frame_u16_stabilized)

    # ⭐ 最后才转换为伪彩色 BGR（用于保存视频）
    frame_bgr = u16_to_bgr(frame_u16_stabilized, rgb_list)

    # 写入输出视频
    out.write(frame_bgr)

    if (i + 1) % 10 == 0:
        print(f"写入进度: {i + 1}/{n_frames - 2}")

out.release()
print("\n✅ 视频稳像完成！输出文件: Basic_IRV_16bit.mp4")
