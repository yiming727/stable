"""
光流法视频稳像 - 支持 IRV 红外视频格式（网格法增强版）
结合网格特征点 + 全局特征点，提高稳像鲁棒性
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


def smooth_trajectory(trajectory, radius):
    """平滑轨迹（支持多次平滑）"""
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        # 多次平滑，提高平滑效果
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=radius)
        smoothed_trajectory[:, i] = moving_average(smoothed_trajectory[:, i], radius=radius)
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
    """计算视频的全局温度范围"""
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


def extract_grid_points(gray_image, grid_size=(4, 4)):
    """
    在图像上均匀分布网格，提取每个网格的最佳特征点

    参数:
        gray_image: 灰度图像
        grid_size: 网格大小 (rows, cols)

    返回:
        grid_points: 网格特征点列表
    """
    h, w = gray_image.shape
    grid_h = h // grid_size[0]
    grid_w = w // grid_size[1]

    grid_points = []

    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            # 计算当前网格的区域
            grid_y = y * grid_h
            grid_x = x * grid_w

            # 确保不超出图像边界
            grid_y_end = min(grid_y + grid_h, h)
            grid_x_end = min(grid_x + grid_w, w)

            # 提取网格区域
            grid_roi = gray_image[grid_y:grid_y_end, grid_x:grid_x_end]

            # 使用 Harris 角点检测找到最显著的角点
            harris_response = cv2.cornerHarris(grid_roi, 2, 3, 0.04)

            # 找到响应值最大的角点
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(harris_response)

            # 将网格局部坐标转换为全局坐标
            global_x = grid_x + max_loc[0]
            global_y = grid_y + max_loc[1]

            grid_points.append((global_x, global_y))

    return np.array(grid_points, dtype=np.float32)


def extract_good_features(gray_image, max_corners=200, quality_level=0.01, min_distance=30):
    """
    提取全局的优质特征点

    参数:
        gray_image: 灰度图像
        max_corners: 最大角点数
        quality_level: 质量水平
        min_distance: 最小距离

    返回:
        good_points: 优质特征点列表
    """
    good_points = cv2.goodFeaturesToTrack(
        gray_image,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=3
    )

    if good_points is not None:
        return good_points.reshape(-1, 2)
    else:
        return np.array([], dtype=np.float32).reshape(0, 2)


# ==================== 配置参数 ====================
SMOOTHING_RADIUS = 50  # 平滑半径
GRID_SIZE = (4, 4)  # 网格大小（4x4）
MAX_CORNERS = 200  # 全局特征点最大数量
QUALITY_LEVEL = 0.01  # 特征点质量水平
MIN_DISTANCE = 30  # 特征点最小距离

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
out = cv2.VideoWriter('./Basic_IRV_Gray_Grid.mp4', fourcc, fps, (width, height), isColor=False)

if not out.isOpened():
    print("警告: 视频写入器打开失败，尝试使用其他编码器...")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./Basic_IRV_Gray_Grid.avi', fourcc, fps, (width, height), isColor=False)
    if not out.isOpened():
        print("Error: 无法创建视频写入器")
        exit()

# 读取第一帧
with open(video_path, "rb") as f:
    prev_u16 = loader.Open_Frame_IRV(f, 0, width, height)

if prev_u16 is None:
    print("Error: 无法读取第一帧")
    exit()

# 转换为 uint8 用于特征检测
prev_gray = u16_to_gray_global(prev_u16, global_min, global_max)
transforms = np.zeros((n_frames - 1, 3), np.float32)

# ==================== 计算帧间变换 ====================
print("\n正在计算帧间变换（网格法 + 全局特征点）...")
for i in range(n_frames - 2):
    # 1. 提取网格特征点
    grid_points = extract_grid_points(prev_gray, grid_size=GRID_SIZE)

    # 2. 提取全局优质特征点
    good_points = extract_good_features(
        prev_gray,
        max_corners=MAX_CORNERS,
        quality_level=QUALITY_LEVEL,
        min_distance=MIN_DISTANCE
    )

    # 读取下一帧
    with open(video_path, "rb") as f:
        curr_u16 = loader.Open_Frame_IRV(f, i + 1, width, height)

    if curr_u16 is None:
        print(f"警告: 无法读取帧 {i + 1}")
        break

    # 转换为 uint8
    curr_gray = u16_to_gray_global(curr_u16, global_min, global_max)

    # 分别跟踪网格点和全局点
    # 网格点光流跟踪
    if len(grid_points) > 0:
        grid_prev_pts = grid_points.reshape(-1, 1, 2)
        grid_curr_pts, status_grid, err_grid = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, grid_prev_pts, None
        )
        # 筛选成功跟踪的网格点
        idx_grid = np.where(status_grid.flatten() == 1)[0]
        grid_prev_pts = grid_prev_pts[idx_grid]
        grid_curr_pts = grid_curr_pts[idx_grid]
    else:
        grid_prev_pts = np.array([], dtype=np.float32).reshape(0, 1, 2)
        grid_curr_pts = np.array([], dtype=np.float32).reshape(0, 1, 2)

    # 全局点光流跟踪
    if len(good_points) > 0:
        good_prev_pts = good_points.reshape(-1, 1, 2)
        good_curr_pts, status_good, err_good = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, good_prev_pts, None
        )
        # 筛选成功跟踪的全局点
        idx_good = np.where(status_good.flatten() == 1)[0]
        good_prev_pts = good_prev_pts[idx_good]
        good_curr_pts = good_curr_pts[idx_good]
    else:
        good_prev_pts = np.array([], dtype=np.float32).reshape(0, 1, 2)
        good_curr_pts = np.array([], dtype=np.float32).reshape(0, 1, 2)

    # 合并网格点和全局点
    if len(grid_prev_pts) > 0 and len(good_prev_pts) > 0:
        prev_pts = np.concatenate([grid_prev_pts, good_prev_pts], axis=0)
        curr_pts = np.concatenate([grid_curr_pts, good_curr_pts], axis=0)
    elif len(grid_prev_pts) > 0:
        prev_pts = grid_prev_pts
        curr_pts = grid_curr_pts
    elif len(good_prev_pts) > 0:
        prev_pts = good_prev_pts
        curr_pts = good_curr_pts
    else:
        # 没有跟踪到任何点，使用单位矩阵
        transforms[i] = [0, 0, 0]
        prev_gray = curr_gray
        print(f"警告: 帧 {i} 未跟踪到任何特征点")
        continue

    # 估计仿射变换
    if prev_pts.shape[0] >= 4:
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    else:
        m = None

    if m is None:
        m = np.eye(2, 3, dtype=np.float32)

    # 提取变换参数
    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])

    transforms[i] = [dx, dy, da]
    prev_gray = curr_gray

    if (i + 1) % 10 == 0:
        print(
            f"进度: {i + 1}/{n_frames - 2} - 网格点: {len(grid_prev_pts)}, 全局点: {len(good_prev_pts)}, 总计: {len(prev_pts)}")

# ==================== 平滑轨迹 ====================
print("\n正在平滑轨迹...")
trajectory = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth_trajectory(trajectory, radius=SMOOTHING_RADIUS)
difference = smoothed_trajectory - trajectory
transforms_smooth = transforms + difference

# ==================== 生成稳定视频 ====================
print("\n正在生成稳定视频...")
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

    # 在 uint16 上进行稳像变换
    frame_u16_stabilized = cv2.warpAffine(
        frame_u16,
        m,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # 修复边界
    frame_u16_stabilized = fix_border_u16(frame_u16_stabilized)

    # 输出时使用全局归一化
    frame_gray = u16_to_gray_global(frame_u16_stabilized, global_min, global_max)

    # 写入输出视频
    out.write(frame_gray)

    if (i + 1) % 10 == 0:
        print(f"写入进度: {i + 1}/{n_frames - 2}")

out.release()
print("\n✅ 处理完成！")
