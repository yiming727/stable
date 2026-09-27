"""
光流法视频稳像 - 支持 IRV 红外视频格式
动态网格法 + 自动裁剪增强版
网格大小根据视频分辨率自动调整，自动计算最优裁剪区域
"""

import argparse
import math
import numpy as np
import cv2
from keii_data_load import keii_data_load


# ============================================================================
# IRV 图像预处理：16 位转 8 位灰度
# ============================================================================

def u16_to_gray_hist_fuse(img_u16):
    """
    将 16 位红外温度图转换为 8 位灰度图（多区间融合自适应映射）

    采用三路不同统计区间分别归一化后取均值的融合策略：
      - 第一路：以帧均值为中心，增强中等灰度区域
      - 第二路：拉伸低灰度细节区间（5%~60% 百分位）
      - 第三路：拉伸高灰度细节区间（40%~95% 百分位）

    注意：该方法为逐帧自适应映射，仅用于特征检测与光流计算，
          稳像变换仍在原始 16 位域执行，以保留完整辐射温度信息。

    参数:
        img_u16: 16 位红外图像（numpy array）

    返回:
        u8_img: 8 位灰度图像（numpy array, uint8）
    """
    u16_channel_data = img_u16
    hist_fuse = []

    for c in range(3):
        if c == 0:
            # 增强中等灰度区域
            channel_mean = np.mean(u16_channel_data)
            bounds = (channel_mean - 5000, channel_mean + 5000)
        elif c == 1:
            # 压缩高灰度，拉伸低灰度（5%~60% 区间）
            lower = np.percentile(u16_channel_data, 5)
            upper = np.percentile(u16_channel_data, 60)
            bounds = (lower - 2000, upper + 500)
        else:
            # 压缩低灰度，拉伸高灰度（40%~95% 区间）
            lower = np.percentile(u16_channel_data, 40)
            upper = np.percentile(u16_channel_data, 95)
            bounds = (lower - 500, upper + 2000)

        channel_data = np.clip(u16_channel_data, bounds[0], bounds[1])
        equalized_channel = cv2.normalize(channel_data, None, 0, 1, cv2.NORM_MINMAX)
        hist_fuse.append(equalized_channel)

    u8_img = (
        (hist_fuse[0].astype(np.float32) + hist_fuse[1] + hist_fuse[2]) / 3 * 255
    ).astype(np.uint8)

    return u8_img


# ============================================================================
# 轨迹平滑类
# ============================================================================

class TrajectorySmoother:
    """
    轨迹平滑器

    使用双次移动平均滤波平滑相机运动轨迹，
    在保留低频运动趋势的同时抑制高频抖动分量。
    """

    def __init__(self, smoothing_radius=50):
        """
        初始化平滑器

        参数:
            smoothing_radius: 平滑半径，值越大平滑效果越强，
                              对应时间窗口为 (2*radius+1) 帧
        """
        self.smoothing_radius = smoothing_radius

    def moving_average(self, curve):
        """
        对一维曲线进行移动平均滤波

        采用边缘填充（edge padding）策略处理边界，
        避免边界效应导致轨迹头尾出现异常偏移。

        参数:
            curve: 一维数组，待平滑的曲线

        返回:
            平滑后的曲线（与输入等长）
        """
        radius = self.smoothing_radius
        window_size = 2 * radius + 1

        # 创建均值滤波核
        f = np.ones(window_size) / window_size

        # 边缘填充，避免边界效应
        curve_pad = np.pad(curve, (radius, radius), 'edge')

        # 卷积操作实现移动平均
        curve_smoothed = np.convolve(curve_pad, f, mode='same')

        # 去除填充部分，恢复原始长度
        curve_smoothed = curve_smoothed[radius:-radius]

        return curve_smoothed

    def smooth(self, trajectory):
        """
        平滑整个运动轨迹（dx、dy、da 三个分量）

        对每个分量连续执行两次移动平均，等效频率响应近似三角形窗，
        旁瓣抑制能力优于单次矩形窗，能更彻底消除高频抖动。

        参数:
            trajectory: shape=(N, 3) 的轨迹数组，每行为 (dx, dy, da)

        返回:
            smoothed_trajectory: 平滑后的轨迹，shape=(N, 3)
        """
        smoothed_trajectory = np.copy(trajectory)

        # 对每个维度分别执行两次移动平均
        for i in range(3):
            smoothed_trajectory[:, i] = self.moving_average(trajectory[:, i])
            smoothed_trajectory[:, i] = self.moving_average(smoothed_trajectory[:, i])

        return smoothed_trajectory


# ============================================================================
# 动态网格划分类
# ============================================================================

class GridCalculator:
    """
    动态网格计算器

    根据视频分辨率自动计算最优网格大小，
    并在每个网格内提取 Harris 最强角点，
    保证特征点在图像空间上的均匀覆盖。
    """

    @staticmethod
    def calculate_optimal_grid_size(width, height,
                                    target_grid_size=100,
                                    min_grids=3,
                                    max_grids=8):
        """
        根据视频分辨率动态计算最优网格大小

        参数:
            width:            视频宽度（像素）
            height:           视频高度（像素）
            target_grid_size: 目标网格尺寸（像素），每个网格的理想边长
            min_grids:        最小网格数（每个维度）
            max_grids:        最大网格数（每个维度）

        返回:
            grid_size: (rows, cols) 网格行列数
            grid_w:    每个网格的宽度（像素）
            grid_h:    每个网格的高度（像素）
        """
        cols = max(min_grids, min(max_grids, width // target_grid_size))
        rows = max(min_grids, min(max_grids, height // target_grid_size))

        grid_w = width // cols
        grid_h = height // rows

        return (rows, cols), grid_w, grid_h

    @staticmethod
    def extract_grid_points(gray_image, grid_size, grid_w, grid_h):
        """
        在图像上均匀分布网格，提取每个网格内的最佳 Harris 角点

        在每个网格子区域内计算 Harris 角点响应值，
        选取响应最强的像素点作为该网格的代表特征点。

        参数:
            gray_image: 8 位灰度图像
            grid_size:  网格大小 (rows, cols)
            grid_w:     每个网格的宽度（像素）
            grid_h:     每个网格的高度（像素）

        返回:
            grid_points: 网格特征点数组，shape=(N, 2)，dtype=float32
        """
        h, w = gray_image.shape
        grid_points = []

        for y in range(grid_size[0]):
            for x in range(grid_size[1]):
                grid_y = y * grid_h
                grid_x = x * grid_w

                grid_y_end = min(grid_y + grid_h, h)
                grid_x_end = min(grid_x + grid_w, w)

                grid_roi = gray_image[grid_y:grid_y_end, grid_x:grid_x_end]

                if grid_roi.size == 0:
                    continue

                # Harris 角点检测，选取响应最强点
                harris_response = cv2.cornerHarris(grid_roi, 2, 3, 0.04)
                _, _, _, max_loc = cv2.minMaxLoc(harris_response)

                # 转换为全局坐标
                global_x = grid_x + max_loc[0]
                global_y = grid_y + max_loc[1]

                grid_points.append((global_x, global_y))

        return np.array(grid_points, dtype=np.float32)


# ============================================================================
# 边界修复相关函数
# ============================================================================

def build_transformation_matrix(transform):
    """
    根据 (dx, dy, da) 构造 2x3 仿射变换矩阵

    参数:
        transform: [dx, dy, da]，平移量（像素）和旋转角（弧度）

    返回:
        transform_matrix: 2x3 仿射变换矩阵（numpy array）
    """
    transform_matrix = np.zeros((2, 3))
    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]
    return transform_matrix


def extreme_corners(frame, transforms):
    """
    遍历所有帧的平滑变换，计算图像四角在四个方向上的最大偏移极值

    参数:
        frame:      参考帧（仅用于获取图像尺寸）
        transforms: 平滑变换序列，shape=(N, 3)

    返回:
        字典 {'min_x', 'min_y', 'max_x', 'max_y'}，表示四个方向的极值偏移量
    """
    h, w = frame.shape[:2]

    frame_corners = np.array(
        [[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]], dtype='float32'
    )
    frame_corners = np.array([frame_corners])

    min_x = min_y = max_x = max_y = 0

    for i in range(transforms.shape[0]):
        transform_mat = build_transformation_matrix(transforms[i, :])
        transformed_corners = cv2.transform(frame_corners, transform_mat)

        delta_corners = transformed_corners - frame_corners
        delta_x = delta_corners[0][:, 0].tolist()
        delta_y = delta_corners[0][:, 1].tolist()

        min_x = min([min_x] + delta_x)
        min_y = min([min_y] + delta_y)
        max_x = max([max_x] + delta_x)
        max_y = max([max_y] + delta_y)

    return {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}


def min_auto_border_size(extreme_frame_corners):
    """
    根据四方向极值偏移量计算最小安全边距

    参数:
        extreme_frame_corners: 极值偏移字典 {'min_x', 'min_y', 'max_x', 'max_y'}

    返回:
        border_size: 最小安全边距（int，向上取整）
    """
    abs_vals = [abs(v) for v in extreme_frame_corners.values()]
    return math.ceil(max(abs_vals))


def fix_border(frame, extreme_frame_corners, border_size):
    """
    通过全局统一缩放修复稳像变换引入的边界缺陷

    以图像中心为基准进行放大缩放，使变换偏移露出的边界区域
    被有效内容填充，在保持输出分辨率不变的前提下消除黑边。

    参数:
        frame:                当前视频帧
        extreme_frame_corners: 极值偏移字典
        border_size:          最小安全边距

    返回:
        scaled_frame: 边界修复后的帧
    """
    if border_size == 0:
        return frame

    frame_h, frame_w = frame.shape[:2]

    # 分别计算水平和垂直方向所需缩放比例，取较大值
    scale_w = frame_w / (
        frame_w - abs(extreme_frame_corners['min_x']) - abs(extreme_frame_corners['max_x'])
    )
    scale_h = frame_h / (
        frame_h - abs(extreme_frame_corners['min_y']) - abs(extreme_frame_corners['max_y'])
    )
    scale = max(scale_w, scale_h)

    # 以图像中心为基准执行缩放变换
    center = (frame_w / 2, frame_h / 2)
    M = cv2.getRotationMatrix2D(center, 0, scale)
    scaled_frame = cv2.warpAffine(
        frame, M, (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return scaled_frame


# ============================================================================
# IRV 视频稳像主类
# ============================================================================

class IRVVideoStabilizer:
    """
    IRV 红外视频稳像处理器

    支持动态网格 Harris 角点提取、全局 Shi-Tomasi 优质角点补充、
    多尺度金字塔光流跟踪、RANSAC 鲁棒运动估计、
    移动平均轨迹平滑以及全局统一缩放边界修复。

    稳像变换在原始 16 位域执行，最终输出为 8 位灰度视频，
    以保留完整的红外辐射温度信息。
    """

    def __init__(self,
                 smoothing_radius=50,
                 target_grid_size=100,
                 min_grids=3,
                 max_grids=8,
                 max_corners=200,
                 quality_level=0.01,
                 min_distance=30):
        """
        初始化 IRV 视频稳像器

        参数:
            smoothing_radius:  轨迹平滑半径（帧数），默认 50
            target_grid_size:  目标网格尺寸（像素），默认 100
            min_grids:         最小网格数（每个维度），默认 3
            max_grids:         最大网格数（每个维度），默认 8
            max_corners:       全局特征点最大数量，默认 200
            quality_level:     Shi-Tomasi 特征点质量水平，默认 0.01
            min_distance:      特征点最小间距（像素），默认 30
        """
        self.smoothing_radius = smoothing_radius
        self.target_grid_size = target_grid_size
        self.min_grids = min_grids
        self.max_grids = max_grids
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

        self.smoother = TrajectorySmoother(smoothing_radius)
        self.grid_calculator = GridCalculator()
        self.loader = keii_data_load()

    def stabilize(self, video_path, output_path):
        """
        执行 IRV 视频稳像处理的完整流程

        流程：读取视频信息 → 计算帧间变换 → 平滑轨迹 → 生成稳定视频

        参数:
            video_path:  输入 IRV 视频文件路径
            output_path: 输出 MP4 视频文件路径
        """
        # ---- 读取视频基本信息 ----
        print("正在读取视频信息...")
        with open(video_path, "rb") as f:
            width, height = self.loader.Read_IRVFrame_wh(f)
        with open(video_path, "rb") as f:
            n_frames = self.loader.Read_TotalFrame_IRV(f)

        if width == 0 or height == 0 or n_frames == 0:
            raise ValueError("无法读取视频信息，请检查文件路径或格式。")

        fps = 25

        # ---- 动态计算最优网格大小 ----
        grid_size, grid_w, grid_h = self.grid_calculator.calculate_optimal_grid_size(
            width, height,
            target_grid_size=self.target_grid_size,
            min_grids=self.min_grids,
            max_grids=self.max_grids
        )

        print(f"视频信息: {width}x{height}, fps={fps}, 总帧数={n_frames}")
        print(f"网格配置: {grid_size[0]} 行 x {grid_size[1]} 列 "
              f"(每格 {grid_w}x{grid_h} 像素)")

        # ---- 读取第一帧 ----
        with open(video_path, "rb") as f:
            prev_u16 = self.loader.Open_Frame_IRV(f, 0, width, height)

        if prev_u16 is None:
            raise ValueError("无法读取视频第一帧。")

        prev_gray = u16_to_gray_hist_fuse(prev_u16)

        # ---- 步骤 1：计算帧间变换 ----
        print("\n步骤 1/3: 计算帧间变换...")
        transforms = self._compute_transforms(
            video_path, prev_gray, n_frames, width, height,
            grid_size, grid_w, grid_h
        )

        # ---- 步骤 2：平滑轨迹 ----
        print("\n步骤 2/3: 平滑运动轨迹...")
        transforms_smooth = self._smooth_transforms(transforms)

        # ---- 步骤 3：生成稳定视频 ----
        print("\n步骤 3/3: 生成稳定视频...")
        self._render_stabilized_video(
            video_path, output_path, transforms_smooth,
            prev_u16, n_frames, width, height, fps
        )

    # --------------------------------------------------------------------------
    # 私有方法：步骤 1 —— 计算帧间变换
    # --------------------------------------------------------------------------

    def _compute_transforms(self, video_path, prev_gray, n_frames,
                            width, height, grid_size, grid_w, grid_h):
        """
        逐帧计算帧间运动参数 (dx, dy, da)

        特征点提取策略：动态网格 Harris 角点 + 全局 Shi-Tomasi 优质角点
        跟踪算法：多尺度金字塔 LK 光流
        运动估计：estimateAffinePartial2D（内置 RANSAC）

        参数:
            video_path: IRV 视频路径
            prev_gray:  第一帧 8 位灰度图
            n_frames:   总帧数
            width:      视频宽度
            height:     视频高度
            grid_size:  网格行列数 (rows, cols)
            grid_w:     网格宽度
            grid_h:     网格高度

        返回:
            transforms: 帧间运动增量序列，shape=(n_frames-1, 3)
        """
        transforms = np.zeros((n_frames - 1, 3), np.float32)

        for i in range(n_frames - 2):

            # 提取动态网格 Harris 特征点
            grid_points = self.grid_calculator.extract_grid_points(
                prev_gray, grid_size, grid_w, grid_h
            )

            # 提取全局 Shi-Tomasi 优质特征点
            good_points = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=3
            )

            # 读取当前帧并转换为 8 位灰度
            with open(video_path, "rb") as f:
                curr_u16 = self.loader.Open_Frame_IRV(f, i + 1, width, height)

            if curr_u16 is None:
                print(f"  警告: 无法读取帧 {i + 1}")
                break

            curr_gray = u16_to_gray_hist_fuse(curr_u16)

            # 网格点光流跟踪
            grid_prev_pts, grid_curr_pts = self._track_points(
                prev_gray, curr_gray,
                grid_points.reshape(-1, 1, 2) if len(grid_points) > 0
                else np.array([], dtype=np.float32).reshape(0, 1, 2)
            )

            # 全局点光流跟踪
            if good_points is not None and len(good_points) > 0:
                good_prev_pts, good_curr_pts = self._track_points(
                    prev_gray, curr_gray,
                    good_points.reshape(-1, 1, 2)
                )
            else:
                good_prev_pts = np.array([], dtype=np.float32).reshape(0, 1, 2)
                good_curr_pts = np.array([], dtype=np.float32).reshape(0, 1, 2)

            # 合并两类特征点
            prev_pts, curr_pts = self._merge_points(
                grid_prev_pts, grid_curr_pts,
                good_prev_pts, good_curr_pts
            )

            if prev_pts is None:
                transforms[i] = [0, 0, 0]
                prev_gray = curr_gray
                print(f"  警告: 帧 {i} 未跟踪到任何特征点")
                continue

            # RANSAC 鲁棒仿射变换估计
            dx, dy, da = self._estimate_motion(prev_pts, curr_pts)
            transforms[i] = [dx, dy, da]
            prev_gray = curr_gray

            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{n_frames - 2} 帧")

        return transforms

    def _track_points(self, prev_gray, curr_gray, prev_pts):
        """
        使用多尺度金字塔 LK 光流跟踪特征点，
        依据 status 标志过滤跟踪失败的点。

        参数:
            prev_gray: 参考帧 8 位灰度图
            curr_gray: 当前帧 8 位灰度图
            prev_pts:  参考帧特征点，shape=(N, 1, 2)

        返回:
            prev_pts_ok: 跟踪成功的参考帧点，shape=(M, 1, 2)
            curr_pts_ok: 对应的当前帧点，shape=(M, 1, 2)
        """
        empty = np.array([], dtype=np.float32).reshape(0, 1, 2)

        if len(prev_pts) == 0:
            return empty, empty

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None
        )
        idx = np.where(status.flatten() == 1)[0]

        return prev_pts[idx], curr_pts[idx]

    @staticmethod
    def _merge_points(grid_prev, grid_curr, good_prev, good_curr):
        """
        合并网格特征点与全局优质特征点

        参数:
            grid_prev: 网格点参考帧坐标
            grid_curr: 网格点当前帧坐标
            good_prev: 全局点参考帧坐标
            good_curr: 全局点当前帧坐标

        返回:
            prev_pts: 合并后的参考帧点（若均为空则返回 None）
            curr_pts: 合并后的当前帧点（若均为空则返回 None）
        """
        has_grid = len(grid_prev) > 0
        has_good = len(good_prev) > 0

        if has_grid and has_good:
            return (np.concatenate([grid_prev, good_prev], axis=0),
                    np.concatenate([grid_curr, good_curr], axis=0))
        elif has_grid:
            return grid_prev, grid_curr
        elif has_good:
            return good_prev, good_curr
        else:
            return None, None

    @staticmethod
    def _estimate_motion(prev_pts, curr_pts):
        """
        使用 estimateAffinePartial2D（内置 RANSAC）估计帧间相似变换，
        提取平移量 (dx, dy) 与旋转角 da。

        参数:
            prev_pts: 参考帧匹配点，shape=(N, 1, 2)
            curr_pts: 当前帧匹配点，shape=(N, 1, 2)

        返回:
            dx: 水平平移量（像素）
            dy: 垂直平移量（像素）
            da: 旋转角（弧度）
        """
        if prev_pts.shape[0] >= 4:
            m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        else:
            m = None

        if m is None:
            m = np.eye(2, 3, dtype=np.float32)

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        return dx, dy, da

    # --------------------------------------------------------------------------
    # 私有方法：步骤 2 —— 平滑轨迹
    # --------------------------------------------------------------------------

    def _smooth_transforms(self, transforms):
        """
        对帧间运动增量序列进行轨迹平滑处理

        流程：增量累积为绝对轨迹 → 双次移动平均平滑 → 差值补偿还原为增量

        参数:
            transforms: 原始帧间运动增量序列，shape=(N, 3)

        返回:
            transforms_smooth: 平滑后的帧间变换序列，shape=(N, 3)
        """
        # 累积为绝对轨迹
        trajectory = np.cumsum(transforms, axis=0)

        # 双次移动平均平滑
        smoothed_trajectory = self.smoother.smooth(trajectory)

        # 计算补偿差值并叠加到原始增量
        difference = smoothed_trajectory - trajectory
        transforms_smooth = transforms + difference

        return transforms_smooth

    # --------------------------------------------------------------------------
    # 私有方法：步骤 3 —— 生成稳定视频
    # --------------------------------------------------------------------------

    def _render_stabilized_video(self, video_path, output_path, transforms_smooth,
                                 first_frame_u16, n_frames, width, height, fps):
        """
        逐帧读取 16 位原始帧，施加平滑变换，执行边界修复，
        转换为 8 位灰度后写入输出视频。

        参数:
            video_path:       输入 IRV 视频路径
            output_path:      输出 MP4 视频路径
            transforms_smooth: 平滑变换序列，shape=(N, 3)
            first_frame_u16:  第一帧 16 位原始数据（用于极值边界计算）
            n_frames:         总帧数
            width:            视频宽度
            height:           视频高度
            fps:              输出帧率
        """
        # 计算全局极值边界与安全边距
        extreme = extreme_corners(first_frame_u16, transforms_smooth)
        border_size = min_auto_border_size(extreme)

        # 计算全局统一缩放比例（用于日志显示）
        scale_w = width / (
            width - abs(extreme['min_x']) - abs(extreme['max_x'])
        )
        scale_h = height / (
            height - abs(extreme['min_y']) - abs(extreme['max_y'])
        )
        scale = max(scale_w, scale_h)

        print(f"  边界修复: 安全边距={border_size}px, 缩放比例={scale:.4f}")

        # 创建视频写入对象（单通道灰度输出）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

        for i in range(n_frames - 2):
            # 读取 16 位原始帧
            with open(video_path, "rb") as f:
                frame_u16 = self.loader.Open_Frame_IRV(f, i, width, height)

            if frame_u16 is None:
                print(f"  警告: 无法读取帧 {i}")
                break

            # 构造平滑仿射变换矩阵
            dx, dy, da = transforms_smooth[i]
            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da);  m[0, 1] = -np.sin(da);  m[0, 2] = dx
            m[1, 0] = np.sin(da);  m[1, 1] =  np.cos(da);  m[1, 2] = dy

            # 在 16 位域执行稳像变换（保留辐射温度信息）
            frame_u16_stabilized = cv2.warpAffine(
                frame_u16, m, (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

            # 全局统一缩放修复边界
            frame_u16_stabilized = fix_border(frame_u16_stabilized, extreme, border_size)

            # 转换为 8 位灰度并写入输出视频
            frame_gray = u16_to_gray_hist_fuse(frame_u16_stabilized)
            out.write(frame_gray)

            if (i + 1) % 50 == 0:
                print(f"  已输出 {i + 1}/{n_frames - 2} 帧")

        out.release()
        print(f"\n处理完成！")
        print(f"输出视频: {output_path}")
        print(f"有效内容区域: {width}x{height} (缩放比例: {scale:.4f})")


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="IRV 红外视频稳像处理工具（动态网格版本）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input',  type=str,   default=r'50号阀定位销漏气.IRV', help='输入 IRV 视频路径')
    parser.add_argument('--output', type=str,   default='./Basic_IRV_AutoCrop1.mp4', help='输出视频路径')
    parser.add_argument('--smoothing',        type=int,   default=50,   help='平滑半径（帧数）')
    parser.add_argument('--target_grid_size', type=int,   default=100,  help='目标网格尺寸（像素）')
    parser.add_argument('--min_grids',        type=int,   default=3,    help='最小网格数（每个维度）')
    parser.add_argument('--max_grids',        type=int,   default=8,    help='最大网格数（每个维度）')
    parser.add_argument('--max_corners',      type=int,   default=200,  help='全局特征点最大数量')
    parser.add_argument('--quality_level',    type=float, default=0.01, help='特征点质量水平')
    parser.add_argument('--min_distance',     type=int,   default=30,   help='特征点最小间距（像素）')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 70)
    print("IRV 红外视频稳像处理（动态网格版本）")
    print("=" * 70)
    print(f"输入视频:   {args.input}")
    print(f"输出视频:   {args.output}")
    print(f"平滑半径:   {args.smoothing}")
    print(f"网格配置:   目标尺寸={args.target_grid_size}px, "
          f"范围=[{args.min_grids}, {args.max_grids}]")
    print(f"特征点配置: 最大数量={args.max_corners}, "
          f"质量={args.quality_level}, 最小间距={args.min_distance}px")
    print("=" * 70)

    stabilizer = IRVVideoStabilizer(
        smoothing_radius=args.smoothing,
        target_grid_size=args.target_grid_size,
        min_grids=args.min_grids,
        max_grids=args.max_grids,
        max_corners=args.max_corners,
        quality_level=args.quality_level,
        min_distance=args.min_distance
    )

    try:
        stabilizer.stabilize(args.input, args.output)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())