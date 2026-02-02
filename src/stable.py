"""
视频稳定处理模块
实现基于光流和仿射变换的视频稳定算法
支持动态网格划分和自动参数优化
"""

import argparse
import math
import numpy as np
import cv2


# ============================================================================
# 轨迹平滑类
# ============================================================================

class TrajectorySmoother:
    """轨迹平滑器，使用移动平均滤波平滑相机运动轨迹"""

    def __init__(self, smoothing_radius=50):
        """
        初始化平滑器

        参数:
            smoothing_radius: 平滑半径，值越大平滑效果越强
        """
        self.smoothing_radius = smoothing_radius

    def moving_average(self, curve):
        """
        对一维曲线进行移动平均滤波

        参数:
            curve: 一维数组，待平滑的曲线

        返回:
            平滑后的曲线
        """
        radius = self.smoothing_radius
        window_size = 2 * radius + 1

        # 创建均值滤波核
        f = np.ones(window_size) / window_size

        # 边缘填充，避免边界效应
        curve_pad = np.pad(curve, (radius, radius), 'edge')

        # 卷积操作实现移动平均
        curve_smoothed = np.convolve(curve_pad, f, mode='same')

        # 去除填充部分
        curve_smoothed = curve_smoothed[radius:-radius]

        return curve_smoothed

    def smooth(self, trajectory):
        """
        平滑整个轨迹（包含 x, y, angle 三个维度）
        支持多次平滑以获得更好的效果

        参数:
            trajectory: shape=(N, 3) 的轨迹数组，每行为 (dx, dy, da)

        返回:
            平滑后的轨迹，shape=(N, 3)
        """
        smoothed_trajectory = np.copy(trajectory)

        # 对每个维度分别进行平滑（两次平滑）
        for i in range(3):
            smoothed_trajectory[:, i] = self.moving_average(trajectory[:, i])
            smoothed_trajectory[:, i] = self.moving_average(smoothed_trajectory[:, i])

        return smoothed_trajectory


# ============================================================================
# 动态网格划分
# ============================================================================

class GridCalculator:
    """动态网格计算器，根据视频分辨率自动计算最优网格大小"""

    @staticmethod
    def calculate_optimal_grid_size(width, height,
                                    target_grid_size=100,
                                    min_grids=3,
                                    max_grids=8):
        """
        根据视频分辨率动态计算最优网格大小

        参数:
            width: 视频宽度
            height: 视频高度
            target_grid_size: 目标网格尺寸（像素），每个网格的理想边长
            min_grids: 最小网格数（每个维度）
            max_grids: 最大网格数（每个维度）

        返回:
            grid_size: (rows, cols) 网格大小
            grid_w: 每个网格的宽度
            grid_h: 每个网格的高度
        """
        # 计算列数（宽度方向）
        cols = max(min_grids, min(max_grids, width // target_grid_size))

        # 计算行数（高度方向）
        rows = max(min_grids, min(max_grids, height // target_grid_size))

        # 计算每个网格的实际尺寸
        grid_w = width // cols
        grid_h = height // rows

        return (rows, cols), grid_w, grid_h

    @staticmethod
    def extract_grid_points(gray_image, grid_size, grid_w, grid_h):
        """
        在图像上均匀分布网格，提取每个网格的最佳特征点

        参数:
            gray_image: 灰度图像
            grid_size: 网格大小 (rows, cols)
            grid_w: 每个网格的宽度
            grid_h: 每个网格的高度

        返回:
            grid_points: 网格特征点数组，shape=(N, 2)
        """
        h, w = gray_image.shape
        grid_points = []

        for y in range(grid_size[0]):
            for x in range(grid_size[1]):
                # 计算网格区域
                grid_y = y * grid_h
                grid_x = x * grid_w

                grid_y_end = min(grid_y + grid_h, h)
                grid_x_end = min(grid_x + grid_w, w)

                grid_roi = gray_image[grid_y:grid_y_end, grid_x:grid_x_end]

                if grid_roi.size == 0:
                    continue

                # Harris 角点检测
                harris_response = cv2.cornerHarris(grid_roi, 2, 3, 0.04)

                # 找到最强角点
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
    根据变换参数构造 2x3 仿射变换矩阵

    参数:
        transform: (dx, dy, da) 元组或数组
            dx: x 方向平移
            dy: y 方向平移
            da: 旋转角度（弧度）

    返回:
        2x3 的仿射变换矩阵
    """
    transform_matrix = np.zeros((2, 3))

    # 旋转部分
    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])

    # 平移部分
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]

    return transform_matrix


def extreme_corners(frame, transforms):
    """
    计算所有帧变换后的边界极值

    参数:
        frame: 视频帧（用于获取尺寸）
        transforms: shape=(N, 3) 的变换数组

    返回:
        字典 {'min_x', 'min_y', 'max_x', 'max_y'}，表示边界的极值
    """
    h, w = frame.shape[:2]

    # 定义帧的四个角点
    frame_corners = np.array([
        [0, 0],
        [0, h - 1],
        [w - 1, 0],
        [w - 1, h - 1]
    ], dtype='float32')
    frame_corners = np.array([frame_corners])

    # 初始化极值
    min_x = min_y = max_x = max_y = 0

    # 遍历所有变换，计算变换后的角点位置
    for i in range(transforms.shape[0]):
        transform = transforms[i, :]
        transform_mat = build_transformation_matrix(transform)

        # 应用变换到角点
        transformed_corners = cv2.transform(frame_corners, transform_mat)

        # 计算角点的位移
        delta_corners = transformed_corners - frame_corners
        delta_y_corners = delta_corners[0][:, 1].tolist()
        delta_x_corners = delta_corners[0][:, 0].tolist()

        # 更新极值
        min_x = min([min_x] + delta_x_corners)
        min_y = min([min_y] + delta_y_corners)
        max_x = max([max_x] + delta_x_corners)
        max_y = max([max_y] + delta_y_corners)

    return {
        'min_x': min_x,
        'min_y': min_y,
        'max_x': max_x,
        'max_y': max_y
    }


def min_auto_border_size(extreme_frame_corners):
    """
    根据边界极值计算最小安全边距

    参数:
        extreme_frame_corners: 边界极值字典

    返回:
        最小安全边距（整数）
    """
    abs_extreme_corners = [abs(x) for x in extreme_frame_corners.values()]
    return math.ceil(max(abs_extreme_corners))


def fix_border(frame, extreme_frame_corners, border_size):
    """
    通过缩放修复变换后的黑边问题

    参数:
        frame: 当前帧
        extreme_frame_corners: 边界极值字典
        border_size: 安全边距

    返回:
        修复后的帧
    """
    if border_size == 0:
        return frame

    frame_h, frame_w = frame.shape[:2]

    # 计算缩放比例，确保变换后的内容完全填充画面
    scale_w = frame_w / (
            frame_w - abs(extreme_frame_corners['min_x']) - abs(extreme_frame_corners['max_x'])
    )
    scale_h = frame_h / (
            frame_h - abs(extreme_frame_corners['min_y']) - abs(extreme_frame_corners['max_y'])
    )
    scale = max(scale_w, scale_h)

    # 以中心点进行缩放
    center = (frame_w / 2, frame_h / 2)
    M = cv2.getRotationMatrix2D(center, 0, scale)

    # 应用仿射变换
    scaled_frame = cv2.warpAffine(
        frame, M, (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return scaled_frame


# ============================================================================
# 视频稳定主类
# ============================================================================

class VideoStabilizer:
    """视频稳定处理器（支持动态网格划分）"""

    def __init__(self,
                 smoothing_radius=50,
                 target_grid_size=100,
                 min_grids=3,
                 max_grids=8,
                 max_corners=200,
                 quality_level=0.01,
                 min_distance=30):
        """
        初始化视频稳定器

        参数:
            smoothing_radius: 轨迹平滑半径
            target_grid_size: 目标网格尺寸（像素）
            min_grids: 最小网格数
            max_grids: 最大网格数
            max_corners: 全局特征点最大数量
            quality_level: 特征点质量水平
            min_distance: 特征点最小距离
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

    def stabilize(self, input_path, output_path):
        """
        执行视频稳定处理

        参数:
            input_path: 输入视频路径
            output_path: 输出视频路径
        """
        # 打开视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")

        # 获取视频属性
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 动态计算最优网格大小
        grid_size, grid_w, grid_h = self.grid_calculator.calculate_optimal_grid_size(
            w, h,
            target_grid_size=self.target_grid_size,
            min_grids=self.min_grids,
            max_grids=self.max_grids
        )

        print(f"视频信息: {w}x{h}, {fps}fps, {n_frames}帧")
        print(f"网格配置: {grid_size[0]}行 x {grid_size[1]}列 "
              f"(每格 {grid_w}x{grid_h} 像素)")

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # 读取第一帧
        success, prev = cap.read()
        if not success:
            raise ValueError("无法读取视频第一帧")

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        # 第一步：计算帧间变换
        print("\n步骤 1/3: 计算帧间变换...")
        trajectories = self._compute_trajectories(
            cap, prev_gray, n_frames, w, h, grid_size, grid_w, grid_h
        )

        # 第二步：平滑轨迹
        print("\n步骤 2/3: 平滑运动轨迹...")
        smoothed_trajectories = self._smooth_trajectories(trajectories)

        # 第三步：应用变换并输出
        print("\n步骤 3/3: 应用稳定变换...")
        self._apply_stabilization(
            cap, out, smoothed_trajectories, prev_gray, n_frames, w, h
        )

        # 清理资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"\n✅ 视频稳定完成！输出: {output_path}")

    def _compute_trajectories(self, cap, prev_gray, n_frames, w, h,
                              grid_size, grid_w, grid_h):
        """计算帧间变换轨迹（使用动态网格）"""
        trajectories = np.zeros((n_frames - 1, 3), np.float32)

        for i in range(n_frames - 2):
            # 提取网格特征点
            grid_points = self.grid_calculator.extract_grid_points(
                prev_gray, grid_size, grid_w, grid_h
            )

            # 提取全局特征点
            good_points = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=3
            )

            # 读取下一帧
            success, curr = cap.read()
            if not success:
                break

            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            # 网格点光流跟踪
            if len(grid_points) > 0:
                grid_prev_pts = grid_points.reshape(-1, 1, 2)
                grid_curr_pts, status_grid, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, grid_prev_pts, None
                )
                idx_grid = np.where(status_grid.flatten() == 1)[0]
                grid_prev_pts = grid_prev_pts[idx_grid]
                grid_curr_pts = grid_curr_pts[idx_grid]
            else:
                grid_prev_pts = np.array([], dtype=np.float32).reshape(0, 1, 2)
                grid_curr_pts = np.array([], dtype=np.float32).reshape(0, 1, 2)

            # 全局点光流跟踪
            if good_points is not None and len(good_points) > 0:
                good_prev_pts = good_points.reshape(-1, 1, 2)
                good_curr_pts, status_good, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, good_prev_pts, None
                )
                idx_good = np.where(status_good.flatten() == 1)[0]
                good_prev_pts = good_prev_pts[idx_good]
                good_curr_pts = good_curr_pts[idx_good]
            else:
                good_prev_pts = np.array([], dtype=np.float32).reshape(0, 1, 2)
                good_curr_pts = np.array([], dtype=np.float32).reshape(0, 1, 2)

            # 合并特征点
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
                trajectories[i] = [0, 0, 0]
                prev_gray = curr_gray
                print(f"  警告: 帧 {i} 未跟踪到任何特征点")
                continue

            # 估计仿射变换
            if prev_pts.shape[0] >= 4:
                M, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
            else:
                M = None

            if M is None:
                M = np.eye(2, 3, dtype=np.float32)

            # 提取变换参数
            dx = M[0, 2]
            dy = M[1, 2]
            da = np.arctan2(M[1, 0], M[0, 0])

            trajectories[i] = [dx, dy, da]

            prev_gray = curr_gray

            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{n_frames - 2} 帧")

        return trajectories

    def _smooth_trajectories(self, trajectories):
        """平滑运动轨迹"""
        # 计算累积轨迹
        trajectory = np.cumsum(trajectories, axis=0)

        # 平滑轨迹
        smoothed_trajectory = self.smoother.smooth(trajectory)

        # 计算修正量
        difference = smoothed_trajectory - trajectory

        # 更新变换
        smoothed_trajectories = trajectories + difference

        return smoothed_trajectories

    def _apply_stabilization(self, cap, out, trajectories, prev_gray, n_frames, w, h):
        """应用稳定变换并输出视频"""
        # 计算边界修复参数
        extreme_frame_corners = extreme_corners(prev_gray, trajectories)
        border_size = min_auto_border_size(extreme_frame_corners)

        # 计算缩放比例（用于信息显示）
        scale_w = w / (w - abs(extreme_frame_corners['min_x']) - abs(extreme_frame_corners['max_x']))
        scale_h = h / (h - abs(extreme_frame_corners['min_y']) - abs(extreme_frame_corners['max_y']))
        scale = max(scale_w, scale_h)

        print(f"  边界修复: 安全边距={border_size}px, 缩放比例={scale:.4f}")

        # 重置视频读取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for i in range(n_frames - 2):
            success, frame = cap.read()
            if not success:
                break

            # 获取变换参数
            dx, dy, da = trajectories[i]

            # 构造仿射变换矩阵
            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy

            # 应用变换
            frame_stabilized = cv2.warpAffine(
                frame, m, (w, h),
                borderMode=cv2.BORDER_REPLICATE
            )

            # 修复边界
            frame_fixed = fix_border(
                frame_stabilized, extreme_frame_corners, border_size
            )

            # 写入输出视频
            out.write(frame_fixed)

            if (i + 1) % 50 == 0:
                print(f"  已输出 {i + 1}/{n_frames - 2} 帧")


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="视频稳定处理工具（支持动态网格划分）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        default='../24.mp4',
        help='输入视频路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../result24.mp4',
        help='输出视频路径'
    )
    parser.add_argument(
        '--smoothing',
        type=int,
        default=50,
        help='平滑半径（值越大平滑效果越强）'
    )
    parser.add_argument(
        '--target_grid_size',
        type=int,
        default=100,
        help='目标网格尺寸（像素），每个网格的理想边长'
    )
    parser.add_argument(
        '--min_grids',
        type=int,
        default=3,
        help='最小网格数（每个维度）'
    )
    parser.add_argument(
        '--max_grids',
        type=int,
        default=8,
        help='最大网格数（每个维度）'
    )
    parser.add_argument(
        '--max_corners',
        type=int,
        default=200,
        help='全局特征点最大数量'
    )
    parser.add_argument(
        '--quality_level',
        type=float,
        default=0.01,
        help='特征点质量水平'
    )
    parser.add_argument(
        '--min_distance',
        type=int,
        default=30,
        help='特征点最小距离'
    )

    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    print("=" * 70)
    print("视频稳定处理（动态网格版本）")
    print("=" * 70)
    print(f"输入视频: {args.input}")
    print(f"输出视频: {args.output}")
    print(f"平滑半径: {args.smoothing}")
    print(f"网格配置: 目标尺寸={args.target_grid_size}px, "
          f"范围=[{args.min_grids}, {args.max_grids}]")
    print(f"特征点配置: 最大数量={args.max_corners}, "
          f"质量={args.quality_level}, 最小距离={args.min_distance}")
    print("=" * 70)

    # 创建稳定器并处理
    stabilizer = VideoStabilizer(
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
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
