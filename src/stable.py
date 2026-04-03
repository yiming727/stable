"""
视频稳定处理模块
实现基于动态网格 Harris + 全局 Shi-Tomasi 特征点提取
+ 金字塔LK光流 + FB-check + RANSAC + ECC兜底的视频稳定算法
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
        self.smoothing_radius = smoothing_radius

    def moving_average(self, curve):
        radius = self.smoothing_radius
        window_size = 2 * radius + 1
        f = np.ones(window_size) / window_size
        curve_pad = np.pad(curve, (radius, radius), 'edge')
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        curve_smoothed = curve_smoothed[radius:-radius]
        return curve_smoothed

    def smooth(self, trajectory):
        """
        平滑整个轨迹（x, y, angle 三个维度），每维度执行两次移动平均

        参数:
            trajectory: shape=(N, 3)，每行为 (dx, dy, da)
        返回:
            平滑后的轨迹，shape=(N, 3)
        """
        smoothed_trajectory = np.copy(trajectory)
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
        cols = max(min_grids, min(max_grids, width  // target_grid_size))
        rows = max(min_grids, min(max_grids, height // target_grid_size))

        grid_w = width  // cols
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
            grid_points: 网格特征点数组，shape=(N, 1, 2)，dtype=float32
                         若无特征点则返回 None
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

        if not grid_points:
            return None
        return np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)


def detect_features(gray_image, grid_calculator, grid_size, grid_w, grid_h,
                    max_corners=200, quality_level=0.01, min_distance=30):
    """
    融合特征点提取：动态网格 Harris 角点 + 全局 Shi-Tomasi 优质角点

    两路特征点互补：
      - 网格 Harris 角点：保证全图空间均匀覆盖，避免特征点聚集
      - 全局 Shi-Tomasi 角点：补充高质量特征，增强匹配鲁棒性

    参数:
        gray_image:     8 位灰度图像
        grid_calculator: GridCalculator 实例
        grid_size:      网格行列数 (rows, cols)
        grid_w:         每个网格的宽度（像素）
        grid_h:         每个网格的高度（像素）
        max_corners:    全局 Shi-Tomasi 特征点最大数量
        quality_level:  Shi-Tomasi 特征点质量水平
        min_distance:   特征点最小间距（像素）

    返回:
        merged_pts: 合并后的特征点，shape=(N, 1, 2)；若均为空则返回 None
    """
    # ── 动态网格 Harris 角点 ──────────────────────────────────────────────
    grid_pts = grid_calculator.extract_grid_points(
        gray_image, grid_size, grid_w, grid_h
    )

    # ── 全局 Shi-Tomasi 优质角点 ──────────────────────────────────────────
    good_pts = cv2.goodFeaturesToTrack(
        gray_image,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=3
    )
    if good_pts is not None:
        good_pts = good_pts.reshape(-1, 1, 2)

    # ── 合并两路特征点 ────────────────────────────────────────────────────
    has_grid = grid_pts is not None and len(grid_pts) > 0
    has_good = good_pts is not None and len(good_pts) > 0

    if has_grid and has_good:
        return np.concatenate([grid_pts, good_pts], axis=0)
    elif has_grid:
        return grid_pts
    elif has_good:
        return good_pts
    else:
        return None


# ============================================================================
# FB-check 前向—反向重跟踪一致性检验
# ============================================================================

def fb_check(prev_gray, curr_gray, prev_pts, threshold=1.0):
    """
    前向—反向重跟踪一致性检验：
      1. 前向跟踪：prev_pts → curr_pts
      2. 反向跟踪：curr_pts → back_pts
      3. 计算回投误差 e = ||prev_pts - back_pts||₂
      4. 保留 e ≤ threshold 且前后向均跟踪成功的点对

    参数:
        prev_gray:  参考帧灰度图
        curr_gray:  当前帧灰度图
        prev_pts:   参考帧特征点，shape=(N, 1, 2)
        threshold:  回投误差阈值（像素），默认 1.0

    返回:
        filtered_prev: 通过检验的参考帧点，shape=(M, 1, 2)
        filtered_curr: 通过检验的当前帧点，shape=(M, 1, 2)
    """
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    curr_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **lk_params
    )
    back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, curr_pts, None, **lk_params
    )

    status_mask = (status_fwd.flatten() == 1) & (status_bwd.flatten() == 1)
    diff = prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2)
    fb_error = np.linalg.norm(diff, axis=1)
    valid_mask = status_mask & (fb_error <= threshold)

    return prev_pts[valid_mask], curr_pts[valid_mask]


# ============================================================================
# ECC 直接法配准（兜底）
# ============================================================================

def estimate_transform_ecc(prev_gray, curr_gray):
    """
    基于增强相关系数（ECC）的直接法仿射配准，用于特征点内点不足时的兜底估计。
    ECC 通过最大化两帧图像的归一化灰度相关性迭代估计仿射变换参数。

    参数:
        prev_gray: 参考帧灰度图（8位）
        curr_gray: 当前帧灰度图（8位）

    返回:
        M: 2×3 仿射变换矩阵；若 ECC 迭代失败则返回单位矩阵
    """
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        200,
        1e-4
    )
    try:
        _, warp_matrix = cv2.findTransformECC(
            prev_gray, curr_gray,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria
        )
    except cv2.error:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    return warp_matrix


# ============================================================================
# 边界修复相关函数
# ============================================================================

def build_transformation_matrix(transform):
    m = np.zeros((2, 3))
    m[0, 0] = np.cos(transform[2])
    m[0, 1] = -np.sin(transform[2])
    m[1, 0] = np.sin(transform[2])
    m[1, 1] = np.cos(transform[2])
    m[0, 2] = transform[0]
    m[1, 2] = transform[1]
    return m


def extreme_corners(frame, transforms):
    h, w = frame.shape[:2]
    frame_corners = np.array([
        [0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]
    ], dtype='float32')[np.newaxis]

    min_x = min_y = max_x = max_y = 0
    for i in range(transforms.shape[0]):
        transform_mat = build_transformation_matrix(transforms[i])
        transformed_corners = cv2.transform(frame_corners, transform_mat)
        delta = transformed_corners - frame_corners
        dx_list = delta[0][:, 0].tolist()
        dy_list = delta[0][:, 1].tolist()
        min_x = min([min_x] + dx_list)
        min_y = min([min_y] + dy_list)
        max_x = max([max_x] + dx_list)
        max_y = max([max_y] + dy_list)

    return {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}


def min_auto_border_size(extreme_frame_corners):
    return math.ceil(max(abs(v) for v in extreme_frame_corners.values()))


def fix_border(frame, extreme_frame_corners, border_size):
    if border_size == 0:
        return frame
    frame_h, frame_w = frame.shape[:2]
    scale_w = frame_w / (
        frame_w - abs(extreme_frame_corners['min_x']) - abs(extreme_frame_corners['max_x'])
    )
    scale_h = frame_h / (
        frame_h - abs(extreme_frame_corners['min_y']) - abs(extreme_frame_corners['max_y'])
    )
    scale = max(scale_w, scale_h)
    center = (frame_w / 2, frame_h / 2)
    M = cv2.getRotationMatrix2D(center, 0, scale)
    return cv2.warpAffine(
        frame, M, (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


# ============================================================================
# 视频稳定主类
# ============================================================================

class VideoStabilizer:
    """
    视频稳定处理器

    帧间匹配流程：
      动态网格 Harris 角点（均匀空间覆盖）
        + 全局 Shi-Tomasi 优质角点（补充高质量特征）
        → 金字塔 LK 光流跟踪
        → FB-check 一致性检验
        → RANSAC 鲁棒仿射估计
        → 内点数量判断
            ├─ 充足 → 输出 RANSAC 变换矩阵
            └─ 不足 → ECC 直接法兜底
    """

    def __init__(self,
                 smoothing_radius=50,
                 target_grid_size=100,
                 min_grids=3,
                 max_grids=8,
                 max_corners=200,
                 quality_level=0.01,
                 min_distance=30,
                 fb_threshold=1.0,
                 min_inliers=8):
        """
        参数:
            smoothing_radius:  轨迹平滑半径（帧数）
            target_grid_size:  目标网格尺寸（像素），每个网格的理想边长
            min_grids:         最小网格数（每个维度）
            max_grids:         最大网格数（每个维度）
            max_corners:       全局 Shi-Tomasi 特征点最大数量
            quality_level:     Shi-Tomasi 特征点质量水平
            min_distance:      特征点最小间距（像素）
            fb_threshold:      FB-check 回投误差阈值（像素）
            min_inliers:       RANSAC 内点数量下限，低于此值切换 ECC
        """
        self.smoothing_radius = smoothing_radius
        self.target_grid_size = target_grid_size
        self.min_grids        = min_grids
        self.max_grids        = max_grids
        self.max_corners      = max_corners
        self.quality_level    = quality_level
        self.min_distance     = min_distance
        self.fb_threshold     = fb_threshold
        self.min_inliers      = min_inliers

        self.smoother         = TrajectorySmoother(smoothing_radius)
        self.grid_calculator  = GridCalculator()

    def stabilize(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps      = cap.get(cv2.CAP_PROP_FPS)

        # 根据分辨率动态计算网格行列数及每格尺寸
        grid_size, grid_w, grid_h = self.grid_calculator.calculate_optimal_grid_size(
            w, h,
            target_grid_size=self.target_grid_size,
            min_grids=self.min_grids,
            max_grids=self.max_grids
        )

        print(f"视频信息:     {w}x{h}, {fps}fps, {n_frames}帧")
        print(f"网格配置:     {grid_size[0]}行 × {grid_size[1]}列 "
              f"(每格 {grid_w}x{grid_h} 像素)")
        print(f"FB-check:     回投阈值 {self.fb_threshold}px")
        print(f"ECC 切换阈值: 内点 < {self.min_inliers}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        success, prev = cap.read()
        if not success:
            raise ValueError("无法读取视频第一帧")
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        print("\n步骤 1/3: 计算帧间变换...")
        trajectories = self._compute_trajectories(
            cap, prev_gray, n_frames, grid_size, grid_w, grid_h
        )

        print("\n步骤 2/3: 平滑运动轨迹...")
        smoothed_trajectories = self._smooth_trajectories(trajectories)

        print("\n步骤 3/3: 应用稳定变换...")
        self._apply_stabilization(cap, out, smoothed_trajectories, prev_gray, n_frames, w, h)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n✅ 视频稳定完成！输出: {output_path}")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _compute_trajectories(self, cap, prev_gray, n_frames,
                              grid_size, grid_w, grid_h):
        """
        逐帧计算帧间变换，返回 shape=(n_frames-1, 3) 的轨迹数组。

        每帧流程：
          1. 动态网格 Harris 角点 + 全局 Shi-Tomasi 优质角点提取
          2. 金字塔 LK 光流跟踪
          3. FB-check 一致性检验
          4. RANSAC 鲁棒仿射估计，统计内点数 N_inlier
          5. 若 N_inlier ≥ min_inliers → 采用 RANSAC 结果
             否则 → 切换 ECC 直接法
        """
        trajectories = np.zeros((n_frames - 1, 3), np.float32)
        ecc_count = 0

        for i in range(n_frames - 2):
            success, curr = cap.read()
            if not success:
                break

            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            # ── (1) 动态网格 Harris + 全局 Shi-Tomasi 特征点提取 ────────
            prev_pts = detect_features(
                prev_gray,
                grid_calculator=self.grid_calculator,
                grid_size=grid_size,
                grid_w=grid_w,
                grid_h=grid_h,
                max_corners=self.max_corners,
                quality_level=self.quality_level,
                min_distance=self.min_distance
            )

            M = None

            if prev_pts is not None and len(prev_pts) >= 4:
                # ── (2)(3) 金字塔 LK 跟踪 + FB-check ───────────────────
                filtered_prev, filtered_curr = fb_check(
                    prev_gray, curr_gray, prev_pts,
                    threshold=self.fb_threshold
                )

                # ── (4) RANSAC 鲁棒仿射估计 ──────────────────────────────
                if len(filtered_prev) >= 4:
                    M_ransac, inlier_mask = cv2.estimateAffinePartial2D(
                        filtered_prev, filtered_curr,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        confidence=0.99,
                        maxIters=2000
                    )
                    if M_ransac is not None and inlier_mask is not None:
                        n_inlier = int(inlier_mask.sum())
                        # ── (5) 内点充足 → 特征点法路径 ──────────────────
                        if n_inlier >= self.min_inliers:
                            M = M_ransac

            # ── (5) 内点不足或提取失败 → ECC 兜底 ──────────────────────
            if M is None:
                M = estimate_transform_ecc(prev_gray, curr_gray)
                ecc_count += 1

            dx = M[0, 2]
            dy = M[1, 2]
            da = np.arctan2(M[1, 0], M[0, 0])
            trajectories[i] = [dx, dy, da]

            prev_gray = curr_gray

            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{n_frames - 2} 帧 "
                      f"（累计 ECC 兜底: {ecc_count} 帧）")

        print(f"  帧间变换计算完成，ECC 兜底共 {ecc_count} 帧")
        return trajectories

    def _smooth_trajectories(self, trajectories):
        trajectory = np.cumsum(trajectories, axis=0)
        smoothed_trajectory = self.smoother.smooth(trajectory)
        difference = smoothed_trajectory - trajectory
        return trajectories + difference

    def _apply_stabilization(self, cap, out, trajectories, prev_gray, n_frames, w, h):
        extreme_frame_corners = extreme_corners(prev_gray, trajectories)
        border_size = min_auto_border_size(extreme_frame_corners)

        scale_w = w / (w - abs(extreme_frame_corners['min_x']) - abs(extreme_frame_corners['max_x']))
        scale_h = h / (h - abs(extreme_frame_corners['min_y']) - abs(extreme_frame_corners['max_y']))
        scale = max(scale_w, scale_h)
        print(f"  边界修复: 安全边距={border_size}px, 缩放比例={scale:.4f}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for i in range(n_frames - 2):
            success, frame = cap.read()
            if not success:
                break

            dx, dy, da = trajectories[i]
            m = np.array([
                [np.cos(da), -np.sin(da), dx],
                [np.sin(da),  np.cos(da), dy]
            ], dtype=np.float32)

            frame_stabilized = cv2.warpAffine(
                frame, m, (w, h),
                borderMode=cv2.BORDER_REPLICATE
            )
            frame_fixed = fix_border(frame_stabilized, extreme_frame_corners, border_size)
            out.write(frame_fixed)

            if (i + 1) % 50 == 0:
                print(f"  已输出 {i + 1}/{n_frames - 2} 帧")


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="视频稳定处理工具（动态网格 Harris + Shi-Tomasi + FB-check + RANSAC + ECC兜底）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input',            type=str,   default='./1.mp4',
                        help='输入视频路径')
    parser.add_argument('--output',           type=str,   default='./result1.mp4',
                        help='输出视频路径')
    parser.add_argument('--smoothing',        type=int,   default=50,
                        help='平滑半径（帧数）')
    parser.add_argument('--target_grid_size', type=int,   default=100,
                        help='目标网格尺寸（像素），每个网格的理想边长')
    parser.add_argument('--min_grids',        type=int,   default=3,
                        help='最小网格数（每个维度）')
    parser.add_argument('--max_grids',        type=int,   default=8,
                        help='最大网格数（每个维度）')
    parser.add_argument('--max_corners',      type=int,   default=200,
                        help='全局 Shi-Tomasi 特征点最大数量')
    parser.add_argument('--quality_level',    type=float, default=0.01,
                        help='Shi-Tomasi 特征点质量水平')
    parser.add_argument('--min_distance',     type=int,   default=30,
                        help='特征点最小间距（像素）')
    parser.add_argument('--fb_threshold',     type=float, default=1.0,
                        help='FB-check 回投误差阈值（像素）')
    parser.add_argument('--min_inliers',      type=int,   default=8,
                        help='RANSAC 内点下限，低于此值切换 ECC')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("视频稳定处理（动态网格 Harris + Shi-Tomasi + FB-check + RANSAC + ECC兜底）")
    print("=" * 70)
    print(f"输入视频:     {args.input}")
    print(f"输出视频:     {args.output}")
    print(f"平滑半径:     {args.smoothing}")
    print(f"网格配置:     目标尺寸={args.target_grid_size}px, "
          f"范围=[{args.min_grids}, {args.max_grids}]")
    print(f"特征点配置:   最大数量={args.max_corners}, "
          f"质量={args.quality_level}, 最小间距={args.min_distance}px")
    print(f"FB-check:     回投阈值 {args.fb_threshold}px")
    print(f"ECC 切换阈值: 内点 < {args.min_inliers}")
    print("=" * 70)

    stabilizer = VideoStabilizer(
        smoothing_radius=args.smoothing,
        target_grid_size=args.target_grid_size,
        min_grids=args.min_grids,
        max_grids=args.max_grids,
        max_corners=args.max_corners,
        quality_level=args.quality_level,
        min_distance=args.min_distance,
        fb_threshold=args.fb_threshold,
        min_inliers=args.min_inliers
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