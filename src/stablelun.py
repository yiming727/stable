"""
视频稳定处理模块
实现基于分桶特征点提取 + 金字塔LK光流 + FB-check + RANSAC + ECC兜底的视频稳定算法
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
# 空间分桶特征点提取
# ============================================================================

def compute_bucket_grid(width, height, bucket_size=120):
    """
    根据图像分辨率动态计算桶的行列数。
    桶的数量与分辨率成正比：分辨率越高，桶越多，空间覆盖密度保持一致。

    参数:
        width:       图像宽度（像素）
        height:      图像高度（像素）
        bucket_size: 每个桶的目标边长（像素），决定桶的密度

    返回:
        grid_rows: 桶的行数
        grid_cols: 桶的列数
    """
    grid_cols = max(1, round(width  / bucket_size))
    grid_rows = max(1, round(height / bucket_size))
    return grid_rows, grid_cols


def detect_bucketing(img, grid_rows, grid_cols, quality=0.01, min_distance=5):
    """
    空间分桶特征点提取：将图像划分为 grid_rows × grid_cols 个桶，
    在每个桶内独立运行 goodFeaturesToTrack，不限制每桶数量上限，
    仅通过 qualityLevel 控制质量门槛（相对于桶内最强角点）。

    参数:
        img:          8位灰度图像
        grid_rows:    桶的行数（由 compute_bucket_grid 根据分辨率计算）
        grid_cols:    桶的列数（由 compute_bucket_grid 根据分辨率计算）
        quality:      qualityLevel，相对于桶内最强角点的质量阈值
        min_distance: 同一桶内特征点的最小间距（像素）

    返回:
        shape=(N, 1, 2) 的特征点数组，若无特征点则返回 None
    """
    h, w = img.shape
    bucket_h = h // grid_rows
    bucket_w = w // grid_cols
    selected = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            y1 = r * bucket_h
            y2 = (r + 1) * bucket_h if r < grid_rows - 1 else h
            x1 = c * bucket_w
            x2 = (c + 1) * bucket_w if c < grid_cols - 1 else w

            bucket_img = img[y1:y2, x1:x2]
            if bucket_img.size == 0:
                continue

            # 不设 maxCorners 上限（传 0 表示不限制），由 qualityLevel 自然筛选
            corners = cv2.goodFeaturesToTrack(
                bucket_img,
                maxCorners=0,
                qualityLevel=quality,
                minDistance=min_distance
            )

            if corners is not None:
                corners = corners.reshape(-1, 2)
                corners[:, 0] += x1
                corners[:, 1] += y1
                selected.extend(corners.tolist())

    if not selected:
        return None
    return np.array(selected, dtype=np.float32).reshape(-1, 1, 2)


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
      分桶特征点提取（桶数由分辨率自动决定，桶内不限数量）
        → 金字塔 LK 光流跟踪
        → FB-check 一致性检验
        → RANSAC 鲁棒仿射估计
        → 内点数量判断
            ├─ 充足 → 输出 RANSAC 变换矩阵
            └─ 不足 → ECC 直接法兜底
    """

    def __init__(self,
                 smoothing_radius=50,
                 bucket_size=120,
                 quality_level=0.01,
                 min_distance=5,
                 fb_threshold=1.0,
                 min_inliers=8):
        """
        参数:
            smoothing_radius: 轨迹平滑半径
            bucket_size:      每个桶的目标边长（像素），分辨率越高桶越多
            quality_level:    桶内角点质量阈值（相对桶内最强角点，不限数量上限）
            min_distance:     桶内特征点最小间距（像素）
            fb_threshold:     FB-check 回投误差阈值（像素）
            min_inliers:      RANSAC 内点数量下限，低于此值切换 ECC
        """
        self.smoothing_radius = smoothing_radius
        self.bucket_size      = bucket_size
        self.quality_level    = quality_level
        self.min_distance     = min_distance
        self.fb_threshold     = fb_threshold
        self.min_inliers      = min_inliers

        self.smoother = TrajectorySmoother(smoothing_radius)

    def stabilize(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps      = cap.get(cv2.CAP_PROP_FPS)

        # 根据分辨率动态计算桶的行列数
        grid_rows, grid_cols = compute_bucket_grid(w, h, self.bucket_size)

        print(f"视频信息:     {w}x{h}, {fps}fps, {n_frames}帧")
        print(f"分桶配置:     {grid_rows}行 × {grid_cols}列"
              f"（桶目标边长 {self.bucket_size}px，桶内不限数量）")
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
            cap, prev_gray, n_frames, grid_rows, grid_cols
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

    def _compute_trajectories(self, cap, prev_gray, n_frames, grid_rows, grid_cols):
        """
        逐帧计算帧间变换，返回 shape=(n_frames-1, 3) 的轨迹数组。

        每帧流程：
          1. 分桶特征点提取（全图，无掩膜，桶内不限数量）
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

            # ── (1) 分桶特征点提取 ──────────────────────────────────────
            prev_pts = detect_bucketing(
                prev_gray,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                quality=self.quality_level,
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
        description="视频稳定处理工具（分桶特征点 + FB-check + RANSAC + ECC兜底）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input',         type=str,   default='./24.mp4',     help='输入视频路径')
    parser.add_argument('--output',        type=str,   default='./result24.mp4', help='输出视频路径')
    parser.add_argument('--smoothing',     type=int,   default=50,             help='平滑半径')
    parser.add_argument('--bucket_size',   type=int,   default=360,            help='每个桶的目标边长（像素）；分辨率越高桶越多，覆盖密度保持一致')
    parser.add_argument('--quality_level', type=float, default=0.01,            help='桶内角点质量阈值（相对桶内最强角点，不限数量上限）')
    parser.add_argument('--min_distance',  type=int,   default=30,             help='桶内特征点最小间距（像素）')
    parser.add_argument('--fb_threshold',  type=float, default=1.0,            help='FB-check 回投误差阈值（像素）')
    parser.add_argument('--min_inliers',   type=int,   default=10,             help='RANSAC 内点下限，低于此值切换 ECC')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("视频稳定处理（分桶特征点 + FB-check + RANSAC + ECC兜底）")
    print("=" * 70)
    print(f"输入视频:     {args.input}")
    print(f"输出视频:     {args.output}")
    print(f"平滑半径:     {args.smoothing}")
    print(f"桶目标边长:   {args.bucket_size}px（桶数随分辨率自动缩放，桶内不限数量）")
    print(f"质量阈值:     {args.quality_level}")
    print(f"最小间距:     {args.min_distance}px")
    print(f"FB-check:     回投阈值 {args.fb_threshold}px")
    print(f"ECC 切换阈值: 内点 < {args.min_inliers}")
    print("=" * 70)

    stabilizer = VideoStabilizer(
        smoothing_radius=args.smoothing,
        bucket_size=args.bucket_size,
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