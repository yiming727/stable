"""
视频稳定处理模块
实现基于分桶特征点提取 + 金字塔LK光流 + FB-check + RANSAC + ECC兜底的视频稳定算法
裁剪方式：基于 FOV（视场角）+ 高斯平滑，替代原固定边距裁剪
"""

import argparse
import math
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d


# ============================================================================
# 轨迹平滑类（不变）
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
        return curve_smoothed[radius:-radius]

    def smooth(self, trajectory):
        smoothed_trajectory = np.copy(trajectory)
        for i in range(3):
            smoothed_trajectory[:, i] = self.moving_average(trajectory[:, i])
            smoothed_trajectory[:, i] = self.moving_average(smoothed_trajectory[:, i])
        return smoothed_trajectory


# ============================================================================
# 空间分桶特征点提取（不变）
# ============================================================================

def compute_bucket_grid(width, height, bucket_size=120):
    grid_cols = max(1, round(width  / bucket_size))
    grid_rows = max(1, round(height / bucket_size))
    return grid_rows, grid_cols


def detect_bucketing(img, grid_rows, grid_cols, quality=0.01, min_distance=5):
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
            corners = cv2.goodFeaturesToTrack(
                bucket_img, maxCorners=0,
                qualityLevel=quality, minDistance=min_distance
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
# FB-check（不变）
# ============================================================================

def fb_check(prev_gray, curr_gray, prev_pts, threshold=1.0):
    lk_params = dict(
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    curr_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **lk_params)
    back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, curr_pts, None, **lk_params)
    status_mask = (status_fwd.flatten() == 1) & (status_bwd.flatten() == 1)
    diff = prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2)
    fb_error = np.linalg.norm(diff, axis=1)
    valid_mask = status_mask & (fb_error <= threshold)
    return prev_pts[valid_mask], curr_pts[valid_mask]


# ============================================================================
# ECC 兜底（不变）
# ============================================================================

def estimate_transform_ecc(prev_gray, curr_gray):
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-4)
    try:
        _, warp_matrix = cv2.findTransformECC(
            prev_gray, curr_gray, warp_matrix, cv2.MOTION_AFFINE, criteria)
    except cv2.error:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    return warp_matrix


# ============================================================================
# ★ FOV 裁剪模块（替换原 fix_border / extreme_corners）
# ============================================================================

def compute_frame_fov(w, h, dx, dy, da):
    """
    计算单帧稳定变换后的 FOV（视场角比例）。

    论文定义：
      P = 图像中心 (cx, cy)
      对稳定变换取逆，将稳定帧四角映射回原始帧坐标系，
      得到一个（可能旋转的）四边形。
      W = P 到该四边形左/右边的最小距离
      H = P 到该四边形上/下边的最小距离
      FOV = min(W / cx, H / cy)            —— 公式 (4-10)

    参数:
        w, h:  帧宽高（像素）
        dx, dy, da: 当前帧稳定变换参数（平移 + 旋转角）

    返回:
        fov:   标量，范围 (0, 1]，越接近 1 表示黑边越少
        W_px:  裁剪矩形半宽（像素）
        H_px:  裁剪矩形半高（像素）
    """
    cx, cy = w / 2.0, h / 2.0

    # 稳定变换矩阵
    cos_a, sin_a = math.cos(da), math.sin(da)
    M = np.array([
        [cos_a, -sin_a, dx],
        [sin_a,  cos_a, dy]
    ], dtype=np.float64)

    # 逆变换（稳定帧 → 原始帧）
    R_inv = M[:2, :2].T
    t_inv = -R_inv @ M[:2, 2]
    M_inv = np.eye(2, 3)
    M_inv[:2, :2] = R_inv
    M_inv[:2,  2] = t_inv

    # 稳定帧四角 → 原始帧坐标
    corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float64)
    ones    = np.ones((4, 1))
    c_orig  = (M_inv @ np.hstack([corners, ones]).T).T  # (4,2)

    # 计算中心点到四边形四条边的内侧有符号距离
    def signed_dist(px, py, ax, ay, bx, by):
        """点 (px,py) 到有向线段 (a→b) 左侧的距离"""
        ex, ey = bx - ax, by - ay
        length = math.hypot(ex, ey)
        if length < 1e-9:
            return 0.0
        nx, ny = -ey / length, ex / length   # 左法向量
        return (px - ax) * nx + (py - ay) * ny

    # 顺时针四边：top(0→1), right(1→2), bottom(2→3), left(3→0)
    d_top    = signed_dist(cx, cy, c_orig[0,0], c_orig[0,1],
                                   c_orig[1,0], c_orig[1,1])
    d_right  = signed_dist(cx, cy, c_orig[1,0], c_orig[1,1],
                                   c_orig[2,0], c_orig[2,1])
    d_bottom = signed_dist(cx, cy, c_orig[2,0], c_orig[2,1],
                                   c_orig[3,0], c_orig[3,1])
    d_left   = signed_dist(cx, cy, c_orig[3,0], c_orig[3,1],
                                   c_orig[0,0], c_orig[0,1])

    # W = 中心到左/右边的最小距离，H = 中心到上/下边的最小距离
    W_px = min(abs(d_left),  abs(d_right))
    H_px = min(abs(d_top),   abs(d_bottom))
    W_px = max(W_px, 1.0)
    H_px = max(H_px, 1.0)

    fov = min(W_px / cx, H_px / cy)   # 公式 (4-10)
    return fov, W_px, H_px


def smooth_fov(fov_array, sigma=15):
    """
    对逐帧 FOV 序列进行高斯平滑（论文 4.1 节）。

    高斯核公式（论文公式 4-11）：
        G(x) = 1/(2√π σ) · exp(−x²/(2σ²))

    步骤：
      (1) 确定高斯核大小（奇数窗口，由 sigma 决定）
      (2) 生成一维高斯核
      (3) 归一化高斯核（权重之和为 1）
      (4) 对 FOV 序列做加权卷积

    参数:
        fov_array: shape=(N,) 的逐帧 FOV 数组
        sigma:     高斯标准差，控制平滑程度（越大越平滑）

    返回:
        平滑后的 FOV 数组，shape=(N,)，值域保持 (0, 1]
    """
    smoothed = gaussian_filter1d(fov_array.astype(np.float64), sigma=sigma,
                                 mode='nearest')
    return np.clip(smoothed, 1e-3, 1.0)


def crop_and_scale_frame(frame, fov):
    """
    根据平滑后的 FOV 对帧进行裁剪并缩放回原始分辨率。

    裁剪矩形以图像中心 P 为中心，半宽 W = fov * w/2，半高 H = fov * h/2。
    裁剪后双线性插值缩放至原始尺寸，保证输出分辨率不变。

    参数:
        frame: BGR 帧，shape=(h, w, 3)
        fov:   当前帧平滑后的 FOV 值，范围 (0, 1]

    返回:
        裁剪并缩放后的帧，shape=(h, w, 3)
    """
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    W_px = fov * cx   # 裁剪矩形半宽
    H_px = fov * cy   # 裁剪矩形半高

    x1 = int(math.floor(cx - W_px))
    y1 = int(math.floor(cy - H_px))
    x2 = int(math.ceil (cx + W_px))
    y2 = int(math.ceil (cy + H_px))

    # 边界保护
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return frame
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ============================================================================
# 视频稳定主类
# ============================================================================

class VideoStabilizer:
    """
    视频稳定处理器

    帧间匹配流程：
      分桶特征点提取 → 金字塔LK光流 → FB-check → RANSAC → ECC兜底
      → 逐帧计算 FOV → 高斯平滑 FOV → 按平滑 FOV 逐帧裁剪缩放
    """

    def __init__(self,
                 smoothing_radius=50,
                 bucket_size=120,
                 quality_level=0.01,
                 min_distance=5,
                 fb_threshold=1.0,
                 min_inliers=8,
                 fov_sigma=15):
        """
        参数:
            smoothing_radius: 轨迹平滑半径
            bucket_size:      每个桶的目标边长（像素）
            quality_level:    桶内角点质量阈值
            min_distance:     桶内特征点最小间距（像素）
            fb_threshold:     FB-check 回投误差阈值（像素）
            min_inliers:      RANSAC 内点数量下限
            fov_sigma:        FOV 高斯平滑标准差（越大裁剪越平稳）
        """
        self.smoothing_radius = smoothing_radius
        self.bucket_size      = bucket_size
        self.quality_level    = quality_level
        self.min_distance     = min_distance
        self.fb_threshold     = fb_threshold
        self.min_inliers      = min_inliers
        self.fov_sigma        = fov_sigma
        self.smoother         = TrajectorySmoother(smoothing_radius)

    def stabilize(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps      = cap.get(cv2.CAP_PROP_FPS)

        grid_rows, grid_cols = compute_bucket_grid(w, h, self.bucket_size)

        print(f"视频信息:     {w}x{h}, {fps}fps, {n_frames}帧")
        print(f"分桶配置:     {grid_rows}行 × {grid_cols}列"
              f"（桶目标边长 {self.bucket_size}px）")
        print(f"FB-check:     回投阈值 {self.fb_threshold}px")
        print(f"ECC 切换阈值: 内点 < {self.min_inliers}")
        print(f"FOV 高斯平滑: sigma={self.fov_sigma}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        success, prev = cap.read()
        if not success:
            raise ValueError("无法读取视频第一帧")
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        # ── 步骤 1：计算帧间变换 ──────────────────────────────────────
        print("\n步骤 1/3: 计算帧间变换...")
        trajectories = self._compute_trajectories(
            cap, prev_gray, n_frames, grid_rows, grid_cols
        )

        # ── 步骤 2：平滑运动轨迹 ──────────────────────────────────────
        print("\n步骤 2/3: 平滑运动轨迹...")
        smoothed_trajectories = self._smooth_trajectories(trajectories)

        # ── 步骤 3：计算逐帧 FOV → 高斯平滑 FOV ─────────────────────
        print("\n步骤 3/4: 计算并平滑 FOV...")
        fov_array = self._compute_fov_array(smoothed_trajectories, w, h)
        smoothed_fov = smooth_fov(fov_array, sigma=self.fov_sigma)
        print(f"  原始 FOV:  min={fov_array.min():.4f}, "
              f"max={fov_array.max():.4f}, std={fov_array.std():.4f}")
        print(f"  平滑 FOV:  min={smoothed_fov.min():.4f}, "
              f"max={smoothed_fov.max():.4f}, std={smoothed_fov.std():.4f}")

        # ── 步骤 4：应用稳定变换 + FOV 裁剪缩放 ─────────────────────
        print("\n步骤 4/4: 应用稳定变换 + FOV 裁剪缩放...")
        self._apply_stabilization(
            cap, out, smoothed_trajectories, smoothed_fov, n_frames, w, h
        )

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n✅ 视频稳定完成！输出: {output_path}")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _compute_trajectories(self, cap, prev_gray, n_frames, grid_rows, grid_cols):
        trajectories = np.zeros((n_frames - 1, 3), np.float32)
        ecc_count = 0

        for i in range(n_frames - 2):
            success, curr = cap.read()
            if not success:
                break
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            prev_pts = detect_bucketing(
                prev_gray, grid_rows=grid_rows, grid_cols=grid_cols,
                quality=self.quality_level, min_distance=self.min_distance
            )
            M = None

            if prev_pts is not None and len(prev_pts) >= 4:
                filtered_prev, filtered_curr = fb_check(
                    prev_gray, curr_gray, prev_pts, threshold=self.fb_threshold
                )
                if len(filtered_prev) >= 4:
                    M_ransac, inlier_mask = cv2.estimateAffinePartial2D(
                        filtered_prev, filtered_curr,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        confidence=0.99,
                        maxIters=2000
                    )
                    if (M_ransac is not None and inlier_mask is not None
                            and int(inlier_mask.sum()) >= self.min_inliers):
                        M = M_ransac

            if M is None:
                M = estimate_transform_ecc(prev_gray, curr_gray)
                ecc_count += 1

            trajectories[i] = [M[0,2], M[1,2], np.arctan2(M[1,0], M[0,0])]
            prev_gray = curr_gray

            if (i + 1) % 50 == 0:
                print(f"  已处理 {i+1}/{n_frames-2} 帧 "
                      f"（累计 ECC 兜底: {ecc_count} 帧）")

        print(f"  帧间变换计算完成，ECC 兜底共 {ecc_count} 帧")
        return trajectories

    def _smooth_trajectories(self, trajectories):
        trajectory = np.cumsum(trajectories, axis=0)
        smoothed_trajectory = self.smoother.smooth(trajectory)
        difference = smoothed_trajectory - trajectory
        return trajectories + difference

    def _compute_fov_array(self, trajectories, w, h):
        """
        对每一帧的稳定变换参数计算 FOV，返回 shape=(N,) 的 FOV 数组。
        FOV 反映该帧稳定后黑边的严重程度：越小表示黑边越多。
        """
        n = len(trajectories)
        fov_array = np.ones(n, dtype=np.float64)
        for i in range(n):
            dx, dy, da = float(trajectories[i, 0]), \
                         float(trajectories[i, 1]), \
                         float(trajectories[i, 2])
            fov, _, _ = compute_frame_fov(w, h, dx, dy, da)
            fov_array[i] = fov
        return fov_array

    def _apply_stabilization(self, cap, out, trajectories,
                              smoothed_fov, n_frames, w, h):
        """
        逐帧应用稳定变换，并按平滑后的 FOV 裁剪缩放，消除黑边。

        与原方案的区别：
          原方案：全局计算最大黑边 → 统一缩放（所有帧缩放比例相同）
          本方案：逐帧按平滑 FOV 裁剪 → 逐帧缩放（裁剪量随运动连续变化）
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for i in range(n_frames - 2):
            success, frame = cap.read()
            if not success:
                break

            # 1. 应用稳定变换（刚体旋转+平移）
            dx, dy, da = trajectories[i]
            m = np.array([
                [np.cos(da), -np.sin(da), dx],
                [np.sin(da),  np.cos(da), dy]
            ], dtype=np.float32)
            frame_stabilized = cv2.warpAffine(
                frame, m, (w, h),
                borderMode=cv2.BORDER_REPLICATE
            )

            # 2. 按平滑 FOV 裁剪并缩放回原始分辨率
            fov_i = float(smoothed_fov[min(i, len(smoothed_fov) - 1)])
            frame_fixed = crop_and_scale_frame(frame_stabilized, fov_i)

            out.write(frame_fixed)

            if (i + 1) % 50 == 0:
                print(f"  已输出 {i+1}/{n_frames-2} 帧 "
                      f"（当前 FOV={fov_i:.4f}）")


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="视频稳定（分桶特征点 + FB-check + RANSAC + ECC兜底 + FOV高斯裁剪）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input',         type=str,   default='./8.mp4')
    parser.add_argument('--output',        type=str,   default='./result8.mp4')
    parser.add_argument('--smoothing',     type=int,   default=50,   help='轨迹平滑半径')
    parser.add_argument('--bucket_size',   type=int,   default=120,  help='桶目标边长（像素）')
    parser.add_argument('--quality_level', type=float, default=0.01, help='角点质量阈值')
    parser.add_argument('--min_distance',  type=int,   default=30,   help='桶内最小点间距')
    parser.add_argument('--fb_threshold',  type=float, default=1.0,  help='FB-check 阈值')
    parser.add_argument('--min_inliers',   type=int,   default=10,   help='RANSAC 内点下限')
    parser.add_argument('--fov_sigma',     type=float, default=1000.0,
                        help='FOV 高斯平滑标准差（越大裁剪越平稳，但可能损失更多画面）')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("视频稳定（分桶特征点 + FB-check + RANSAC + ECC兜底 + FOV高斯裁剪）")
    print("=" * 70)
    print(f"输入视频:     {args.input}")
    print(f"输出视频:     {args.output}")
    print(f"平滑半径:     {args.smoothing}")
    print(f"桶目标边长:   {args.bucket_size}px")
    print(f"质量阈值:     {args.quality_level}")
    print(f"最小间距:     {args.min_distance}px")
    print(f"FB-check:     {args.fb_threshold}px")
    print(f"ECC 切换阈值: 内点 < {args.min_inliers}")
    print(f"FOV 高斯 σ:   {args.fov_sigma}")
    print("=" * 70)

    stabilizer = VideoStabilizer(
        smoothing_radius=args.smoothing,
        bucket_size=args.bucket_size,
        quality_level=args.quality_level,
        min_distance=args.min_distance,
        fb_threshold=args.fb_threshold,
        min_inliers=args.min_inliers,
        fov_sigma=args.fov_sigma
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