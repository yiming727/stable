"""
特征点提取与跟踪一致性检验。

来源（多份副本，**并非全都等价**——差异已参数化或标注）：

    GridCalculator.calculate_optimal_grid_size
        offline/basic/src/stable.py   与  offline/basic/u16/stable.py  完全一致

    GridCalculator.extract_grid_points
        src/stable.py   返回 (N, 1, 2)，无点时返回 None
        u16/stable.py   返回 (N, 2)，无点时返回空数组
        ⚠️ 返回形状不同，是会静默出错的差异 → 用 shape 参数区分

    detect_features                    只有 src/stable.py 有

    fb_check / estimate_transform_ecc  src/stable.py 与 src/stable1.py 完全一致

    compute_bucket_grid                src/stable1.py 与 src/huatu2.py 完全一致
    detect_bucketing                   两者函数体一致，但默认 min_distance
                                       在 stable1.py 是 5、在 huatu2.py 是 10
                                       → 保留 5 并在此注明
"""

import cv2
import numpy as np


class GridCalculator:
    """动态网格计算器：按分辨率自动定网格，每格取 Harris 最强角点。"""

    @staticmethod
    def calculate_optimal_grid_size(width, height,
                                    target_grid_size=100,
                                    min_grids=3,
                                    max_grids=8):
        """根据视频分辨率动态计算网格大小。

        返回:
            grid_size: (rows, cols)
            grid_w:    每格宽度（像素）
            grid_h:    每格高度（像素）
        """
        cols = max(min_grids, min(max_grids, width // target_grid_size))
        rows = max(min_grids, min(max_grids, height // target_grid_size))

        grid_w = width // cols
        grid_h = height // rows

        return (rows, cols), grid_w, grid_h

    @staticmethod
    def extract_grid_points(gray_image, grid_size, grid_w, grid_h, shape='3d'):
        """在每个网格内取 Harris 响应最强的点作为该格代表点。

        参数:
            gray_image: 8 位灰度图
            grid_size:  (rows, cols)
            grid_w:     每格宽度
            grid_h:     每格高度
            shape:      '3d' → 返回 (N, 1, 2)，无点时返回 None
                                （对应 offline/basic/src/stable.py）
                        '2d' → 返回 (N, 2)，无点时返回空数组
                                （对应 offline/basic/u16/stable.py）

        ⚠️ 两种形状不能混用：cv2.calcOpticalFlowPyrLK 需要 '3d'。
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

                harris_response = cv2.cornerHarris(grid_roi, 2, 3, 0.04)
                _, _, _, max_loc = cv2.minMaxLoc(harris_response)

                global_x = grid_x + max_loc[0]
                global_y = grid_y + max_loc[1]

                grid_points.append((global_x, global_y))

        if shape == '2d':
            return np.array(grid_points, dtype=np.float32).reshape(-1, 2)

        if not grid_points:
            return None
        return np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)


def detect_features(gray_image, grid_calculator, grid_size, grid_w, grid_h,
                    max_corners=200, quality_level=0.01, min_distance=30):
    """融合两路特征点：动态网格 Harris（空间均匀）+ 全局 Shi-Tomasi（高质量）。

    返回:
        合并后的特征点，shape=(N, 1, 2)；若两路都为空则返回 None
    """
    grid_pts = grid_calculator.extract_grid_points(
        gray_image, grid_size, grid_w, grid_h
    )

    good_pts = cv2.goodFeaturesToTrack(
        gray_image,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=3
    )
    if good_pts is not None:
        good_pts = good_pts.reshape(-1, 1, 2)

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


def fb_check(prev_gray, curr_gray, prev_pts, threshold=1.0):
    """前向—反向重跟踪一致性检验（过滤跟踪失败与回投误差过大的点）。

    参数:
        prev_gray: 参考帧灰度图
        curr_gray: 当前帧灰度图
        prev_pts:  参考帧特征点，shape=(N, 1, 2)
        threshold: 回投误差阈值（像素），默认 1.0

    返回:
        (通过检验的参考帧点, 对应的当前帧点)，均为 shape=(M, 1, 2)
    """
    lk_params = dict(
        winSize=(21, 21), maxLevel=3,
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


def estimate_transform_ecc(prev_gray, curr_gray):
    """基于增强相关系数（ECC）的直接法仿射配准，作为特征点不足时的兜底。

    返回:
        2x3 仿射矩阵；ECC 迭代失败则返回单位矩阵
    """
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-4)
    try:
        _, warp_matrix = cv2.findTransformECC(
            prev_gray, curr_gray, warp_matrix, cv2.MOTION_AFFINE, criteria
        )
    except cv2.error:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    return warp_matrix


def compute_bucket_grid(width, height, bucket_size=120):
    """按桶尺寸算网格行列数。返回 (grid_rows, grid_cols)。"""
    grid_cols = max(1, round(width / bucket_size))
    grid_rows = max(1, round(height / bucket_size))
    return grid_rows, grid_cols


def detect_bucketing(img, grid_rows, grid_cols, quality=0.01, min_distance=5):
    """分桶均匀提取角点（每桶不限数量）。

    ⚠️ min_distance 默认 5，对应 offline/basic/src/stable1.py；
    src/huatu2.py 用的是 10，若要与它对齐请显式传 10。

    返回:
        shape=(N, 1, 2)；无角点时返回 None
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
