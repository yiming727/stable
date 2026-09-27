"""
变换矩阵构造与黑边修复。

来源（4 份重复实现）：

    offline/basic/src/stable.py          build_transformation_matrix / extreme_corners
                                         min_auto_border_size / fix_border
    offline/basic/u16/stable.py          同上（仅局部变量名不同，行为一致）
    offline/basic/src/stable1.py         （不用 fix_border，改用 FOV 裁剪）
    offline/enhanced/fixborder.py        同上，**但**：
                                           - 独有 auto_border_start / auto_border_length
                                           - fix_border 缺少 borderMode=cv2.BORDER_REPLICATE

⚠️ 关键差异：`offline/basic` 的 fix_border 用 BORDER_REPLICATE（边缘像素外延），
`offline/enhanced/fixborder.py` 不带该参数（即默认 BORDER_CONSTANT，补 0 黑边）。
当变换把画面推出边界时，两者输出**确实不同**。所以这里做成 border_mode 参数，
各处传各自原来的值——不要"统一"，也不要"简化"掉这个参数。
"""

import math

import cv2
import numpy as np


def build_transformation_matrix(transform):
    """根据 (dx, dy, da) 构造 2x3 仿射变换矩阵。

    参数:
        transform: 长度 3 的序列 (dx, dy, da)，da 为弧度
    返回:
        2x3 变换矩阵
    """
    m = np.zeros((2, 3))
    m[0, 0] = np.cos(transform[2])
    m[0, 1] = -np.sin(transform[2])
    m[1, 0] = np.sin(transform[2])
    m[1, 1] = np.cos(transform[2])
    m[0, 2] = transform[0]
    m[1, 2] = transform[1]
    return m


def extreme_corners(frame, transforms):
    """统计所有帧变换后，画面四角相对原位置的偏移极值。

    参数:
        frame:      任意一帧（只用来取高宽）
        transforms: shape=(N, 3) 的全局变换序列
    返回:
        {'min_x', 'min_y', 'max_x', 'max_y'}
    """
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
    """由极值字典算出最小安全边距（向上取整）。"""
    return math.ceil(max(abs(v) for v in extreme_frame_corners.values()))


def fix_border(frame, extreme_frame_corners, border_size,
               border_mode=cv2.BORDER_REPLICATE):
    """以画面中心等比放大，把变换造成的黑边推出可视区。

    参数:
        frame:                 当前帧
        extreme_frame_corners: extreme_corners() 的返回值
        border_size:           安全边距；为 0 时原样返回
        border_mode:           边界填充方式。默认 BORDER_REPLICATE，
                               对应 offline/basic 的实现；
                               offline/enhanced/fixborder.py 请传 BORDER_CONSTANT。
    返回:
        处理后的帧
    """
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
    m = cv2.getRotationMatrix2D(center, 0, scale)
    return cv2.warpAffine(
        frame, m, (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode
    )


# ---------------------------------------------------------------------------
# 以下两个只有 offline/enhanced/fixborder.py 用到，其它实现里没有对应物。
# 它们配合 "自动裁剪" 路线使用，不是 fix_border 的替代品。
# ---------------------------------------------------------------------------

def auto_border_start(min_corner_point, border_size):
    """最小安全边距区域的起始坐标。"""
    return math.floor(border_size - abs(min_corner_point))


def auto_border_length(frame_dim, extreme_corner, border_size):
    """最小安全边距区域的长度。"""
    return math.ceil(frame_dim - (border_size - extreme_corner))


def fix_border_simple(frame, scale=1.04):
    """按固定比例中心缩放的简易黑边修复。

    来源：offline/basic/视频防抖.py、src/huatu1.py、src/amplitude_limit.py
    这三处各自的实现里就直接写着 cv2.getRotationMatrix2D(center, 0, 1.04)，
    手写重复度不高，此处仅作为规范实现收录。
    """
    h, w = frame.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(frame, m, (w, h))
