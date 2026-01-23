import cv2
import numpy as np
import math

# 修复由于变换导致的边界问题
def build_transformation_matrix(transform):
    """
    根据(dx, dy, da)构造仿射变换矩阵
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
    计算所有帧的极值
    :param frame: 当前视频帧
    :param transforms: 全局变换矩阵
    :return: {'min_x', 'min_y', 'max_x', 'max_y'}
    """
    h, w = frame.shape[:2]
    frame_corners = np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]], dtype='float32')
    frame_corners = np.array([frame_corners])

    min_x = min_y = max_x = max_y = 0
    for i in range(transforms.shape[0]):
        transform = transforms[i, :]
        transform_mat = build_transformation_matrix(transform)
        transformed_frame_corners = cv2.transform(frame_corners, transform_mat)
        delta_corners = transformed_frame_corners - frame_corners
        delta_y_corners = delta_corners[0][:, 1].tolist()
        delta_x_corners = delta_corners[0][:, 0].tolist()
        min_x = min([min_x] + delta_x_corners)
        min_y = min([min_y] + delta_y_corners)
        max_x = max([max_x] + delta_x_corners)
        max_y = max([max_y] + delta_y_corners)
    return {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}

def min_auto_border_size(extreme_frame_corners):
    """
    计算所有帧的极值，并返回最小安全边距
    :param extreme_frame_corners: 全局极值字典 {'min_x', 'min_y', 'max_x', 'max_y'}
    :return: 最小安全边距（int）
    """
    abs_extreme_corners = [abs(x) for x in extreme_frame_corners.values()]
    return math.ceil(max(abs_extreme_corners))

def auto_border_start(min_corner_point, border_size):
    """
    计算所有帧的极值，并返回最小安全边距的起始点
    :param min_corner_point: 最小边距点（int）
    :param border_size: 最小安全边距（int）
    :return: 最小安全边距的起始点（int）
    """
    return math.floor(border_size - abs(min_corner_point))

def auto_border_length(frame_dim, extreme_corner, border_size):
    """
    计算所有帧的极值，并返回最小安全边距的长度
    :param frame_dim: 帧尺寸（int）
    :param extreme_corner: 最小边距点（int）
    :param border_size: 最小安全边距（int）
    :return: 最小安全边距的长度（int）
    """
    return math.ceil(frame_dim - (border_size - extreme_corner))

def fix_border(frame, extreme_frame_corners, border_size):
    """
    修复由于变换导致的边界问题
    :param frame: 当前视频帧
    :param extreme_frame_corners: 全局极值字典 {'min_x', 'min_y', 'max_x', 'max_y'}
    :param border_size: 最小安全边距（int）
    :return: 修复后的帧
    """
    if border_size == 0:
        return frame

    frame_h, frame_w = frame.shape[:2]
    # 自动计算缩放比例
    scale_w = frame_w / (frame_w - abs(extreme_frame_corners['min_x']) - abs(extreme_frame_corners['max_x']))
    scale_h = frame_h / (frame_h - abs(extreme_frame_corners['min_y']) - abs(extreme_frame_corners['max_y']))
    scale = max(scale_w, scale_h)

    # warpAffine以中心缩放
    center = (frame_w / 2, frame_h / 2)
    M = cv2.getRotationMatrix2D(center, 0, scale)
    scaled_frame = cv2.warpAffine(frame, M, (frame_w, frame_h), flags=cv2.INTER_LINEAR)

    # 如果你想严格保证内容完整，可以再做一次center crop（不过warpAffine已按原分辨率输出）
    # 一般来说，warpAffine输出尺寸和输入一致，不需要再裁剪

    return scaled_frame



