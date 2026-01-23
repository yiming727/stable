import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

# 设置字体为SimHei（黑体）以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_optical_flow_with_ransac(prev_frame, next_frame):
    """
    计算光流，返回相邻帧之间的运动矢量
    """
    # 转换为灰度图像
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # 使用Farneback法计算光流
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 提取光流中的点
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    points = np.column_stack((x.ravel(), y.ravel()))
    flow_vectors = flow.reshape(-1, 2)

    # 使用RANSAC筛选运动矢量
    model, inliers = cv2.estimateAffinePartial2D(points, points + flow_vectors, method=cv2.RANSAC)

    if model is None:
        return np.zeros_like(flow[..., 0])  # 返回零矢量

    # 筛选内点
    inliers_mask = inliers.ravel() == 1
    filtered_flow = flow_vectors[inliers_mask]

    # 计算光流的幅值
    magnitude, _ = cv2.cartToPolar(filtered_flow[:, 0], filtered_flow[:, 1])

    return magnitude


def detect_peaks_and_widths(signal):
    """
    检测信号中的峰值并计算其宽度。
    """
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    height_threshold = np.nanmean(signal) + 0.5 * np.nanstd(signal)
    prominence_threshold = 0.1 * np.max(signal)
    print(np.isnan(signal).any())  # 检查是否有 NaN
    print(np.isinf(signal).any())  # 检查是否有 inf
    print(height_threshold)
    print(prominence_threshold)

    peaks, properties = find_peaks(signal, height=height_threshold, prominence=prominence_threshold)
    results_half = peak_widths(signal, peaks, rel_height=0.5)

    return peaks, results_half[0]


def analyze_shake_per_frame(signal):
    """
    对运动信号进行峰值检测，基于轨迹平滑性检测抖动并返回抖动的帧数。
    """
    peaks, widths = detect_peaks_and_widths(signal)
    shake_frames = []

    for peak in peaks:
        shake_frames.extend(range(max(0, peak - int(widths[0] // 2)), min(len(signal), peak + int(widths[0] // 2))))

    return sorted(set(shake_frames))


def detect_shake(video_path):
    """
    使用峰值检测检测视频中的抖动
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    if not ret:
        print("无法读取视频")
        return

    shake_signals = []
    total_frames = 0

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        # 计算光流
        magnitude = calculate_optical_flow_with_ransac(prev_frame, next_frame)

        avg_motion = np.mean(magnitude) if magnitude.size > 0 else 0
        shake_signals.append(avg_motion)

        prev_frame = next_frame
        total_frames += 1

    cap.release()

    # 使用峰值检测分析局部抖动
    shake_frames = analyze_shake_per_frame(shake_signals)

    if len(shake_frames) > 0:
        shake_ratio = len(shake_frames) / total_frames
        print(f"检测到抖动，抖动帧数：{len(shake_frames)}，占比：{shake_ratio:.2f}")
        print(f"抖动的帧索引为：{shake_frames}")
    else:
        print(f"视频较为平稳，无明显抖动。")

    plt.plot(shake_signals)
    plt.title("运动信号")
    plt.xlabel("帧数")
    plt.ylabel("运动强度")
    plt.show()


# 使用示例
video_path = '../Data/test1.mp4'
detect_shake(video_path)
