"""
基于光流 + 峰值检测的视频抖动检测。

来源（2 份**字节完全相同**的副本）：

    offline/basic/src/detection.py
    offline/enhanced/detection.py

这是全仓库唯一一对完全一致且**无任何文件引用**的重复
（两处的 `__main__` 演示都直接写在第 122 行、没有保护），
所以改动的爆炸半径为 0。

⚠️ 与原文件的一处**有意差异**：原 detection.py 在第 122 行
**裸执行** `detect_shake('../Data/test1.mp4')`，导致任何 `import detection`
都会立刻读视频、画图并阻塞在 plt.show()。这里把演示挪进了
`if __name__ == '__main__':`，函数体本身逐行保持一致。
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_optical_flow_with_ransac(prev_frame, next_frame):
    """用 Farneback 光流估计相邻帧运动，RANSAC 筛内点后返回运动幅值。"""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    points = np.column_stack((x.ravel(), y.ravel()))
    flow_vectors = flow.reshape(-1, 2)

    model, inliers = cv2.estimateAffinePartial2D(points, points + flow_vectors, method=cv2.RANSAC)

    if model is None:
        return np.zeros_like(flow[..., 0])

    inliers_mask = inliers.ravel() == 1
    filtered_flow = flow_vectors[inliers_mask]

    magnitude, _ = cv2.cartToPolar(filtered_flow[:, 0], filtered_flow[:, 1])

    return magnitude


def detect_peaks_and_widths(signal):
    """检测运动信号中的峰值及其宽度。"""
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    height_threshold = np.nanmean(signal) + 0.5 * np.nanstd(signal)
    prominence_threshold = 0.1 * np.max(signal)
    print(np.isnan(signal).any())
    print(np.isinf(signal).any())
    print(height_threshold)
    print(prominence_threshold)

    peaks, properties = find_peaks(signal, height=height_threshold, prominence=prominence_threshold)
    results_half = peak_widths(signal, peaks, rel_height=0.5)

    return peaks, results_half[0]


def analyze_shake_per_frame(signal):
    """把峰值位置按宽度展开成受抖动影响的帧索引集合。"""
    peaks, widths = detect_peaks_and_widths(signal)
    shake_frames = []

    for peak in peaks:
        shake_frames.extend(
            range(max(0, peak - int(widths[0] // 2)), min(len(signal), peak + int(widths[0] // 2)))
        )

    return sorted(set(shake_frames))


def detect_shake(video_path, plot=True):
    """检测视频中的抖动帧。

    参数:
        video_path: 视频路径
        plot:       是否绘制运动信号曲线（原实现固定会画并 show()）
    返回:
        抖动帧索引列表；视频读不出来时返回 None
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    if not ret:
        print("无法读取视频")
        return None

    shake_signals = []
    total_frames = 0

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        magnitude = calculate_optical_flow_with_ransac(prev_frame, next_frame)

        avg_motion = np.mean(magnitude) if magnitude.size > 0 else 0
        shake_signals.append(avg_motion)

        prev_frame = next_frame
        total_frames += 1

    cap.release()

    shake_frames = analyze_shake_per_frame(shake_signals)

    if len(shake_frames) > 0:
        shake_ratio = len(shake_frames) / total_frames
        print(f"检测到抖动，抖动帧数：{len(shake_frames)}，占比：{shake_ratio:.2f}")
        print(f"抖动的帧索引为：{shake_frames}")
    else:
        print("视频较为平稳，无明显抖动。")

    if plot:
        plt.plot(shake_signals)
        plt.title("运动信号")
        plt.xlabel("帧数")
        plt.ylabel("运动强度")
        plt.show()

    return shake_frames


if __name__ == '__main__':
    detect_shake('../Data/test1.mp4')
