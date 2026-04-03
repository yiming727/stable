"""
画Farneback光流图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import find_peaks, peak_widths

# 设置字体为SimHei（黑体）以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_optical_flow_with_ransac(prev_frame, next_frame):
    """
    计算光流，返回相邻帧之间的运动矢量及原始光流场
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    points = np.column_stack((x.ravel(), y.ravel()))
    flow_vectors = flow.reshape(-1, 2)

    model, inliers = cv2.estimateAffinePartial2D(points, points + flow_vectors, method=cv2.RANSAC)

    if model is None:
        return np.zeros_like(flow[..., 0]), flow

    inliers_mask = inliers.ravel() == 1
    filtered_flow = flow_vectors[inliers_mask]

    magnitude, _ = cv2.cartToPolar(filtered_flow[:, 0], filtered_flow[:, 1])

    return magnitude, flow  # 同时返回原始光流场


def detect_peaks_and_widths(signal):
    """
    检测信号中的峰值并计算其宽度。
    """
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
    """
    对运动信号进行峰值检测，基于轨迹平滑性检测抖动并返回抖动的帧数。
    """
    peaks, widths = detect_peaks_and_widths(signal)
    shake_frames = []

    for peak in peaks:
        shake_frames.extend(range(
            max(0, peak - int(widths[0] // 2)),
            min(len(signal), peak + int(widths[0] // 2))
        ))

    return sorted(set(shake_frames))


def visualize_optical_flow(flow, save_path=None):
    """
    可视化光流场：左图为HSV颜色编码，右图为箭头向量图
    """
    h, w = flow.shape[:2]
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)
    angle = np.arctan2(flow_y, flow_x)

    # ── 左图：HSV颜色编码 ──
    hue = (angle + np.pi) / (2 * np.pi)
    sat = np.ones_like(hue)
    val = magnitude / (magnitude.max() + 1e-5)
    rgb_img = mcolors.hsv_to_rgb(np.stack([hue, sat, val], axis=-1))

    # ── 右图：箭头向量图（归一化方向） ──
    norm = magnitude + 1e-5
    flow_x_n = flow_x / norm
    flow_y_n = flow_y / norm

    step = max(h // 15, 10)
    ys = np.arange(step, h, step)
    xs = np.arange(step, w, step)
    X, Y = np.meshgrid(xs, ys)
    U = flow_x_n[Y, X]
    V = -flow_y_n[Y, X]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(rgb_img, origin='upper')
    axes[0].axhline(h // 2, color='black', linewidth=1.5)
    axes[0].axvline(w // 2, color='black', linewidth=1.5)
    axes[0].set_title('光流 - 颜色编码', fontsize=12)
    axes[0].axis('off')

    axes[1].set_facecolor('white')
    axes[1].quiver(X, Y, U, V, color='black',
                   scale=18, width=0.004, headwidth=4, headlength=5)
    axes[1].set_xlim(0, w)
    axes[1].set_ylim(h, 0)
    axes[1].set_aspect('equal')
    axes[1].set_title('光流 - 箭头向量图', fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"光流图已保存至：{save_path}")
    plt.show()
    plt.close()


def detect_shake(video_path, save_path=None, flow_save_path=None):
    """
    使用峰值检测检测视频中的抖动，并生成运动信号轨迹图和光流可视化图
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    if not ret:
        print("无法读取视频")
        return

    shake_signals = []
    total_frames = 0
    last_flow = None

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        magnitude, flow = calculate_optical_flow_with_ransac(prev_frame, next_frame)

        avg_motion = np.mean(magnitude) if magnitude.size > 0 else 0
        shake_signals.append(avg_motion)

        last_flow = flow
        prev_frame = next_frame
        total_frames += 1

    cap.release()

    # ── 峰值检测分析抖动 ──
    shake_frames = analyze_shake_per_frame(shake_signals)

    if len(shake_frames) > 0:
        shake_ratio = len(shake_frames) / total_frames
        print(f"检测到抖动，抖动帧数：{len(shake_frames)}，占比：{shake_ratio:.2f}")
        print(f"抖动的帧索引为：{shake_frames}")
    else:
        print("视频较为平稳，无明显抖动。")

    # ── 生成蓝色运动信号轨迹图 ──
    fig, ax = plt.subplots(figsize=(10, 5))

    # ★ 曲线改为蓝色
    ax.plot(shake_signals, color='#1565C0', linewidth=1.2, label='运动强度')

    # ★ 抖动区域背景改为浅蓝色
    if len(shake_frames) > 0:
        shake_regions = []
        start = shake_frames[0]
        prev = shake_frames[0]
        for f in shake_frames[1:]:
            if f != prev + 1:
                shake_regions.append((start, prev))
                start = f
            prev = f
        shake_regions.append((start, prev))

        for i, (s, e) in enumerate(shake_regions):
            ax.axvspan(s, e,
                       color='#90CAF9',   # ★ 浅蓝色填充
                       alpha=0.4,
                       label='抖动区域' if i == 0 else "")

    ax.set_title('运动信号', fontsize=14)
    ax.set_xlabel('帧数', fontsize=12)
    ax.set_ylabel('运动强度', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    # ★ 网格线改为浅蓝色
    ax.grid(True, linestyle='--', alpha=0.4, color='#90CAF9')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"轨迹图已保存至：{save_path}")

    plt.show()
    plt.close()

    # ── 生成光流可视化图 ──
    if last_flow is not None:
        visualize_optical_flow(last_flow, save_path=flow_save_path)


# ── 使用示例 ──
video_path = './24.mp4'
detect_shake(
    video_path,
    save_path='motion_signal_blue.png',
    flow_save_path='optical_flow_visual.png'
)