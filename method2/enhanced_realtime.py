"""
增强版实时视频稳像推理脚本
核心改进：
1. 在线实时推理模式（流式处理）
2. 自适应平滑策略
3. 运动补偿校正
4. 实时性能优化
"""
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from improved_model import ImprovedStabilizerNet, OnlineStabilizer, HybridStabilizer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class StabilizerConfig:
    """稳像器配置"""
    # 运动检测
    max_corners: int = 200
    quality_level: float = 0.01
    min_distance: int = 30
    block_size: int = 3
    ransac_threshold: float = 3.0

    # 网络参数
    window_size: int = 30  # 减小窗口加速推理
    model_path: str = "improved_stabilizer.pth"
    use_gpu: bool = False

    # 后处理平滑
    post_smoothing_alpha: float = 0.25  # EMA平滑因子
    enable_motion_smooth: bool = True

    # 运动补偿
    motion_threshold: float = 50.0  # 异常运动阈值
    max_correction: float = 30.0  # 最大校正值

    # 边界处理
    border_scale: float = 1.05  # 边界放大比例
    fill_border: bool = True


class EnhancedMotionEstimator:
    """增强版运动估计器"""
    def __init__(self, config: StabilizerConfig):
        self.config = config

        # 特征点跟踪
        self.prev_pts = None
        self.prev_gray = None

        # 运动历史
        self.motion_history = deque(maxlen=10)
        self.frame_count = 0

    def estimate_motion(self, prev_frame, curr_frame) -> Tuple[float, float, float]:
        """
        估计两帧之间的运动变换

        Returns:
            dx, dy, da: 平移和旋转
        """
        if self.frame_count == 0:
            self.prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            self.frame_count += 1
            return 0.0, 0.0, 0.0

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 第一帧检测特征点
        if self.prev_pts is None:
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=self.config.max_corners,
                qualityLevel=self.config.quality_level,
                minDistance=self.config.min_distance,
                blockSize=self.config.block_size
            )

        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=self.config.max_corners,
                qualityLevel=self.config.quality_level,
                minDistance=self.config.min_distance,
                blockSize=self.config.block_size
            )
            if self.prev_pts is None:
                return 0.0, 0.0, 0.0

        # 光流跟踪
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_pts, None
        )

        if curr_pts is None:
            return 0.0, 0.0, 0.0

        # 筛选有效特征点
        good_prev = self.prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < 10 or len(good_curr) < 10:
            return 0.0, 0.0, 0.0

        # 估计仿射变换
        try:
            m = cv2.estimateAffinePartial2D(
                good_prev, good_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold
            )[0]

            if m is None:
                return 0.0, 0.0, 0.0

        except:
            return 0.0, 0.0, 0.0

        dx = float(m[0, 2])
        dy = float(m[1, 2])
        da = float(np.arctan2(m[1, 0], m[0, 0]))

        # 异常值过滤
        dx, dy, da = self._filter_outliers(dx, dy, da)

        # 更新状态
        self.prev_gray = curr_gray
        self.prev_pts = good_curr.reshape(-1, 1, 2)
        self.frame_count += 1

        return dx, dy, da

    def _filter_outliers(self, dx, dy, da) -> Tuple[float, float, float]:
        """过滤异常运动值"""
        self.motion_history.append([dx, dy, da])

        if len(self.motion_history) < 3:
            return dx, dy, da

        history = np.array(self.motion_history)
        mean = history.mean(axis=0)
        std = history.std(axis=0) + 1e-8

        # 检查是否超出阈值
        threshold = self.config.motion_threshold

        if abs(dx - mean[0]) > threshold * std[0] + 5:
            dx = mean[0]
        if abs(dy - mean[1]) > threshold * std[1] + 5:
            dy = mean[1]
        if abs(da - mean[2]) > threshold * std[2] + 0.1:
            da = mean[2]

        return dx, dy, da

    def reset(self):
        """重置状态"""
        self.prev_pts = None
        self.prev_gray = None
        self.motion_history.clear()
        self.frame_count = 0


class AdaptiveSmoother:
    """自适应平滑器"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.motion_var = None

    def update(self, dx, dy, da) -> Tuple[float, float, float]:
        """更新并返回平滑值"""
        self.history.append([dx, dy, da])

        if len(self.history) < 2:
            return dx, dy, da

        history = np.array(self.history)

        # 计算局部方差
        local_var = history.var(axis=0) + 1e-8

        # 自适应平滑因子：方差大时减小平滑，方差小时增加平滑
        alpha = np.clip(1.0 / (1.0 + local_var * 0.1), 0.1, 0.5)

        # 加权平均
        weights = np.array([alpha[i] * (1 - alpha[i]) ** (len(self.history) - 1 - j)
                          for i, j in enumerate(range(len(self.history)))])

        weights = weights / weights.sum()

        smoothed = history[-1]  # 简化处理

        return smoothed[0], smoothed[1], smoothed[2]

    def reset(self):
        self.history.clear()
        self.motion_var = None


def apply_stabilization_transform(frame, dx, dy, da, width, height,
                                   border_scale=1.05, fill=True) -> np.ndarray:
    """应用稳像变换"""
    # 构建变换矩阵
    transform = np.array([
        [np.cos(da), -np.sin(da), dx],
        [np.sin(da), np.cos(da), dy]
    ], dtype=np.float32)

    # 可选：边界放大
    if fill:
        # 计算变换后的边界
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        corners_extended = np.hstack([corners, np.ones((4, 1))])
        transformed_corners = (transform @ corners_extended.T).T

        min_x, max_x = transformed_corners[:, 0].min(), transformed_corners[:, 0].max()
        min_y, max_y = transformed_corners[:, 1].min(), transformed_corners[:, 1].max()

        # 计算需要的平移
        pad_x = max(0, -min_x, max_x - width * border_scale)
        pad_y = max(0, -min_y, max_y - height * border_scale)

        transform[0, 2] += pad_x
        transform[1, 2] += pad_y

    # 应用变换
    stabilized = cv2.warpAffine(
        frame, transform,
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return stabilized


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="增强版实时视频稳像")

    # 输入输出
    parser.add_argument('--input', type=str, default='24.mp4', help='输入视频路径')
    parser.add_argument('--output', type=str, default='result24_enhanced.mp4', help='输出视频路径')

    # 模型参数
    parser.add_argument('--model', type=str, default='improved_stabilizer.pth',
                        help='模型权重路径')
    parser.add_argument('--window_size', type=int, default=30,
                        help='网络输入窗口大小')

    # 处理参数
    parser.add_argument('--smoothing_method', type=str, default='hybrid',
                        choices=['network', 'hybrid', 'adaptive', 'ema'],
                        help='平滑方法')
    parser.add_argument('--post_alpha', type=float, default=0.25,
                        help='后处理平滑因子')
    parser.add_argument('--no_border_fill', action='store_true',
                        help='不进行边界填充')

    # 性能参数
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批处理大小')
    parser.add_argument('--show_preview', action='store_true',
                        help='显示预览窗口')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 配置
    config = StabilizerConfig(
        window_size=args.window_size,
        model_path=args.model,
        post_smoothing_alpha=args.post_alpha,
        fill_border=not args.no_border_fill
    )

    print("=" * 60)
    print("增强版实时视频稳像")
    print("=" * 60)
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"平滑方法: {args.smoothing_method}")
    print("=" * 60)

    # 初始化组件
    motion_estimator = EnhancedMotionEstimator(config)

    # 初始化稳像器
    if args.smoothing_method == 'network':
        stabilizer = OnlineStabilizer(
            args.model,
            window_size=config.window_size,
            device='cpu'
        )
    else:
        stabilizer = HybridStabilizer(
            args.model,
            window_size=config.window_size,
            device='cpu',
            post_smoothing=config.post_smoothing_alpha,
            motion_smooth=args.smoothing_method != 'network'
        )

    # 加载归一化参数
    mean_path = "stable_rel_mean.npy"
    std_path = "stable_rel_std.npy"

    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean = np.load(mean_path)
        std = np.load(std_path)
        stabilizer.set_normalization(mean, std)
        print(f"加载归一化参数: mean={mean}, std={std}")

    # 打开视频
    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"错误：无法打开视频 {args.input}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # 数据记录
    original_dx, original_dy, original_da = [], [], []
    smoothed_dx, smoothed_dy, smoothed_da = [], [], []

    # 相对运动
    prev_cum_dx = prev_cum_dy = prev_cum_da = 0.0
    rel_history = []

    # 处理循环
    ret, prev_frame = cap.read()
    frame_count = 0
    print("\n开始处理...")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 30 == 0:
            print(f"进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

        # 估计运动
        dx, dy, da = motion_estimator.estimate_motion(prev_frame, curr_frame)

        # 转换为相对运动
        if frame_count == 1:
            rel_dx, rel_dy, rel_da = dx, dy, da
        else:
            rel_dx = dx
            rel_dy = dy
            rel_da = da

        # 稳像
        smooth_rel_dx, smooth_rel_dy, smooth_rel_da = stabilizer.stabilize(rel_dx, rel_dy, rel_da)

        # 转换为累积校正值
        if frame_count == 1:
            smooth_cum_dx = smooth_rel_dx
            smooth_cum_dy = smooth_rel_dy
            smooth_cum_da = smooth_rel_da
        else:
            smooth_cum_dx = prev_cum_dx + smooth_rel_dx
            smooth_cum_dy = prev_cum_dy + smooth_rel_dy
            smooth_cum_da = prev_cum_da + smooth_rel_da

        # 计算校正
        correction_dx = smooth_cum_dx - prev_cum_dx
        correction_dy = smooth_cum_dy - prev_cum_dy
        correction_da = smooth_cum_da - prev_cum_da

        # 限制校正范围
        max_corr = config.max_correction
        correction_dx = np.clip(correction_dx, -max_corr, max_corr)
        correction_dy = np.clip(correction_dy, -max_corr, max_corr)
        correction_da = np.clip(correction_da, -0.1, 0.1)

        # 应用稳像
        stabilized = apply_stabilization_transform(
            curr_frame, correction_dx, correction_dy, correction_da,
            width, height, border_scale=config.border_scale,
            fill=config.fill_border
        )

        # 写入输出
        out.write(stabilized)

        # 显示预览
        if args.show_preview:
            canvas = np.hstack((curr_frame, stabilized))
            cv2.imshow("Original vs Stabilized", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户中断")
                break

        # 记录数据
        original_dx.append(dx)
        original_dy.append(dy)
        original_da.append(da)
        smoothed_dx.append(smooth_rel_dx)
        smoothed_dy.append(smooth_rel_dy)
        smoothed_da.append(smooth_rel_da)

        prev_cum_dx = smooth_cum_dx
        prev_cum_dy = smooth_cum_dy
        prev_cum_da = smooth_cum_da
        prev_frame = curr_frame

    # 清理
    cap.release()
    out.release()
    if args.show_preview:
        cv2.destroyAllWindows()

    print(f"\n处理完成！")
    print(f"输出文件: {args.output}")
    print(f"处理帧数: {frame_count}")

    # 绘制轨迹图
    print("\n绘制轨迹对比图...")
    plt.figure(figsize=(14, 10))

    frames = list(range(len(original_dx)))
    labels = ['dx (水平位移)', 'dy (垂直位移)', 'da (旋转角度)']

    for i in range(3):
        plt.subplot(3, 1, i + 1)

        if i == 2:
            # 角度需要处理周期性
            original_vals = np.array(original_da)
            smoothed_vals = np.array(smoothed_da)
        else:
            original_vals = original_dx if i == 0 else original_dy
            smoothed_vals = smoothed_dx if i == 0 else smoothed_dy

        plt.plot(frames, original_vals, 'r-', alpha=0.6, label='原始帧间运动', linewidth=1)
        plt.plot(frames, smoothed_vals, 'b-', alpha=0.8, label='平滑后运动', linewidth=1.5)

        plt.title(f'{labels[i]} - 帧间运动对比')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_trajectory_comparison.png', dpi=150)
    print("轨迹对比图已保存: enhanced_trajectory_comparison.png")

    plt.show()


if __name__ == "__main__":
    import os
    main()
