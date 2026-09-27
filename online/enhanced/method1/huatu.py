"""
test_timing.py
在 test.py 基础上，增加各模块耗时统计。
测量项目：
  - 光流估计 get_transform()
  - 模型推理（含归一化/反归一化）
  - warpAffine 变换
  - fix_border 边界修复
  - 写帧 out.write()
  - 单帧总耗时
  - 整体汇总统计（控制台表格 + 可视化图表）
"""

import argparse
import time
from collections import deque, defaultdict

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import load_model

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 计时工具
# ============================================================

class Timer:
    """轻量级分模块计时器（单位：ms）"""

    def __init__(self):
        self._records: dict[str, list[float]] = defaultdict(list)
        self._start:   dict[str, float]       = {}

    def start(self, name: str):
        self._start[name] = time.perf_counter()

    def stop(self, name: str):
        elapsed = time.perf_counter() - self._start[name]
        self._records[name].append(elapsed * 1000)   # → ms

    def summary(self):
        """在控制台打印各模块耗时汇总表格"""
        print("\n" + "=" * 66)
        print(f"  {'模块':<24} {'调用次数':>6}  {'总耗时(ms)':>10}  {'均值(ms)':>9}  {'占比':>6}")
        print("-" * 66)
        # 只统计子模块（排除"单帧总耗时"）用于占比计算
        sub_names  = [k for k in self._records if k != "单帧总耗时"]
        total_sub  = sum(sum(self._records[k]) for k in sub_names)
        for name, vals in self._records.items():
            total = sum(vals)
            mean  = total / len(vals)
            if name == "单帧总耗时":
                ratio_str = "—"
            else:
                ratio_str = f"{total / total_sub * 100:5.1f}%"
            print(f"  {name:<24} {len(vals):>6}  {total:>10.1f}  {mean:>9.3f}  {ratio_str:>6}")
        print("-" * 66)
        total_frame = sum(self._records["单帧总耗时"])
        n_frames    = len(self._records["单帧总耗时"])
        print(f"  平均每帧总耗时: {total_frame/n_frames:.2f} ms  "
              f"→ 理论最大帧率: {1000/(total_frame/n_frames):.1f} FPS")
        print("=" * 66)

    def plot(self, save_path: str = "./timing_report.png"):
        """绘制各模块耗时柱状图 + 总耗时占比饼图"""
        sub_names  = [k for k in self._records if k != "单帧总耗时"]
        means      = [np.mean(self._records[k]) for k in sub_names]
        totals     = [np.sum (self._records[k]) for k in sub_names]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("各模块运行耗时分析", fontsize=13, fontweight="bold")

        # 左图：均值柱状图
        bars = axes[0].bar(sub_names, means, color="steelblue", edgecolor="white")
        axes[0].set_title("各模块平均耗时 (ms/帧)")
        axes[0].set_ylabel("耗时 (ms)")
        axes[0].tick_params(axis="x", rotation=25)
        for bar, val in zip(bars, means):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9
            )

        # 右图：总耗时占比饼图
        axes[1].pie(totals, labels=sub_names, autopct="%1.1f%%", startangle=140)
        axes[1].set_title("各模块总耗时占比")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✅ 耗时报告图已保存: {save_path}")
        plt.show()


# ============================================================
# 2. 归一化工具（与 train.py 保持一致）
# ============================================================

def normalize(traj: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (traj - mean) / std

def denormalize(traj: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return traj * std + mean


# ============================================================
# 3. 光流变换估计
# ============================================================

def get_transform(prev_frame: np.ndarray, curr_frame: np.ndarray):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
    )
    if prev_pts is None:
        return 0.0, 0.0, 0.0

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    if curr_pts is None:
        return 0.0, 0.0, 0.0

    good_prev = prev_pts[status.flatten() == 1]
    good_curr = curr_pts[status.flatten() == 1]
    if len(good_prev) < 10:
        return 0.0, 0.0, 0.0

    m, _ = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC)
    if m is None:
        return 0.0, 0.0, 0.0

    return float(m[0, 2]), float(m[1, 2]), float(np.arctan2(m[1, 0], m[0, 0]))


# ============================================================
# 4. 边界修复
# ============================================================

def fix_border(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.1)
    return cv2.warpAffine(frame, T, (w, h))


# ============================================================
# 5. 轨迹对比可视化
# ============================================================

def plot_trajectory(
    orig_dx, orig_dy, orig_da,
    smth_dx, smth_dy, smth_da,
    save_path: str = "./trajectory_comparison.png"
):
    frames = range(len(orig_dx))
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("累计平移与旋转数据对比", fontsize=13, fontweight="bold")
    data = [
        (orig_dx, smth_dx, "dx (像素)", "原始 dx", "平滑 dx"),
        (orig_dy, smth_dy, "dy (像素)", "原始 dy", "平滑 dy"),
        (orig_da, smth_da, "da (弧度)", "原始 da", "平滑 da"),
    ]
    for ax, (orig, smth, ylabel, lo, ls) in zip(axes, data):
        ax.plot(frames, orig, "r-", alpha=0.7, linewidth=0.8, label=lo)
        ax.plot(frames, smth, "b-", linewidth=1.5,            label=ls)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("帧数")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ 轨迹对比图已保存: {save_path}")
    plt.show()


# ============================================================
# 6. 推理稳定主函数（含计时）
# ============================================================

def stabilize(
    video_input:  str,
    video_output: str,
    model_path:   str  = "./multi_model_ensemble.pth",
    norm_dir:     str  = ".",
    window_size:  int  = 125,
    show_preview: bool = True
):
    import os
    timer = Timer()   # ← 计时器实例

    # --- 加载归一化参数 ---
    mean_path = os.path.join(norm_dir, "noisy_mean.npy")
    std_path  = os.path.join(norm_dir, "noisy_std.npy")
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(
            f"未找到归一化参数文件，请先运行 train.py。\n"
            f"  期望路径: {mean_path}, {std_path}"
        )
    noisy_mean = np.load(mean_path)
    noisy_std  = np.load(std_path)
    print(f"归一化参数已加载: mean={noisy_mean.round(4)}, std={noisy_std.round(4)}")

    # --- 加载模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(model_path, device)

    # --- 打开视频 ---
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_input}")

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {width}x{height}  {fps}fps  {total}帧")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("无法读取第一帧")

    traj_window = deque(maxlen=window_size)
    cum = np.zeros(3, dtype=np.float32)
    traj_window.append(cum.copy())

    orig_dx, orig_dy, orig_da = [], [], []
    smth_dx, smth_dy, smth_da = [], [], []
    frame_count = 0

    print("\n开始处理视频...")
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"  进度: {frame_count}/{total}  ({frame_count/total*100:.1f}%)")

        # ══════════════ 单帧计时开始 ══════════════
        timer.start("单帧总耗时")

        # ── 1. 光流估计 ──────────────────────────
        timer.start("光流估计")
        dx, dy, da = get_transform(prev_frame, curr_frame)
        timer.stop("光流估计")

        cum = cum + np.array([dx, dy, da], dtype=np.float32)
        traj_window.append(cum.copy())

        # ── 2. 模型推理（含归一化/反归一化）──────
        timer.start("模型推理")
        input_window = np.array(traj_window, dtype=np.float32)
        norm_input   = normalize(input_window, noisy_mean, noisy_std)
        input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0).to(device)
        if len(traj_window) == window_size:
            with torch.no_grad():
                pred = model(input_tensor).squeeze(0).cpu().numpy()
            smooth_pt = denormalize(pred, noisy_mean, noisy_std)
        else:
            smooth_pt = cum.copy()
        timer.stop("模型推理")

        smooth_dx, smooth_dy, smooth_da = smooth_pt
        orig_dx.append(float(cum[0])); orig_dy.append(float(cum[1])); orig_da.append(float(cum[2]))
        smth_dx.append(smooth_dx);     smth_dy.append(smooth_dy);     smth_da.append(smooth_da)

        # ── 3. warpAffine 变换 ───────────────────
        diff_dx = smooth_dx - float(cum[0])
        diff_dy = smooth_dy - float(cum[1])
        diff_da = smooth_da - float(cum[2])
        corrected_dx = dx + diff_dx
        corrected_dy = dy + diff_dy
        corrected_da = da + diff_da
        T_mat = np.array([
            [np.cos(corrected_da), -np.sin(corrected_da), corrected_dx],
            [np.sin(corrected_da),  np.cos(corrected_da), corrected_dy]
        ], dtype=np.float32)

        timer.start("warpAffine 变换")
        stabilized = cv2.warpAffine(curr_frame, T_mat, (width, height))
        timer.stop("warpAffine 变换")

        # ── 4. 边界修复 ──────────────────────────
        timer.start("fix_border 边界修复")
        stabilized = fix_border(stabilized)
        timer.stop("fix_border 边界修复")

        # ── 5. 写帧 ──────────────────────────────
        timer.start("写帧 out.write")
        out.write(stabilized)
        timer.stop("写帧 out.write")

        # ══════════════ 单帧计时结束 ══════════════
        timer.stop("单帧总耗时")

        if show_preview:
            canvas = np.hstack((curr_frame, stabilized))
            cv2.imshow("Original | Stabilized", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("用户中断")
                break

        prev_frame = curr_frame

    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"\n✅ 稳定完成  输出: {video_output}  处理帧数: {frame_count}/{total}")

    # ── 6. 打印 & 绘制耗时报告 ──────────────────
    timer.summary()
    timer.plot(save_path="./timing_report.png")

    # --- 绘制轨迹对比图 ---
    plot_trajectory(
        orig_dx, orig_dy, orig_da,
        smth_dx, smth_dy, smth_da,
        save_path="./trajectory_comparison.png"
    )


# ============================================================
# 7. 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="视频稳定推理（含耗时统计）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input",       default="./24.mp4",                   help="输入视频路径")
    parser.add_argument("--output",      default="./result24.mp4",              help="输出视频路径")
    parser.add_argument("--model",       default="./multi_model_ensemble.pth",  help="模型权重路径")
    parser.add_argument("--norm_dir",    default=".",                            help="归一化参数目录")
    parser.add_argument("--window_size", type=int, default=125,                  help="滑动窗口大小")
    parser.add_argument("--no_preview",  action="store_true",                   help="关闭实时预览窗口")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stabilize(
        video_input  = args.input,
        video_output = args.output,
        model_path   = args.model,
        norm_dir     = args.norm_dir,
        window_size  = args.window_size,
        show_preview = not args.no_preview
    )