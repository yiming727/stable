"""
test.py
包含：
  - 光流变换估计 get_transform()
  - 边界修复 fix_border()
  - 推理稳定主函数 stabilize()
  - 轨迹对比可视化
  - 命令行入口
"""

import argparse
from collections import deque

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import load_model

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 归一化工具（与 train.py 保持一致）
# ============================================================

def normalize(traj: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (traj - mean) / std

def denormalize(traj: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return traj * std + mean


# ============================================================
# 2. 光流变换估计
# ============================================================

def get_transform(prev_frame: np.ndarray, curr_frame: np.ndarray):
    """
    用 Lucas-Kanade 光流估计相邻帧之间的仿射变换。
    返回: (dx, dy, da) —— 平移量（像素）和旋转角（弧度）
    特征点不足或估计失败时返回 (0, 0, 0)。
    """
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
# 3. 边界修复
# ============================================================

def fix_border(frame: np.ndarray) -> np.ndarray:
    """
    对稳定后帧做轻微放大（1.1x），消除 warpAffine 产生的黑边。
    """
    h, w = frame.shape[:2]
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.1)
    return cv2.warpAffine(frame, T, (w, h))


# ============================================================
# 4. 轨迹对比可视化
# ============================================================

def plot_trajectory(
    orig_dx, orig_dy, orig_da,
    smth_dx, smth_dy, smth_da,
    save_path: str = "./trajectory_comparison.png"
):
    """绘制原始轨迹与平滑轨迹的对比曲线并保存。"""
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
# 5. 推理稳定主函数
# ============================================================

def stabilize(
    video_input:  str,
    video_output: str,
    model_path:   str = "./multi_model_ensemble.pth",
    norm_dir:     str = ".",
    window_size:  int = 125,
    show_preview: bool = True
):
    """
    对输入视频做深度学习稳定，输出稳定后视频。

    归一化策略：
      - 加载训练时保存的 noisy_mean / noisy_std
      - 推理输入用同一套参数归一化，输出用同一套参数反归一化
      - 与 train.py 完全一致，消除训练/推理域不匹配问题
    """
    import os

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
    cum = np.zeros(3, dtype=np.float32)   # [cum_dx, cum_dy, cum_da]
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

        # 光流估计
        dx, dy, da = get_transform(prev_frame, curr_frame)
        cum = cum + np.array([dx, dy, da], dtype=np.float32)
        traj_window.append(cum.copy())

        # 用 noisy_mean/std 归一化输入
        input_window = np.array(traj_window, dtype=np.float32)
        norm_input   = normalize(input_window, noisy_mean, noisy_std)
        input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0).to(device)

        if len(traj_window) == window_size:
            # 窗口已满：使用模型预测
            with torch.no_grad():
                pred = model(input_tensor).squeeze(0).cpu().numpy()  # (3,)
            smooth_pt = denormalize(pred, noisy_mean, noisy_std)
        else:
            # 窗口未满：冷启动阶段直接输出原始累计轨迹（不做任何平滑）
            smooth_pt = cum.copy()

        smooth_dx, smooth_dy, smooth_da = smooth_pt

        orig_dx.append(float(cum[0])); orig_dy.append(float(cum[1])); orig_da.append(float(cum[2]))
        smth_dx.append(smooth_dx);     smth_dy.append(smooth_dy);     smth_da.append(smooth_da)

        # 计算修正量并应用变换
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

        stabilized = cv2.warpAffine(curr_frame, T_mat, (width, height))
        stabilized = fix_border(stabilized)
        out.write(stabilized)

        if show_preview:
            canvas = np.hstack((curr_frame, stabilized))
            cv2.imshow("Original | Stabilized", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("用户中断")
                break

        prev_frame = curr_frame

    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"\n✅ 稳定完成  输出: {video_output}  处理帧数: {frame_count}/{total}")

    # --- 绘制轨迹对比图 ---
    plot_trajectory(
        orig_dx, orig_dy, orig_da,
        smth_dx, smth_dy, smth_da,
        save_path="./trajectory_comparison.png"
    )


# ============================================================
# 6. 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="视频稳定推理",
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