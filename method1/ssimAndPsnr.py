"""
视频稳定质量评估模块（帧间相似性版本）
计算视频相邻帧之间的 SSIM 和 PSNR，衡量画面平滑程度（抖动量）
- 数值越高 → 帧间越平滑 → 稳定效果越好
- 原视频和稳定后视频可分别独立计算，无需互相对比
"""

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 单帧指标计算
# ============================================================================

def compute_psnr(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    计算相邻两帧之间的峰值信噪比（PSNR）。
    帧间 PSNR 越高 → 相邻帧越相似 → 画面越平滑 → 抖动越小。
    当两帧完全相同（MSE=0）时，返回 100.0 dB 作为上限值。

    参数:
        frame1: 第 i   帧（BGR，uint8）
        frame2: 第 i+1 帧（BGR，uint8）
    返回:
        psnr_value: PSNR 值（dB），完全相同时返回 100.0
    """
    mse = np.mean((frame1.astype(np.float64) - frame2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0  # 两帧完全相同，避免除以零
    return psnr(frame1, frame2, data_range=255)


def compute_ssim(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    计算相邻两帧之间的结构相似性（SSIM）。
    帧间 SSIM 越高 → 相邻帧结构越一致 → 画面越平滑 → 抖动越小。

    参数:
        frame1: 第 i   帧（BGR，uint8），转灰度后计算
        frame2: 第 i+1 帧（BGR，uint8），转灰度后计算
    返回:
        ssim_value: SSIM 值，范围 [-1, 1]
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2, data_range=255)


# ============================================================================
# 单视频帧间评估
# ============================================================================

def evaluate_interframe(
    video_path: str,
    label: str = "视频",
    max_frames: int = None,
    verbose: bool = True
) -> dict:
    """
    对单个视频逐对相邻帧计算 SSIM 和 PSNR，汇总统计信息。

    计算方式：
        对第 i 帧与第 i+1 帧计算 SSIM / PSNR，遍历全部相邻帧对后取均值。
        数值越高表示帧间越平滑，即抖动越小、稳定效果越好。

    参数:
        video_path: 视频文件路径
        label:      视频标签（用于打印和绘图）
        max_frames: 最多读取帧数（None = 全部）
        verbose:    是否打印逐帧进度

    返回:
        result 字典，包含：
            label, psnr_list, ssim_list,
            psnr_mean/min/max/std,
            ssim_mean/min/max/std,
            n_frames, fps, resolution
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames is not None:
        total = min(total, max_frames)

    print(f"\n[{label}]  路径: {video_path}")
    print(f"  分辨率: {w}x{h}  |  FPS: {fps}  |  评估帧数: {total} 帧")
    print(f"  计算方式: 相邻帧对 SSIM / PSNR（共 {total - 1} 对）")
    print("  " + "-" * 46)

    psnr_list = []
    ssim_list = []

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("无法读取第一帧")

    for i in range(1, total):
        ret, curr_frame = cap.read()
        if not ret:
            break

        p = compute_psnr(prev_frame, curr_frame)
        s = compute_ssim(prev_frame, curr_frame)

        psnr_list.append(p)
        ssim_list.append(s)

        if verbose and i % 50 == 0:
            print(f"  已处理 {i}/{total - 1} 帧对  |  "
                  f"PSNR={p:.2f} dB  SSIM={s:.4f}")

        prev_frame = curr_frame

    cap.release()

    psnr_arr = np.array(psnr_list)
    ssim_arr = np.array(ssim_list)

    return {
        'label':      label,
        'psnr_list':  psnr_list,
        'ssim_list':  ssim_list,
        'psnr_mean':  float(np.mean(psnr_arr)),
        'psnr_min':   float(np.min(psnr_arr)),
        'psnr_max':   float(np.max(psnr_arr)),
        'psnr_std':   float(np.std(psnr_arr)),
        'ssim_mean':  float(np.mean(ssim_arr)),
        'ssim_min':   float(np.min(ssim_arr)),
        'ssim_max':   float(np.max(ssim_arr)),
        'ssim_std':   float(np.std(ssim_arr)),
        'n_frames':   len(psnr_list),
        'fps':        fps,
        'resolution': (w, h)
    }


# ============================================================================
# 打印对比摘要
# ============================================================================

def print_comparison(result_orig: dict, result_stab: dict):
    """
    打印原视频与稳定后视频的帧间 SSIM / PSNR 均值对比。

    参数:
        result_orig: evaluate_interframe 对原视频的返回结果
        result_stab: evaluate_interframe 对稳定视频的返回结果
    """
    print("\n" + "=" * 50)
    print("         帧间平滑性评估结果")
    print("   （数值越高 = 帧间越平滑 = 抖动越小）")
    print("=" * 50)
    print(f"  原视频：")
    print(f"    SSIM(均值) = {result_orig['ssim_mean']:.4f}")
    print(f"    PSNR(均值) = {result_orig['psnr_mean']:.2f} dB")
    print(f"  稳定后：")
    print(f"    SSIM(均值) = {result_stab['ssim_mean']:.4f}")
    print(f"    PSNR(均值) = {result_stab['psnr_mean']:.2f} dB")
    print("=" * 50)


# ============================================================================
# 可视化
# ============================================================================

def plot_comparison(result_orig: dict, result_stab: dict, save_path: str = None):
    """
    绘制原视频与稳定后视频的帧间 PSNR / SSIM 逐帧对比曲线。

    参数:
        result_orig: 原视频评估结果
        result_stab: 稳定视频评估结果
        save_path:   图像保存路径（None 则仅显示）
    """
    n = min(result_orig['n_frames'], result_stab['n_frames'])
    frames = list(range(1, n + 1))

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("视频稳定质量评估：帧间 PSNR & SSIM 对比\n"
                 "（相邻帧对计算，数值越高=画面越平滑=抖动越小）",
                 fontsize=13, fontweight='bold')

    colors = {'orig': '#5B9BD5', 'stab': '#ED7D31'}

    # ── PSNR ──
    ax1 = axes[0]
    ax1.plot(frames, result_orig['psnr_list'][:n],
             color=colors['orig'], linewidth=0.8, alpha=0.8,
             label=f"原视频  均值={result_orig['psnr_mean']:.2f} dB")
    ax1.plot(frames, result_stab['psnr_list'][:n],
             color=colors['stab'], linewidth=0.8, alpha=0.8,
             label=f"稳定后  均值={result_stab['psnr_mean']:.2f} dB")
    ax1.axhline(result_orig['psnr_mean'], color=colors['orig'],
                linestyle='--', linewidth=1.2, alpha=0.6)
    ax1.axhline(result_stab['psnr_mean'], color=colors['stab'],
                linestyle='--', linewidth=1.2, alpha=0.6)
    ax1.set_ylabel("PSNR (dB)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("峰值信噪比（帧间）", fontsize=10)

    # ── SSIM ──
    ax2 = axes[1]
    ax2.plot(frames, result_orig['ssim_list'][:n],
             color=colors['orig'], linewidth=0.8, alpha=0.8,
             label=f"原视频  均值={result_orig['ssim_mean']:.4f}")
    ax2.plot(frames, result_stab['ssim_list'][:n],
             color=colors['stab'], linewidth=0.8, alpha=0.8,
             label=f"稳定后  均值={result_stab['ssim_mean']:.4f}")
    ax2.axhline(result_orig['ssim_mean'], color=colors['orig'],
                linestyle='--', linewidth=1.2, alpha=0.6)
    ax2.axhline(result_stab['ssim_mean'], color=colors['stab'],
                linestyle='--', linewidth=1.2, alpha=0.6)
    ax2.set_ylabel("SSIM", fontsize=11)
    ax2.set_xlabel("帧编号", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("结构相似性（帧间）", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  📊 评估曲线已保存: {save_path}")

    plt.show()


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="视频稳定质量评估（帧间 SSIM + PSNR）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--original',   type=str, default='./11.mp4',
                        help='原始（未稳定）视频路径')
    parser.add_argument('--stabilized', type=str, default='./result11.mp4',
                        help='稳定后视频路径')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='最多评估帧数（None=全部）')
    parser.add_argument('--save_plot',  type=str, default='./quality_metrics.png',
                        help='评估曲线保存路径（填 None 则不保存）')
    parser.add_argument('--no_verbose', action='store_true',
                        help='关闭逐帧进度打印')
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.no_verbose

    print("=" * 62)
    print("  视频稳定质量评估（帧间 SSIM + PSNR）")
    print("  评估逻辑：对每个视频独立计算相邻帧对的指标")
    print("  数值越高 = 帧间越平滑 = 抖动越小 = 稳定效果越好")
    print("=" * 62)

    # 分别对原视频和稳定视频独立评估
    result_orig = evaluate_interframe(
        args.original,
        label="原视频",
        max_frames=args.max_frames,
        verbose=verbose
    )
    result_stab = evaluate_interframe(
        args.stabilized,
        label="稳定后视频",
        max_frames=args.max_frames,
        verbose=verbose
    )

    # 打印对比摘要
    print_comparison(result_orig, result_stab)

    # 绘图
    save_path = None if args.save_plot == 'None' else args.save_plot
    plot_comparison(result_orig, result_stab, save_path=save_path)

    return 0


if __name__ == '__main__':
    exit(main())
