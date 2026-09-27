"""
train.py
包含：
  - 归一化工具
  - 数据加载 / 增强 / 数据集构建
  - 训练主函数 train()
  - 命令行入口
"""

import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from model import MultiModelEnsembleNet, SmoothnessLoss, save_model

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 归一化工具
# ============================================================

def compute_norm_params(traj: np.ndarray):
    """计算均值和标准差"""
    mean = traj.mean(axis=0)
    std  = traj.std(axis=0) + 1e-8
    return mean, std

def normalize(traj: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (traj - mean) / std

def denormalize(traj: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return traj * std + mean


# ============================================================
# 2. 数据加载
# ============================================================

def load_paired_trajectories(folder_path: str):
    """
    加载文件夹下所有配对的 *_noisy.npy 和 *_stable.npy 轨迹文件。
    长度不一致时自动裁剪为较短者。
    """
    noisy_files  = sorted(glob.glob(os.path.join(folder_path, "*_noisy.npy")))
    stable_files = sorted(glob.glob(os.path.join(folder_path, "*_stable.npy")))

    if len(noisy_files) == 0:
        raise FileNotFoundError(f"未找到轨迹文件，请检查路径: {folder_path}")
    if len(noisy_files) != len(stable_files):
        raise ValueError(f"noisy({len(noisy_files)}) 与 stable({len(stable_files)}) 数量不一致！")

    all_noisy, all_stable = [], []
    for nf, sf in zip(noisy_files, stable_files):
        noisy  = np.load(nf).astype(np.float32)
        stable = np.load(sf).astype(np.float32)
        min_len = min(len(noisy), len(stable))
        if len(noisy) != len(stable):
            print(f"  ⚠ 长度不一致，裁剪为 {min_len} 帧: {os.path.basename(nf)}")
        all_noisy.append(noisy[:min_len])
        all_stable.append(stable[:min_len])

    combined_noisy  = np.concatenate(all_noisy,  axis=0)
    combined_stable = np.concatenate(all_stable, axis=0)
    print(f"✅ 加载 {len(noisy_files)} 对轨迹，总帧数: {combined_noisy.shape[0]}")
    return combined_noisy, combined_stable


# ============================================================
# 3. 数据增强
# ============================================================

def augment_pair(noisy_win: np.ndarray, stable_win: np.ndarray):
    """
    对单个窗口对做数据增强，返回增强后的样本列表：
      (a) 原始
      (b) 时间翻转
      (c) noisy 加高斯噪声（5% 标准差）
      (d) 随机幅度缩放 [0.85, 1.15]
    """
    samples = []

    # (a) 原始
    samples.append((noisy_win.copy(), stable_win.copy()))

    # (b) 时间翻转
    samples.append((noisy_win[::-1].copy(), stable_win[::-1].copy()))

    # (c) 加高斯噪声
    noise_scale  = np.std(noisy_win, axis=0) * 0.05
    noisy_noised = noisy_win + np.random.randn(*noisy_win.shape) * noise_scale
    samples.append((noisy_noised.astype(np.float32), stable_win.copy()))

    # (d) 随机缩放
    scale = np.random.uniform(0.85, 1.15)
    samples.append(((noisy_win * scale).astype(np.float32),
                    (stable_win * scale).astype(np.float32)))

    return samples


# ============================================================
# 4. 数据集构建
# ============================================================

def build_dataset(
    noisy_traj:  np.ndarray,
    stable_traj: np.ndarray,
    window_size: int  = 125,
    stride:      int  = 1,
    augment:     bool = True
):
    """
    滑动窗口采样，目标为窗口最后一帧的稳定轨迹点（单帧预测）。
    stride=1 最大化样本数量。
    augment=True 启用数据增强（约 4x 样本量）。
    """
    inputs, targets = [], []
    n = len(noisy_traj)

    for i in range(0, n - window_size, stride):
        noisy_win  = noisy_traj [i : i + window_size]
        stable_win = stable_traj[i : i + window_size]

        pairs = augment_pair(noisy_win, stable_win) if augment else [(noisy_win, stable_win)]

        for nw, sw in pairs:
            inputs.append(nw)
            targets.append(sw[-1])   # 预测窗口最后一帧的稳定值

    inputs  = np.array(inputs,  dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    print(f"✅ 数据集构建完成: {len(inputs)} 个样本"
          f"（增强={'开启' if augment else '关闭'}，stride={stride}）")
    return inputs, targets


# ============================================================
# 5. PyTorch Dataset
# ============================================================

class TrajectoryDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs  = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ============================================================
# 6. 训练主函数
# ============================================================

def train(
    folder:      str   = "../trajectories",
    window_size: int   = 125,
    epochs:      int   = 500,
    batch_size:  int   = 32,
    lr:          float = 1e-4,
    save_dir:    str   = ".",
    augment:     bool  = True,
    stride:      int   = 1
):
    """
    训练 MultiModelEnsembleNet。

    归一化策略（修复版）：
      - 统一使用 noisy 轨迹的均值/标准差归一化 noisy 和 stable
      - 将 noisy_mean / noisy_std 保存到 save_dir，供 test.py 推理使用
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- 加载数据 ---
    noisy_raw, stable_raw = load_paired_trajectories(folder)

    # ✅ 修复：统一用 noisy 的统计量，stable 也用同一套参数归一化
    noisy_mean, noisy_std = compute_norm_params(noisy_raw)
    np.save(os.path.join(save_dir, "noisy_mean.npy"), noisy_mean)
    np.save(os.path.join(save_dir, "noisy_std.npy"),  noisy_std)
    print(f"归一化参数已保存 → mean={noisy_mean.round(4)}, std={noisy_std.round(4)}")

    noisy_norm  = normalize(noisy_raw,  noisy_mean, noisy_std)
    stable_norm = normalize(stable_raw, noisy_mean, noisy_std)  # ← 同一套参数

    # --- 构建数据集 ---
    inputs, targets = build_dataset(
        noisy_norm, stable_norm,
        window_size = window_size,
        stride      = stride,
        augment     = augment
    )

    dataset    = TrajectoryDataset(inputs, targets)
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )

    # --- 模型 / 优化器 ---
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = MultiModelEnsembleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SmoothnessLoss(lambda_smooth=0.1)

    print(f"\n开始训练：设备={device}  epochs={epochs}  batch={batch_size}  样本数={len(dataset)}")
    print("-" * 60)

    all_loss = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)                    # (B, 3)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        all_loss.append(total_loss)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}  "
                  f"Loss: {total_loss:.6f}  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    # --- 保存模型 ---
    model_path = os.path.join(save_dir, "multi_model_ensemble.pth")
    save_model(model, model_path)

    # --- 绘制 Loss 曲线 ---
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(all_loss) + 1), all_loss, linewidth=1.2)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("训练 Loss 曲线（MSE + 平滑损失）")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    loss_fig = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_fig, dpi=150)
    print(f"Loss 曲线已保存: {loss_fig}")
    plt.show()

    return model


# ============================================================
# 7. 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="轨迹平滑模型训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--folder",      default="../trajectories", help="轨迹文件夹路径")
    parser.add_argument("--window_size", type=int,   default=125,  help="滑动窗口大小")
    parser.add_argument("--epochs",      type=int,   default=500,  help="训练轮数")
    parser.add_argument("--batch_size",  type=int,   default=32,   help="批大小")
    parser.add_argument("--lr",          type=float, default=1e-4, help="学习率")
    parser.add_argument("--save_dir",    default=".",             help="模型/参数保存目录")
    parser.add_argument("--no_augment",  action="store_true",      help="关闭数据增强")
    parser.add_argument("--stride",      type=int,   default=1,    help="滑动窗口步长")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        folder      = args.folder,
        window_size = args.window_size,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        save_dir    = args.save_dir,
        augment     = not args.no_augment,
        stride      = args.stride
    )