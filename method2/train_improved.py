"""
改进的稳像网络训练脚本
核心改进：
1. 训练相对运动（帧间变换）的平滑
2. 使用更小的窗口大小（30帧 vs 125帧）
3. 输出整个序列的平滑结果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import glob

from improved_model import ImprovedStabilizerNet, RelativeMotionDataset

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ---------- 归一化 ----------
def normalize_trajectory(traj):
    """轨迹归一化"""
    mean = traj.mean(axis=0)
    std = traj.std(axis=0) + 1e-8
    return (traj - mean) / std, mean, std


def denormalize_trajectory(traj, mean, std):
    """轨迹反归一化"""
    return traj * std + mean


def load_paired_trajectories(folder_path):
    """加载配对轨迹"""
    noisy_files = sorted(glob.glob(os.path.join(folder_path, "*_noisy.npy")))
    stable_files = sorted(glob.glob(os.path.join(folder_path, "*_stable.npy")))

    if len(noisy_files) != len(stable_files):
        raise ValueError("不稳定轨迹和稳定轨迹数量不一致！")

    all_noisy, all_stable = [], []

    for noisy_file, stable_file in zip(noisy_files, stable_files):
        noisy = np.load(noisy_file)
        stable = np.load(stable_file)

        min_len = min(noisy.shape[0], stable.shape[0])
        if noisy.shape[0] != stable.shape[0]:
            print(f"形状不一致：{os.path.basename(noisy_file)} - noisy: {noisy.shape}, stable: {stable.shape}，裁剪为 {min_len} 帧")
            noisy = noisy[:min_len]
            stable = stable[:min_len]

        all_noisy.append(noisy)
        all_stable.append(stable)

    combined_noisy = np.concatenate(all_noisy, axis=0)
    combined_stable = np.concatenate(all_stable, axis=0)
    print(f"加载 {len(noisy_files)} 对轨迹，总长度：{combined_noisy.shape[0]} 帧")

    return combined_noisy, combined_stable


def generate_relative_motion_data(noisy_traj, stable_traj, window_size=30):
    """
    生成相对运动训练数据

    核心改进：
    - 输入：连续N帧的相对运动（帧间变换）
    - 输出：平滑后的N帧相对运动
    """
    # 转换为相对运动
    noisy_rel = RelativeMotionDataset.cumulative_to_relative(noisy_traj)
    stable_rel = RelativeMotionDataset.cumulative_to_relative(stable_traj)

    inputs, targets = [], []

    for i in range(len(noisy_rel) - window_size + 1):
        noisy_window = noisy_rel[i:i + window_size]
        stable_window = stable_rel[i:i + window_size]

        inputs.append(noisy_window)
        targets.append(stable_window)

    return np.array(inputs), np.array(targets)


class RelativeMotionDataset(Dataset):
    """相对运动数据集"""
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train_improved_model():
    """训练改进的稳像网络"""
    # 配置
    WINDOW_SIZE = 30  # 减小窗口大小，加速训练
    BATCH_SIZE = 64
    EPOCHS = 300
    LR = 1e-4
    HIDDEN_DIM = 64
    NUM_LAYERS = 2

    # 加载数据
    folder = "../trajectories"
    noisy_traj, stable_traj = load_paired_trajectories(folder)

    # 转换为相对运动并归一化
    noisy_rel = RelativeMotionDataset.cumulative_to_relative(noisy_traj)
    stable_rel = RelativeMotionDataset.cumulative_to_relative(stable_traj)

    # 归一化
    noisy_rel, noisy_mean, noisy_std = normalize_trajectory(noisy_rel)
    stable_rel, stable_mean, stable_std = normalize_trajectory(stable_rel)

    # 保存归一化参数
    np.save("stable_rel_mean.npy", stable_mean)
    np.save("stable_rel_std.npy", stable_std)

    # 生成训练数据
    inputs, targets = generate_relative_motion_data(noisy_rel, stable_rel, window_size=WINDOW_SIZE)

    print(f"训练数据: 输入 {inputs.shape}, 目标 {targets.shape}")

    # 创建数据集和数据加载器
    dataset = RelativeMotionDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    model = ImprovedStabilizerNet(
        window_size=WINDOW_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 损失函数
    criterion = nn.MSELoss()

    # 记录
    all_loss = []

    print(f"\n开始训练... (窗口={WINDOW_SIZE}, 批次={BATCH_SIZE}, 轮次={EPOCHS})")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            # 前向传播
            pred = model(x)  # [B, T, 3]

            # 损失：MSE + 平滑正则（鼓励输出变化平滑）
            loss = criterion(pred, y)

            # 可选：添加时间平滑正则
            if epoch > 50:  # 后期加入平滑约束
                temp_smooth = F.mse_loss(pred[:, 1:], pred[:, :-1])
                loss = loss + 0.01 * temp_smooth

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        scheduler.step()
        avg_loss = total_loss / batch_count
        all_loss.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{EPOCHS}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # 保存模型
    torch.save(model.state_dict(), "improved_stabilizer.pth")
    print(f"\n模型训练完成，已保存为 improved_stabilizer.pth")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(all_loss) + 1), all_loss, 'b-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('改进网络训练损失曲线')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("improved_loss_curve.png", dpi=150)
    print("损失曲线已保存: improved_loss_curve.png")

    return model, all_loss


def test_and_visualize(model=None):
    """测试和可视化"""
    import improved_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WINDOW_SIZE = 30

    # 加载模型
    if model is None:
        model = ImprovedStabilizerNet(window_size=WINDOW_SIZE)
        model.load_state_dict(torch.load("improved_stabilizer.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # 加载数据
    folder = "./trajectories"
    noisy_traj, stable_traj = load_paired_trajectories(folder)

    # 转换为相对运动
    noisy_rel = RelativeMotionDataset.cumulative_to_relative(noisy_traj)
    stable_rel = RelativeMotionDataset.cumulative_to_relative(stable_traj)

    # 归一化
    noisy_rel, _, _ = normalize_trajectory(noisy_rel)
    stable_rel, stable_mean, stable_std = normalize_trajectory(stable_rel)

    # 生成测试数据
    inputs, targets = generate_relative_motion_data(noisy_rel, stable_rel, window_size=WINDOW_SIZE)

    # 预测
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(inputs), 32):
            batch = torch.tensor(inputs[i:i+32], dtype=torch.float32).unsqueeze(0).to(device)
            if batch.shape[1] != WINDOW_SIZE:
                continue
            pred = model(batch).squeeze(0).cpu().numpy()
            all_preds.append(pred)

    all_preds = np.concatenate(all_preds, axis=0)

    # 反归一化
    all_preds = denormalize_trajectory(all_preds, stable_mean, stable_std)
    targets = denormalize_trajectory(targets, stable_mean, stable_std)
    inputs = denormalize_trajectory(inputs, stable_mean, stable_std)

    # 转换为累积轨迹便于可视化
    pred_cumsum = RelativeMotionDataset.relative_to_cumulative(all_preds)
    target_cumsum = RelativeMotionDataset.relative_to_cumulative(targets)
    input_cumsum = RelativeMotionDataset.relative_to_cumulative(inputs)

    # 可视化
    frames = list(range(len(target_cumsum)))
    labels = ['dx', 'dy', 'da']

    plt.figure(figsize=(14, 10))

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(frames, input_cumsum[:, i], 'r-', alpha=0.5, label='Noisy Input', linewidth=1)
        plt.plot(frames, target_cumsum[:, i], 'g--', alpha=0.7, label='Stable (GT)', linewidth=1.5)
        plt.plot(frames, pred_cumsum[:, i], 'b-', alpha=0.7, label='Predicted', linewidth=1.5)
        plt.title(f'{labels[i]} - 轨迹对比')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("improved_comparison.png", dpi=150)
    print("对比图已保存: improved_comparison.png")

    # 计算指标
    mse = np.mean((all_preds - targets) ** 2)
    print(f"\n测试MSE: {mse:.6f}")

    return all_preds, targets


if __name__ == "__main__":
    model, loss = train_improved_model()
    test_and_visualize(model)
