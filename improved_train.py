"""
改进的训练脚本
===============
改进点:
1. 多种改进的模型选择
2. 综合损失函数 (MSE + 平滑性)
3. 数据增强 (噪声、翻转、缩放)
4. 更好的学习率调度
5. 早停机制
6. 混合精度训练
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import glob
import random
from improved_model import (
    ImprovedVideoStabilizer,
    UncertaintyVideoStabilizer,
    HybridVideoStabilizer,
    TrajectoryLoss,
    NLLWithVarianceLoss,
    create_improved_model
)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 归一化 ==========
def normalize_trajectory(traj):
    mean = traj.mean(axis=0)
    std = traj.std(axis=0) + 1e-8
    return (traj - mean) / std, mean, std


def denormalize_trajectory(traj, mean, std):
    return traj * std + mean


# ========== 数据增强 ==========
class TrajectoryAugmentation:
    """轨迹数据增强"""
    
    @staticmethod
    def add_noise(traj, noise_level=0.02):
        """添加高斯噪声"""
        noise = np.random.randn(*traj.shape) * noise_level
        return traj + noise
    
    @staticmethod
    def scale(traj, scale_factor=(0.95, 1.05)):
        """随机缩放"""
        factor = random.uniform(*scale_factor)
        return traj * factor
    
    @staticmethod
    def flip(traj):
        """随机翻转"""
        if random.random() < 0.5:
            traj = -traj
        return traj
    
    @staticmethod
    def time_warp(traj, sigma=0.2):
        """时间扭曲 - 打乱帧顺序的局部区域"""
        if random.random() < 0.3:
            traj = traj.copy()
            # 随机选择一段进行反向
            start = random.randint(0, len(traj) - 10)
            end = random.randint(start + 5, min(start + 20, len(traj)))
            traj[start:end] = traj[start:end][::-1]
        return traj
    
    @staticmethod
    def augment(traj):
        """组合增强"""
        traj = TrajectoryAugmentation.add_noise(traj)
        traj = TrajectoryAugmentation.scale(traj)
        traj = TrajectoryAugmentation.flip(traj)
        traj = TrajectoryAugmentation.time_warp(traj)
        return traj


# ========== 数据加载 ==========
def load_paired_trajectories(folder_path):
    """加载成对的轨迹数据"""
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


def generate_training_data(noisy_traj, stable_traj, window_size=125, stride=10):
    """
    生成训练数据
    Args:
        noisy_traj: [N, 3]
        stable_traj: [N, 3]
        window_size: 滑动窗口大小
        stride: 滑动步长 (减小冗余)
    """
    inputs, targets = [], []

    for i in range(0, len(noisy_traj) - window_size, stride):
        noisy_window = noisy_traj[i:i + window_size]
        stable_window = stable_traj[i:i + window_size]
        
        # 计算修正量: stable - noisy
        correction = stable_window - noisy_window
        inputs.append(noisy_window)
        targets.append(correction)  # 预测修正量而非直接预测稳定轨迹

    return np.array(inputs), np.array(targets)


class PairedDataset(Dataset):
    """成对轨迹数据集"""
    def __init__(self, inputs, targets, use_augmentation=True):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.use_augmentation = use_augmentation
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = self.inputs[idx].numpy()
        y = self.targets[idx].numpy()
        
        if self.use_augmentation:
            x = TrajectoryAugmentation.augment(x)
        
        return torch.tensor(x), torch.tensor(y)


# ========== 训练函数 ==========
def train_model(
    model_type='hybrid',
    window_size=125,
    hidden_dim=96,
    batch_size=32,
    epochs=500,
    lr=1e-4,
    use_augmentation=True,
    smooth_weight=0.5,
    patience=30,
    save_path='improved_model.pth'
):
    """
    训练改进的模型
    
    Args:
        model_type: 模型类型 ('transformer', 'uncertainty', 'hybrid')
        window_size: 轨迹窗口大小
        hidden_dim: 隐藏层维度
        batch_size: 批大小
        epochs: 训练轮数
        lr: 学习率
        use_augmentation: 是否使用数据增强
        smooth_weight: 平滑损失权重
        patience: 早停耐心值
        save_path: 模型保存路径
    """
    # 数据准备
    folder = "./trajectories"
    noisy_traj, stable_traj = load_paired_trajectories(folder)
    
    # 归一化
    noisy_traj, noisy_mean, noisy_std = normalize_trajectory(noisy_traj)
    stable_traj, stable_mean, stable_std = normalize_trajectory(stable_traj)
    
    # 保存归一化参数
    np.save("improved_mean.npy", stable_mean)
    np.save("improved_std.npy", stable_std)
    
    # 生成训练数据
    inputs, targets = generate_training_data(noisy_traj, stable_traj, window_size, stride=10)
    print(f"生成训练样本: {len(inputs)} 个")
    
    # 划分训练集和验证集
    val_ratio = 0.1
    val_size = int(len(inputs) * val_ratio)
    indices = torch.randperm(len(inputs))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_dataset = PairedDataset(
        inputs[train_indices], 
        targets[train_indices], 
        use_augmentation=use_augmentation
    )
    val_dataset = PairedDataset(inputs[val_indices], targets[val_indices], use_augmentation=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if model_type == 'uncertainty':
        model = UncertaintyVideoStabilizer(input_dim=3, hidden_dim=hidden_dim)
        criterion = NLLWithVarianceLoss()
    else:
        model = create_improved_model(
            model_type=model_type,
            input_dim=3,
            window_size=window_size,
            hidden_dim=hidden_dim
        )
        criterion = TrajectoryLoss(smooth_weight=smooth_weight)
    
    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度 (余弦退火)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )
    
    # 早停
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 训练循环
    train_losses = []
    val_losses = []
    
    print(f"\n开始训练 ({epochs} epochs)...")
    
    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_smooth = 0.0
        
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            if model_type == 'uncertainty':
                mean, logvar = model(x)
                loss = criterion(mean, logvar, y)
            else:
                pred, smooth_pred = model(x) if hasattr(model, 'smooth_head') else (model(x), None)
                loss, mse, smooth = criterion(pred, y, trajectory=x)
                train_mse += mse.item()
                train_smooth += smooth
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        train_smooth /= len(train_loader)
        train_losses.append(train_loss)
        
        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                
                if model_type == 'uncertainty':
                    mean, logvar = model(x)
                    loss = criterion(mean, logvar, y)
                else:
                    pred, _ = model(x) if hasattr(model, 'smooth_head') else (model(x), None)
                    loss, _, _ = criterion(pred, y)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 学习率更新
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if model_type == 'uncertainty':
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} (mse:{train_mse:.4f}, sm:{train_smooth:.4f}) | Val: {val_loss:.4f} | LR: {current_lr:.2e}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发！最佳验证损失: {best_val_loss:.4f}")
                break
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(train_losses, label='Train Loss', alpha=0.7)
    plt.semilogy(val_losses, label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.title('训练损失曲线 (对数)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("improved_loss_curve.png", dpi=150)
    plt.show()
    
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")
    print(f"模型已保存至: {save_path}")
    
    return model, train_losses, val_losses


def test_and_visualize(model_path='improved_model.pth', model_type='hybrid', window_size=125):
    """测试和可视化"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载归一化参数
    mean = np.load("improved_mean.npy")
    std = np.load("improved_std.npy")
    
    # 加载模型
    if model_type == 'uncertainty':
        model = UncertaintyVideoStabilizer(input_dim=3, hidden_dim=96)
    else:
        model = create_improved_model(model_type, input_dim=3, window_size=window_size, hidden_dim=96)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    
    # 加载测试数据
    folder = "./trajectories"
    noisy_files = sorted(glob.glob(os.path.join(folder, "*_noisy.npy")))
    stable_files = sorted(glob.glob(os.path.join(folder, "*_stable.npy")))
    
    # 取一个样本测试
    noisy = np.load(noisy_files[0])
    stable = np.load(stable_files[0])
    min_len = min(len(noisy), len(stable))
    noisy = noisy[:min_len]
    stable = stable[:min_len]
    
    # 归一化
    noisy_norm = (noisy - mean) / std
    stable_norm = (stable - mean) / std
    
    # 生成测试窗口
    stride = 10
    predictions = []
    
    for i in range(0, len(noisy_norm) - window_size, stride):
        window = noisy_norm[i:i + window_size]
        window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if model_type == 'uncertainty':
                pred, _ = model(window_tensor)
            else:
                pred = model(window_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]
            predictions.append(pred.squeeze(0).cpu().numpy())
    
    predictions = np.array(predictions)
    
    # 反归一化修正量
    predictions = predictions * std + mean
    stable = stable * std + mean
    noisy = noisy * std + mean
    
    # 计算预测的稳定轨迹
    predicted_stable = []
    for i, pred in enumerate(predictions):
        start_idx = i * stride
        end_idx = start_idx + window_size
        noisy_window = noisy[start_idx:end_idx]
        corrected = noisy_window + pred  # 应用修正量
        predicted_stable.append(corrected[-1])
    
    predicted_stable = np.array(predicted_stable)
    
    # 可视化
    frames = list(range(len(noisy)))
    labels = ['dx', 'dy', 'da']
    
    plt.figure(figsize=(14, 8))
    
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(frames, noisy[:, i], label='Noisy Input', alpha=0.6, linewidth=1)
        plt.plot(frames, stable[:, i], color='green', linestyle='--', label='Ground Truth', linewidth=1.5)
        pred_frames = list(range(window_size//2, len(predicted_stable)*stride + window_size//2, stride))
        plt.plot(pred_frames, predicted_stable[:, i], color='red', linestyle='-', label='Predicted', linewidth=1.5)
        plt.title(labels[i])
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle("改进模型轨迹预测对比")
    plt.tight_layout()
    plt.savefig("improved_comparison.png", dpi=150)
    plt.show()
    
    # 计算误差
    mse_noisy = np.mean((noisy - stable) ** 2)
    mse_pred = np.mean((predicted_stable - stable[window_size//2:len(predicted_stable)*stride + window_size//2]) ** 2)
    
    print(f"\n测试结果:")
    print(f"  原始轨迹 MSE: {mse_noisy:.4f}")
    print(f"  预测轨迹 MSE: {mse_pred:.4f}")
    print(f"  改善比例: {(mse_noisy - mse_pred) / mse_noisy * 100:.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="改进的视频稳定模型训练")
    parser.add_argument('--model', type=str, default='hybrid', 
                       choices=['transformer', 'uncertainty', 'hybrid'],
                       help='模型类型')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--window', type=int, default=125, help='窗口大小')
    parser.add_argument('--aug', action='store_true', help='使用数据增强')
    parser.add_argument('--smooth', type=float, default=0.5, help='平滑损失权重')
    parser.add_argument('--test', action='store_true', help='仅运行测试')
    
    args = parser.parse_args()
    
    if args.test:
        test_and_visualize(model_type=args.model, window_size=args.window)
    else:
        train_model(
            model_type=args.model,
            window_size=args.window,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            use_augmentation=args.aug,
            smooth_weight=args.smooth,
            save_path=f'{args.model}_model.pth'
        )

