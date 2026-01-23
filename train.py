import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import glob

from Graph.enhancedEdition.model import MultiModelEnsembleNet

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 归一化 ----------
def normalize_trajectory(traj):
    mean = traj.mean(axis=0)
    std = traj.std(axis=0) + 1e-8  # 防止除0
    return (traj - mean) / std, mean, std

def denormalize_trajectory(traj, mean, std):
    return traj * std + mean

def load_paired_trajectories(folder_path):
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

# --- Synthetic Training Data Generation ---
def generate_from_paired_trajectories(noisy_traj, stable_traj, window_size=125, output_whole_seq=False):
    inputs, targets = [], []

    for i in range(len(noisy_traj) - window_size):
        noisy_window = noisy_traj[i:i + window_size]
        stable_window = stable_traj[i:i + window_size]

        inputs.append(noisy_window)
        if output_whole_seq:
            targets.append(stable_window)
        else:
            targets.append(stable_window[-1])

    return np.array(inputs), np.array(targets)

# --- Custom Dataset ---
class TrajectoryDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# --- Training and Testing Functions ---
def train_multi_model_ensemble():
    folder = "./trajectories"
    noisy_traj, stable_traj = load_paired_trajectories(folder)

    noisy_traj, noisy_mean, noisy_std = normalize_trajectory(noisy_traj)
    stable_traj, stable_mean, stable_std = normalize_trajectory(stable_traj)

    np.save("stable_mean.npy", stable_mean)
    np.save("stable_std.npy", stable_std)

    inputs, targets = generate_from_paired_trajectories(noisy_traj, stable_traj)
    dataset = TrajectoryDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModelEnsembleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # 学习率衰减
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()

    all_loss = []

    for epoch in range(500):
        total_loss = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        all_loss.append(total_loss)
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    plt.figure()
    plt.plot(range(1, len(all_loss)+1), all_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('融合神经网络训练 Loss 曲线')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()

    torch.save(model.state_dict(), "multi_model_ensemble.pth")
    print("\n 模型训练完成，已保存为 multi_model_ensemble.pth")

    return model, noisy_traj, stable_traj

def test_and_visualize(model, noisy_traj, stable_traj, window_size=125):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    mean = np.load("stable_mean.npy")
    std = np.load("stable_std.npy")

    inputs, targets = generate_from_paired_trajectories(noisy_traj, stable_traj, window_size=window_size)
    all_preds = []

    for i in range(len(inputs)):
        sample_input = torch.tensor(inputs[i], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(sample_input).squeeze(0).cpu().numpy()

        all_preds.append(pred)

    all_preds = np.array(all_preds)

    all_preds = denormalize_trajectory(all_preds, mean, std)
    stable_traj = denormalize_trajectory(stable_traj, mean, std)
    noisy_traj = denormalize_trajectory(noisy_traj, mean, std)

    frames = list(range(len(noisy_traj)))
    labels = ['dx', 'dy', 'da']

    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(frames, noisy_traj[:, i], label='Noisy Input', alpha=0.6)
        plt.plot(frames, stable_traj[:, i], color='green', linestyle='--', label='Stable (GT)')
        plt.plot(frames[window_size:], all_preds[:, i], color='red', linestyle='-', label='Predicted')
        plt.title(labels[i])
        plt.legend()

    plt.suptitle("轨迹预测对比（不稳定 vs 预测 vs 稳定）")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model, noisy_traj, stable_traj = train_multi_model_ensemble()
    test_and_visualize(model, noisy_traj[:500], stable_traj[:500])
