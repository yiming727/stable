"""
model.py
包含：
  - MultiModelEnsembleNet 模型结构
  - SmoothnessLoss 平滑损失函数
  - 模型保存 / 加载工具函数
"""

import os
import torch
import torch.nn as nn


# ============================================================
# 子模块：LSTM 分支
# ============================================================

class LSTMBranch(nn.Module):
    """
    单向 LSTM，捕捉轨迹的时序依赖关系。
    输入: (B, T, 3)
    输出: (B, hidden_size)
    """
    def __init__(self, input_size: int = 3, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 取最后一个时间步的隐状态
        out, _ = self.lstm(x)          # (B, T, hidden)
        return out[:, -1, :]           # (B, hidden)


# ============================================================
# 子模块：Transformer 分支
# ============================================================

class TransformerBranch(nn.Module):
    """
    Transformer Encoder，捕捉全局帧间注意力关系。
    输入: (B, T, 3)
    输出: (B, d_model)
    """
    def __init__(self, input_size: int = 3, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model    = d_model,
            nhead      = nhead,
            dim_feedforward = d_model * 4,
            dropout    = 0.1,
            batch_first = True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)   # 全局平均池化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)          # (B, T, d_model)
        x = self.encoder(x)             # (B, T, d_model)
        x = x.transpose(1, 2)          # (B, d_model, T)
        x = self.pool(x).squeeze(-1)   # (B, d_model)
        return x


# ============================================================
# 子模块：1D-CNN 分支
# ============================================================

class CNNBranch(nn.Module):
    """
    1D 卷积，提取局部运动模式特征。
    输入: (B, T, 3)
    输出: (B, out_channels)
    """
    def __init__(self, input_size: int = 3, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (B, 3, T)
        x = self.net(x).squeeze(-1)    # (B, out_channels)
        return x


# ============================================================
# 主模型：多分支融合网络
# ============================================================

class MultiModelEnsembleNet(nn.Module):
    """
    融合 LSTM + Transformer + CNN 三个分支，
    预测输入窗口对应的平滑轨迹（最后一帧的 dx/dy/da）。

    输入:  (B, T, 3)   —— 归一化后的 noisy 轨迹窗口
    输出:  (B, 3)      —— 预测的平滑轨迹点
    """
    def __init__(
        self,
        input_size:   int = 3,
        lstm_hidden:  int = 128,
        tf_d_model:   int = 64,
        cnn_channels: int = 64
    ):
        super().__init__()
        self.lstm_branch = LSTMBranch(input_size, lstm_hidden)
        self.tf_branch   = TransformerBranch(input_size, tf_d_model)
        self.cnn_branch  = CNNBranch(input_size, cnn_channels)

        fusion_in = lstm_hidden + tf_d_model + cnn_channels
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)   # 输出 3 维
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_lstm = self.lstm_branch(x)    # (B, 128)
        f_tf   = self.tf_branch(x)      # (B, 64)
        f_cnn  = self.cnn_branch(x)     # (B, 64)

        fused = torch.cat([f_lstm, f_tf, f_cnn], dim=-1)  # (B, 256)
        return self.fusion(fused)                          # (B, 3)


# ============================================================
# 损失函数：MSE + 平滑惩罚
# ============================================================

class SmoothnessLoss(nn.Module):
    """
    MSE 损失 + 相邻预测帧差分惩罚，鼓励输出平滑。

    当模型输出单帧 (B, 3) 时，退化为纯 MSE。
    当模型输出序列 (B, T, 3) 时，额外加平滑项。

    参数:
        lambda_smooth: 平滑项权重（默认 0.1）
    """
    def __init__(self, lambda_smooth: float = 0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)

        if pred.dim() == 3 and pred.shape[1] > 1:
            diff        = pred[:, 1:, :] - pred[:, :-1, :]
            smooth_loss = (diff ** 2).mean()
            return mse_loss + self.lambda_smooth * smooth_loss

        return mse_loss


# ============================================================
# 工具函数：保存 / 加载模型
# ============================================================

def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ 模型已保存: {path}")


def load_model(path: str, device: torch.device = None) -> MultiModelEnsembleNet:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModelEnsembleNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ 模型已加载: {path}  设备: {device}")
    return model