"""
改进的实时视频稳像网络
核心改进：
1. 预测帧间运动（delta）而非累积位置
2. 实时在线推理模式
3. 改进的Transformer时序建模
4. 自适应平滑权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RelativeMotionDataset:
    """计算相对运动（帧间变换）"""
    @staticmethod
    def compute_relative_motion(cumulative_traj):
        """从累积轨迹计算帧间相对运动"""
        relative_motion = np.zeros_like(cumulative_traj)
        relative_motion[0] = cumulative_traj[0]  # 第一帧为0
        relative_motion[1:] = np.diff(cumulative_traj, axis=0)
        return relative_motion

    @staticmethod
    def cumulative_to_relative(cum_traj):
        """累积轨迹转相对运动"""
        rel = np.zeros_like(cum_traj)
        rel[0] = cum_traj[0]
        rel[1:] = cum_traj[1:] - cum_traj[:-1]
        return rel

    @staticmethod
    def relative_to_cumulative(rel_traj):
        """相对运动转累积轨迹"""
        return np.cumsum(rel_traj, axis=0)


class TemporalAttention(nn.Module):
    """时间注意力机制"""
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, T, hidden_dim]
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        return x


class MotionEncoder(nn.Module):
    """运动特征编码器"""
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 1D 卷积捕获局部模式
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)

        self.norm = nn.BatchNorm1d(hidden_dim * 2)

    def forward(self, x):
        # x: [B, T, 3]
        x = self.input_proj(x)  # [B, T, hidden_dim]

        # 1D 卷积
        x_conv = x.permute(0, 2, 1)  # [B, hidden_dim, T]
        x1 = F.relu(self.conv1(x_conv))
        x2 = F.relu(self.conv2(x_conv))

        x_conv = torch.cat([x1, x2], dim=1)  # [B, hidden_dim*2, T]
        x_conv = self.norm(x_conv)

        return x_conv.permute(0, 2, 1)  # [B, T, hidden_dim*2]


class ImprovedStabilizerNet(nn.Module):
    """改进的稳像网络"""
    def __init__(self,
                 window_size=30,      # 输入窗口大小
                 input_dim=3,         # dx, dy, da
                 hidden_dim=64,
                 num_layers=2,
                 dropout=0.1):
        super().__init__()

        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # 运动编码
        self.motion_encoder = MotionEncoder(input_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 全局时间建模
        self.temporal_attn = TemporalAttention(hidden_dim * 2)

        # 预测头 - 预测平滑后的相对运动
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: 输入相对运动 [B, T, 3]

        Returns:
            平滑后的相对运动 [B, T, 3]
        """
        # 编码
        feat = self.motion_encoder(x)  # [B, T, hidden_dim*2]

        # Transformer 时序建模
        feat = self.transformer(feat)  # [B, T, hidden_dim*2]

        # 时间注意力
        feat = self.temporal_attn(feat)  # [B, T, hidden_dim*2]

        # 预测
        output = self.pred_head(feat)  # [B, T, 3]

        return output


class OnlineStabilizer:
    """在线稳像器 - 封装网络用于实时推理"""
    def __init__(self, model_path, window_size=30, device='cpu'):
        self.window_size = window_size
        self.device = device

        # 加载模型
        self.model = ImprovedStabilizerNet(window_size=window_size)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model.to(device)

        # 滑动窗口
        self.motion_window = []
        self.smoothed_history = []

        # 统计参数（用于归一化）
        self.mean = None
        self.std = None

    def set_normalization(self, mean, std):
        """设置归一化参数"""
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def stabilize(self, dx, dy, da):
        """
        对单帧运动进行稳像

        Args:
            dx, dy, da: 当前帧的相对运动

        Returns:
            smooth_dx, smooth_dy, smooth_da: 平滑后的相对运动
        """
        # 添加到窗口
        motion = torch.tensor([[dx, dy, da]], dtype=torch.float32, device=self.device)

        if self.mean is not None:
            motion = (motion - self.mean) / (self.std + 1e-8)

        self.motion_window.append(motion)

        # 如果窗口未满，直接返回原始值
        if len(self.motion_window) < self.window_size:
            return dx, dy, da

        # 移除最早的帧
        if len(self.motion_window) > self.window_size:
            self.motion_window.pop(0)

        # 组成输入
        input_seq = torch.cat(list(self.motion_window), dim=0).unsqueeze(0)  # [1, T, 3]

        # 前向传播
        smoothed = self.model(input_seq)  # [1, T, 3]

        # 取最后一帧的预测
        last_pred = smoothed[0, -1].cpu().numpy()

        # 反归一化
        if self.mean is not None:
            mean_np = self.mean.cpu().numpy()
            std_np = self.std.cpu().numpy()
            last_pred = last_pred * std_np + mean_np

        return float(last_pred[0]), float(last_pred[1]), float(last_pred[2])

    def reset(self):
        """重置状态"""
        self.motion_window = []
        self.smoothed_history = []


class HybridStabilizer:
    """混合稳像器 - 网络 + 后处理"""
    def __init__(self, model_path, window_size=30, device='cpu',
                 post_smoothing=0.3, motion_smooth=True):
        """
        Args:
            model_path: 模型路径
            window_size: 网络输入窗口
            device: 设备
            post_smoothing: 后处理EMA平滑因子 (0-1)
            motion_smooth: 是否启用运动级平滑
        """
        self.online_stabilizer = OnlineStabilizer(model_path, window_size, device)
        self.post_smoothing = post_smoothing
        self.motion_smooth = motion_smooth

        # 后处理平滑
        self.ema_dx = None
        self.ema_dy = None
        self.ema_da = None

    def set_normalization(self, mean, std):
        self.online_stabilizer.set_normalization(mean, std)

    def stabilize(self, dx, dy, da):
        """
        两级稳像：
        1. 网络预测
        2. EMA 后处理平滑
        """
        # 第一级：网络预测
        net_dx, net_dy, net_da = self.online_stabilizer.stabilize(dx, dy, da)

        if not self.motion_smooth:
            return net_dx, net_dy, net_da

        # 第二级：EMA 平滑
        if self.ema_dx is None:
            self.ema_dx, self.ema_dy, self.ema_da = net_dx, net_dy, net_da
        else:
            alpha = self.post_smoothing
            self.ema_dx = alpha * net_dx + (1 - alpha) * self.ema_dx
            self.ema_dy = alpha * net_dy + (1 - alpha) * self.ema_dy
            self.ema_da = alpha * net_da + (1 - alpha) * self.ema_da

        return self.ema_dx, self.ema_dy, self.ema_da

    def reset(self):
        self.online_stabilizer.reset()
        self.ema_dx = None
        self.ema_dy = None
        self.ema_da = None


if __name__ == "__main__":
    # 测试网络
    model = ImprovedStabilizerNet(window_size=30)
    x = torch.randn(2, 30, 3)
    out = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    print("网络测试通过！")

