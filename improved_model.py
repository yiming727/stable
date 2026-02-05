"""
改进的视频稳定网络模型
=============================
改进点:
1. 引入Transformer时序建模
2. 多尺度TCN特征提取
3. 注意力融合机制
4. 残差连接和LayerNorm
5. 端到端平滑轨迹预测
6. 不确定性建模
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码 - 为序列注入时序信息"""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape
        query = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        output = torch.matmul(attn, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.w_o(output)
        
        return output


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model, d_ff=256, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层(带残差连接)"""
    def __init__(self, d_model, num_heads=4, d_ff=256, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, mask=None):
        # Self-attention with residual
        src2 = self.self_attn(src, src, src, mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # Feed-forward with residual
        src2 = self.feed_forward(src)
        src = src + src2
        src = self.norm2(src)
        
        return src


class TemporalConvNet(nn.Module):
    """多尺度时序卷积网络 - 替换原版CNN"""
    def __init__(self, input_dim, num_channels=[128, 64], kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size // 2
            )
            layers.extend([
                conv,
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.network = nn.Sequential(*layers)
        self.output_norm = nn.LayerNorm(num_channels[-1])
    
    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x.transpose(1, 2)
        return self.output_norm(x)


class EnhancedLSTM(nn.Module):
    """增强型LSTM - 带残差连接"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(EnhancedLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        # 投影层：将双向输出投影回原始输入维度，用于残差连接
        self.proj = nn.Linear(hidden_dim * 2, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 投影到输入维度
        projected = self.proj(lstm_out)
        out = projected + x  # 残差连接
        out = self.dropout(out)
        out = self.layer_norm(out)
        return out


class AttentionFusion(nn.Module):
    """注意力融合模块 - 自适应加权不同特征"""
    def __init__(self, feature_dim, num_features=3):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, features_list):
        # features_list: [feat1, feat2, feat3] 每个是 [B, feature_dim]
        stacked = torch.stack(features_list, dim=1)  # [B, 3, feature_dim]
        weights = self.attention(stacked)  # [B, 3, 1]
        weights = F.softmax(weights, dim=1)
        fused = torch.sum(stacked * weights, dim=1)  # [B, feature_dim]
        return fused


class ImprovedVideoStabilizer(nn.Module):
    """
    改进的视频稳定网络
    ====================
    架构:
    1. 输入投影 -> 位置编码
    2. Transformer编码器 (捕捉长程依赖)
    3. TCN多尺度特征 (捕捉局部模式)
    4. 增强LSTM (捕捉时序动态)
    5. 注意力融合 -> 输出
    """
    def __init__(self, input_dim=3, window_size=125, hidden_dim=64):
        super(ImprovedVideoStabilizer, self).__init__()
        
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Transformer时序建模
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=500, dropout=0.1)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads=4, d_ff=hidden_dim*4, dropout=0.1)
            for _ in range(2)
        ])
        
        # 多尺度时序卷积
        self.tcn = TemporalConvNet(
            hidden_dim,
            num_channels=[hidden_dim, hidden_dim],
            kernel_size=3,
            dropout=0.2
        )
        
        # 增强LSTM
        self.enhanced_lstm = EnhancedLSTM(
            hidden_dim, hidden_dim, num_layers=2, dropout=0.2
        )
        
        # 特征融合
        self.fusion = AttentionFusion(hidden_dim, num_features=3)
        
        # 输出层 - 预测修正量
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, input_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: [B, T, 3] - 窗口内的轨迹 (dx, dy, da)
        Returns:
            pred: [B, 3] - 预测的修正量
        """
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)  # [B, T, hidden_dim]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        for layer in self.transformer_layers:
            x = layer(x)
        
        # TCN多尺度特征
        tcn_features = self.tcn(x)
        
        # LSTM时序建模
        lstm_features = self.enhanced_lstm(x)
        
        # 全局特征池化
        global_transformer = x.mean(dim=1)
        global_tcn = tcn_features.mean(dim=1)
        global_lstm = lstm_features.mean(dim=1)
        
        # 注意力融合
        fused_features = self.fusion([global_transformer, global_tcn, global_lstm])
        
        # 输出预测
        prediction = self.output_layer(fused_features)
        
        return prediction


class UncertaintyVideoStabilizer(nn.Module):
    """
    带不确定性估计的视频稳定器
    ===============================
    额外输出每个预测的不确定性，用于后处理加权
    """
    def __init__(self, input_dim=3, hidden_dim=128):
        super(UncertaintyVideoStabilizer, self).__init__()
        
        # 共享特征提取
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 时序建模
        self.temporal = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 注意力聚合
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 预测均值
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 预测log方差 (用于不确定性)
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        """
        前向传播
        Returns:
            mean: [B, 3] - 预测修正量
            logvar: [B, 3] - log方差 (用于计算不确定性损失)
        """
        batch_size = x.size(0)
        
        # 共享特征
        shared = self.shared_net(x)
        
        # 时序建模
        temporal_out, _ = self.temporal(shared)
        
        # 注意力加权
        attn_weights = F.softmax(self.attention(temporal_out), dim=1)
        temporal_attended = torch.sum(temporal_out * attn_weights, dim=1)
        
        # 融合特征
        global_features = temporal_out.mean(dim=1)
        fused = torch.cat([global_features, temporal_attended], dim=-1)
        
        # 预测均值和方差
        mean = self.mean_head(fused)
        logvar = self.logvar_head(fused)
        
        return mean, logvar


class HybridVideoStabilizer(nn.Module):
    """
    混合视频稳定器 - 集成多种架构
    ================================
    结合Transformer、TCN和LSTM的优点
    """
    def __init__(self, input_dim=3, hidden_dim=96, window_size=125):
        super(HybridVideoStabilizer, self).__init__()
        
        # ====== 特征提取分支 ======
        
        # 分支1: 深度DNN (全局特征)
        self.dnn_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * window_size, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 分支2: Transformer (长程依赖)
        self.trans_proj = nn.Linear(input_dim, hidden_dim)
        self.trans_pos = PositionalEncoding(hidden_dim, max_len=500, dropout=0.1)
        self.trans_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads=4, d_ff=hidden_dim*4, dropout=0.1)
            for _ in range(2)
        ])
        
        # 分支3: TCN (多尺度局部特征)
        self.tcn_branch = TemporalConvNet(
            input_dim,
            num_channels=[hidden_dim//2, hidden_dim],
            kernel_size=5,
            dropout=0.2
        )
        
        # 分支4: BiLSTM (时序动态)
        self.lstm_branch = EnhancedLSTM(
            input_dim, hidden_dim//2, num_layers=2, dropout=0.2
        )
        # 投影层：将LSTM输出投影到融合所需的维度
        self.lstm_proj = nn.Linear(input_dim, hidden_dim // 2)
        
        # ====== 融合与输出 ======
        
        # 特征融合 (注意: dnn=hidden_dim*2, trans=hidden_dim, tcn=hidden_dim, lstm=hidden_dim//2)
        fusion_input_dim = hidden_dim * 2 + hidden_dim + hidden_dim + hidden_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 多输出头
        self.refine_head = nn.Linear(hidden_dim*2, input_dim)
        self.smooth_head = nn.Linear(hidden_dim*2, input_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, 3] - 轨迹窗口
        Returns:
            refine: [B, 3] - 精细修正量
            smooth: [B, 3] - 平滑修正量
        """
        batch_size = x.size(0)
        
        # DNN分支
        dnn_feat = self.dnn_branch(x)  # [B, hidden_dim*2]
        
        # Transformer分支
        trans_x = self.trans_proj(x)
        trans_x = self.trans_pos(trans_x)
        for layer in self.trans_layers:
            trans_x = layer(trans_x)
        trans_feat = trans_x.mean(dim=1)  # [B, hidden_dim]
        
        # TCN分支
        tcn_feat = self.tcn_branch(x).mean(dim=1)  # [B, hidden_dim]
        
        # LSTM分支 - EnhancedLSTM返回[B, T, input_dim]，需要投影到hidden_dim//2
        lstm_feat = self.lstm_branch(x).mean(dim=1)  # [B, input_dim]
        lstm_feat = self.lstm_proj(lstm_feat)  # [B, hidden_dim//2]
        
        # 融合
        fused = torch.cat([dnn_feat, trans_feat, tcn_feat, lstm_feat], dim=-1)
        fused = self.fusion(fused)
        
        # 多输出
        refine = self.refine_head(fused)
        smooth = self.smooth_head(fused)
        
        return refine, smooth


# ====== 损失函数 ======

class SmoothnessLoss(nn.Module):
    """平滑性损失 - 惩罚轨迹的剧烈变化"""
    def __init__(self):
        super(SmoothnessLoss, self).__init__()
    
    def forward(self, trajectory):
        """
        Args:
            trajectory: [B, T, 3] or [T, 3]
        Returns:
            smoothness_loss: 标量
        """
        # 计算相邻帧的差异
        diff = trajectory[:, 1:] - trajectory[:, :-1]  # [B, T-1, 3] 或 [T-1, 3]
        
        # 最小化差异 (L2范数)
        loss = torch.mean(diff ** 2)
        
        return loss


class TrajectoryLoss(nn.Module):
    """综合轨迹损失"""
    def __init__(self, mse_weight=1.0, smooth_weight=0.5):
        super(TrajectoryLoss, self).__init__()
        self.mse_weight = mse_weight
        self.smooth_weight = smooth_weight
        self.mse_loss = nn.MSELoss()
        self.smooth_loss = SmoothnessLoss()
    
    def forward(self, pred, target, trajectory=None):
        """
        Args:
            pred: [B, 3] - 预测的修正量
            target: [B, 3] - 目标修正量
            trajectory: [B, T, 3] - 可选的轨迹用于平滑损失
        Returns:
            total_loss: 综合损失
        """
        mse = self.mse_loss(pred, target)
        
        smooth = 0.0
        if trajectory is not None:
            smooth = self.smooth_loss(trajectory)
        
        total = self.mse_weight * mse + self.smooth_weight * smooth
        
        return total, mse, smooth


class NLLWithVarianceLoss(nn.Module):
    """带方差的NLL损失 (用于不确定性模型)"""
    def __init__(self):
        super(NLLWithVarianceLoss, self).__init__()
    
    def forward(self, mean, logvar, target):
        """
        Args:
            mean: [B, 3] - 预测均值
            logvar: [B, 3] - 预测log方差
            target: [B, 3] - 目标值
        Returns:
            loss: NLL损失
        """
        # 负对数似然: -log N(y|mu, sigma^2)
        # = 0.5 * log(2*pi*sigma^2) + (y-mu)^2 / (2*sigma^2)
        
        variance = torch.exp(logvar) + 1e-6  # 避免除零
        mse_term = (target - mean) ** 2 / variance
        log_var_term = torch.log(variance)
        
        loss = 0.5 * (mse_term + log_var_term)
        loss = loss.mean()
        
        return loss


# ====== 辅助函数 ======

def create_improved_model(model_type='hybrid', **kwargs):
    """
    创建改进的模型
    ==================
    Args:
        model_type: 'transformer', 'uncertainty', 'hybrid'
        **kwargs: 模型参数
    Returns:
        model: PyTorch模型
    """
    models = {
        'transformer': ImprovedVideoStabilizer,
        'uncertainty': UncertaintyVideoStabilizer,
        'hybrid': HybridVideoStabilizer,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)


if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    window_size = 125
    input_dim = 3
    
    # 测试Hybrid模型
    print("Testing HybridVideoStabilizer...")
    model = HybridVideoStabilizer(input_dim=input_dim, window_size=window_size)
    x = torch.randn(batch_size, window_size, input_dim)
    refine, smooth = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Refine output shape: {refine.shape}")
    print(f"  Smooth output shape: {smooth.shape}")
    
    # 测试不确定性模型
    print("\nTesting UncertaintyVideoStabilizer...")
    model = UncertaintyVideoStabilizer(input_dim=input_dim, hidden_dim=128)
    mean, logvar = model(x)
    print(f"  Mean shape: {mean.shape}")
    print(f"  Logvar shape: {logvar.shape}")
    print(f"  Variance range: [{torch.exp(logvar).min():.4f}, {torch.exp(logvar).max():.4f}]")
    
    # 测试损失函数
    print("\nTesting Loss Functions...")
    pred = torch.randn(batch_size, 3)
    target = torch.randn(batch_size, 3)
    
    traj_loss = TrajectoryLoss()
    loss, mse, smooth = traj_loss(pred, target)
    print(f"  TrajectoryLoss: {loss.item():.4f} (mse: {mse.item():.4f}, smooth: {smooth:.4f})")
    
    nll_loss = NLLWithVarianceLoss()
    loss = nll_loss(mean, logvar, target)
    print(f"  NLLWithVarianceLoss: {loss.item():.4f}")
    
    print("\nAll tests passed!")

