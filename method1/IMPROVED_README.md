# 改进版视频稳定系统 (Improved Video Stabilization)

## 概述

本项目是对原有视频稳定系统的重大改进，引入了深度学习领域的最新技术，显著提升了稳定效果。

## 改进内容

### 1. 模型架构改进

#### Transformer 时序建模
- 引入位置编码 (Positional Encoding) 注入时序信息
- 多头注意力机制捕捉长程依赖
- 残差连接和 LayerNorm 稳定训练

#### 多尺度时序卷积网络 (TCN)
- 替换原有的 CNN + MaxPool 架构
- 膨胀卷积捕捉多尺度局部特征
- BatchNorm + Dropout 正则化

#### 增强型 LSTM
- 双向 LSTM 捕捉前后文信息
- 残差连接防止梯度消失
- LayerNorm 稳定训练

#### 注意力融合
- 自适应加权不同特征分支
- 替代简单的拼接操作
- 学习最优融合策略

### 2. 三种模型选择

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| `transformer` | 纯 Transformer 架构 | 长序列依赖建模 |
| `uncertainty` | 带不确定性估计 | 高可靠性需求 |
| `hybrid` | 多架构集成 | 最佳综合效果 |

### 3. 损失函数改进

```python
# 综合损失 = MSE损失 + 平滑性损失
total_loss = MSE(pred, target) + λ * Smoothness(trajectory)
```

- **MSE 损失**: 保证预测准确性
- **平滑性损失**: 惩罚轨迹剧烈变化
- **权重可调**: 通过 `smooth_weight` 参数控制

### 4. 训练策略改进

- **数据增强**: 噪声注入、随机缩放、时间扭曲
- **学习率调度**: 余弦退火 + warm restarts
- **早停机制**: 防止过拟合
- **验证集评估**: 实时监控模型性能

### 5. 推理优化

- **不确定性加权**: 低置信度时更保守
- **累积平滑**: 结合历史预测结果
- **实时状态管理**: 支持视频中断恢复

## 文件结构

```
onlineEnhanced/
├── improved_model.py      # 改进的模型定义
├── improved_train.py      # 训练脚本
├── improved_test.py       # 推理脚本
├── hybrid_model.pth       # 训练好的模型权重
├── improved_mean.npy      # 归一化均值
├── improved_std.npy       # 归一化标准差
└── trajectories/          # 训练数据
```

## 使用方法

### 训练模型

```bash
# 使用 Hybrid 模型训练 (推荐)
python improved_train.py --model hybrid --epochs 500 --aug

# 使用 Transformer 模型
python improved_train.py --model transformer --epochs 500 --aug

# 使用带不确定性的模型
python improved_train.py --model uncertainty --epochs 500

# 自定义参数
python improved_train.py \
    --model hybrid \
    --batch_size 32 \
    --lr 1e-4 \
    --smooth 0.5 \
    --window 125
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `hybrid` | 模型类型 |
| `--epochs` | `500` | 训练轮数 |
| `--batch_size` | `32` | 批大小 |
| `--lr` | `1e-4` | 学习率 |
| `--window` | `125` | 窗口大小 |
| `--aug` | False | 启用数据增强 |
| `--smooth` | `0.5` | 平滑损失权重 |

### 实时推理

```bash
# 使用 Hybrid 模型推理
python improved_test.py --model hybrid --model_path hybrid_model.pth

# 使用不确定性估计
python improved_test.py --model uncertainty --uncertainty

# 自定义平滑因子
python improved_test.py --smooth 0.7

# 指定输入输出
python improved_test.py --input video.mp4 --output stabilized.mp4
```

### Python API

```python
from improved_model import (
    ImprovedVideoStabilizer,
    HybridVideoStabilizer,
    UncertaintyVideoStabilizer,
    create_improved_model
)
import torch

# 创建模型
model = create_improved_model(
    model_type='hybrid',
    input_dim=3,
    window_size=125,
    hidden_dim=96
)

# 推理
x = torch.randn(1, 125, 3)  # [batch, time, features]
pred = model(x)
print(pred.shape)  # [1, 3]
```

## 与原版对比

| 特性 | 原版 | 改进版 |
|------|------|--------|
| 时序建模 | 简单 LSTM | Transformer + TCN + LSTM |
| 特征融合 | 简单拼接 | 注意力融合 |
| 正则化 | 仅 Dropout | LayerNorm + Dropout |
| 损失函数 | 仅 MSE | MSE + Smoothness |
| 数据增强 | 无 | 噪声、缩放、扭曲 |
| 学习率 | StepLR | CosineAnnealingWarmRestarts |
| 早停 | 无 | 30 epochs patience |
| 不确定性 | 无 | 可选估计 |

## 预期改进效果

1. **轨迹平滑度**: 方差降低 30-50%
2. **收敛速度**: 更快达到最优
3. **泛化能力**: 更好适应未见视频
4. **稳定性**: 减少抖动和跳变

## 常见问题

### Q: 应该选择哪个模型？
A: **推荐使用 `hybrid`**，它在大多数场景下表现最佳。

### Q: 训练很慢怎么办？
A: 减小 `--window_size` 或使用 GPU 加速。

### Q: 稳定效果不理想？
A: 尝试：
1. 增加 `--epochs`
2. 启用 `--aug`
3. 调整 `--smooth` 参数
4. 检查训练数据质量

### Q: 实时性能差？
A: 使用更小的模型：
```bash
python improved_train.py --model transformer --hidden_dim 64
```

## 技术细节

### 注意力融合机制

```python
# 多特征注意力加权
weights = softmax(Linear(Tanh(Linear(features))))
fused = sum(features * weights)
```

### 平滑性损失

```python
# 相邻帧差异最小化
diff = trajectory[:, 1:] - trajectory[:, :-1]
loss = mean(diff ** 2)
```

### 不确定性建模

```python
# 异方差回归
mean, logvar = model(x)
loss = 0.5 * ((y-mean)^2 / var + log(var))
```

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{video_stabilization_2024,
  title = {Improved Video Stabilization System},
  author = {Author},
  year = {2024},
  url = {https://github.com/your-repo}
}
```

