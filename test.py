import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt

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

# --------------------------
# 工具函数
# --------------------------
def get_transform(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # 去除失配点
    if prev_pts is None or curr_pts is None:
        return 0, 0, 0  # 无法估计

    good_prev_pts = prev_pts[status.flatten() == 1]
    good_curr_pts = curr_pts[status.flatten() == 1]

    if len(good_prev_pts) < 10:
        return 0, 0, 0  # 特征点太少，跳过

    m = cv2.estimateAffinePartial2D(good_prev_pts, good_curr_pts, method=cv2.RANSAC)[0]

    if m is None:
        return 0, 0, 0  # 估计失败

    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])

    return dx, dy, da

# 定义一个函数，用于修复由于变换导致的边界问题
def fix_border(frame):
    s = frame.shape  # 获取帧的尺寸
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)  # 创建一个旋转矩阵
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))  # 应用旋转和缩放
    return frame

# --------------------------
# 参数配置
# --------------------------
WINDOW_SIZE = 125  # 训练模型使用的窗口长度
MODEL_PATH = "multi_model_ensemble.pth"
MEAN_PATH = "stable_mean.npy"
STD_PATH = "stable_std.npy"

# --------------------------
# 模型初始化
# --------------------------
model = MultiModelEnsembleNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()  # Assume pretrained weights loaded if available

# --------------------------
# 数据准备
# --------------------------
trajectory_window = deque(maxlen=WINDOW_SIZE)
cumulative_dx = cumulative_dy = cumulative_da = 0.0
trajectory_window.append([cumulative_dx, cumulative_dy, cumulative_da])

# 加载归一化参数
stable_mean = np.load(MEAN_PATH)
stable_std = np.load(STD_PATH)

# 用于存储每帧滤波前和滤波后的累计变换数据（便于后续画图）
original_dx, original_dy, original_da = [], [], []
smoothed_dx, smoothed_dy, smoothed_da = [], [], []

# --------------------------
# 视频读取与处理
# --------------------------
cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
if not ret:
    print("Failed to open camera.")
    cap.release()
    exit()

while True:
    ret, curr_frame = cap.read()
    if not ret:
        break

    dx, dy, da = get_transform(prev_frame, curr_frame)
    cumulative_dx += dx
    cumulative_dy += dy
    cumulative_da += da
    trajectory_window.append([cumulative_dx, cumulative_dy, cumulative_da])

    if len(trajectory_window) == WINDOW_SIZE:
        input_window = np.array(trajectory_window)
        norm_input, mean, std = normalize_trajectory(input_window)
        input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0)  # shape: [1, 125, 3]

        with torch.no_grad():
            pred = model(input_tensor).squeeze(0).cpu().numpy()  # shape: [3]

        smooth_pred = denormalize_trajectory(pred, mean, std)
        smooth_pred_dx, smooth_pred_dy, smooth_pred_da = smooth_pred
    else:
        smooth_pred_dx, smooth_pred_dy, smooth_pred_da = cumulative_dx, cumulative_dy, cumulative_da

    # 记录数据（用于后续画图）
    original_dx.append(cumulative_dx)
    original_dy.append(cumulative_dy)
    original_da.append(cumulative_da)
    smoothed_dx.append(smooth_pred_dx)
    smoothed_dy.append(smooth_pred_dy)
    smoothed_da.append(smooth_pred_da)

    # 计算变换差值
    diff_dx = smooth_pred_dx - cumulative_dx
    diff_dy = smooth_pred_dy - cumulative_dy
    diff_da = smooth_pred_da - cumulative_da

    # 将当前帧相对变换加上修正量，得到稳定化后的相对变换
    corrected_dx = dx + diff_dx
    corrected_dy = dy + diff_dy
    corrected_da = da + diff_da

    transform = np.array([
        [np.cos(corrected_da), -np.sin(corrected_da), diff_dx],
        [np.sin(corrected_da),  np.cos(corrected_da), diff_dy]
    ], dtype=np.float32)

    stabilized_frame = cv2.warpAffine(curr_frame, transform, (curr_frame.shape[1], curr_frame.shape[0]))
    stabilized_frame = fix_border(stabilized_frame)

    # 显示对比
    canvas = np.hstack((curr_frame, stabilized_frame))
    cv2.imshow("Original vs Stabilized", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_frame = curr_frame

cap.release()
cv2.destroyAllWindows()

"""
绘制滤波前与滤波后的累计变换曲线
"""
frames = range(len(original_dx))
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(frames, original_dx, 'r-', label="原始 dx")
plt.plot(frames, smoothed_dx, 'b-', label="平滑 dx")
plt.ylabel("dx")
plt.legend()
plt.title("累计平移与旋转数据对比")

plt.subplot(3, 1, 2)
plt.plot(frames, original_dy, 'r-', label="原始 dy")
plt.plot(frames, smoothed_dy, 'b-', label="平滑 dy")
plt.ylabel("dy")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(frames, original_da, 'r-', label="原始 da")
plt.plot(frames, smoothed_da, 'b-', label="平滑 da")
plt.xlabel("帧数")
plt.ylabel("da")
plt.legend()

plt.tight_layout()
plt.show()
