import argparse
from collections import deque
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import MultiModelEnsembleNet

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 归一化 ----------
def normalize_trajectory(traj):
    mean = traj.mean(axis=0)
    std = traj.std(axis=0) + 1e-8
    return (traj - mean) / std, mean, std

def denormalize_trajectory(traj, mean, std):
    return traj * std + mean

# --------------------------
# 光流跟踪模块
# --------------------------
def get_transform(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    if prev_pts is None:
        return 0, 0, 0

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    if curr_pts is None:
        return 0, 0, 0

    good_prev_pts = prev_pts[status.flatten() == 1]
    good_curr_pts = curr_pts[status.flatten() == 1]

    if len(good_prev_pts) < 10:
        return 0, 0, 0

    m = cv2.estimateAffinePartial2D(good_prev_pts, good_curr_pts, method=cv2.RANSAC)[0]

    if m is None:
        return 0, 0, 0

    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])

    return dx, dy, da


def fix_border(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

# --------------------------
# 命令行参数解析
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="视频稳定处理（深度学习版）")
    parser.add_argument('--input', type=str, default='24.mp4',
                        help='输入视频路径（默认：input.mp4）')
    parser.add_argument('--output', type=str, default='result24.mp4',
                        help='输出视频路径（默认：output.mp4）')
    parser.add_argument('--model', type=str, default='multi_model_ensemble.pth',
                        help='模型权重路径')
    parser.add_argument('--window_size', type=int, default=125,
                        help='滑动窗口大小')
    return parser.parse_args()


# --------------------------
# 主函数
# --------------------------
def main():
    # 解析参数
    args = parse_args()

    # --------------------------
    # 参数配置
    # --------------------------
    WINDOW_SIZE = args.window_size
    MODEL_PATH = args.model
    MEAN_PATH = "../stable_mean.npy"
    STD_PATH = "../stable_std.npy"
    VIDEO_INPUT = args.input  # 输入视频路径
    VIDEO_OUTPUT = args.output  # 输出视频路径

    # --------------------------
    # 模型初始化
    # --------------------------
    print(f"加载模型: {MODEL_PATH}")
    model = MultiModelEnsembleNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # --------------------------
    # 数据准备
    # --------------------------
    trajectory_window = deque(maxlen=WINDOW_SIZE)
    cumulative_dx = cumulative_dy = cumulative_da = 0.0
    trajectory_window.append([cumulative_dx, cumulative_dy, cumulative_da])

    # 加载归一化参数
    stable_mean = np.load(MEAN_PATH)
    stable_std = np.load(STD_PATH)

    original_dx, original_dy, original_da = [], [], []
    smoothed_dx, smoothed_dy, smoothed_da = [], [], []

    # --------------------------
    # 视频读取（从文件）
    # --------------------------
    cap = cv2.VideoCapture(VIDEO_INPUT)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f" 错误：无法打开视频文件 {VIDEO_INPUT}")
        return

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print(" 错误：无法读取视频第一帧")
        cap.release()
        return

    # --------------------------
    # 视频处理循环
    # --------------------------
    frame_count = 0
    print("\n开始处理视频...")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 显示进度
        if frame_count % 30 == 0:
            print(f"处理进度: {frame_count}/{total_frames} ({frame_count / total_frames * 100:.1f}%)")

        dx, dy, da = get_transform(prev_frame, curr_frame)
        cumulative_dx += dx
        cumulative_dy += dy
        cumulative_da += da
        trajectory_window.append([cumulative_dx, cumulative_dy, cumulative_da])

        if len(trajectory_window) == WINDOW_SIZE:
            input_window = np.array(trajectory_window)
            norm_input, mean, std = normalize_trajectory(input_window)
            input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred = model(input_tensor).squeeze(0).cpu().numpy()

            smooth_pred = denormalize_trajectory(pred, mean, std)
            smooth_pred_dx, smooth_pred_dy, smooth_pred_da = smooth_pred
        else:
            smooth_pred_dx, smooth_pred_dy, smooth_pred_da = cumulative_dx, cumulative_dy, cumulative_da

        original_dx.append(cumulative_dx)
        original_dy.append(cumulative_dy)
        original_da.append(cumulative_da)
        smoothed_dx.append(smooth_pred_dx)
        smoothed_dy.append(smooth_pred_dy)
        smoothed_da.append(smooth_pred_da)

        diff_dx = smooth_pred_dx - cumulative_dx
        diff_dy = smooth_pred_dy - cumulative_dy
        diff_da = smooth_pred_da - cumulative_da

        corrected_dx = dx + diff_dx
        corrected_dy = dy + diff_dy
        corrected_da = da + diff_da

        transform = np.array([
            [np.cos(corrected_da), -np.sin(corrected_da), corrected_dx],
            [np.sin(corrected_da), np.cos(corrected_da), corrected_dy]
        ], dtype=np.float32)

        stabilized_frame = cv2.warpAffine(curr_frame, transform, (width, height))
        stabilized_frame = fix_border(stabilized_frame)

        # 写入输出视频
        out.write(stabilized_frame)

        # 显示对比（可选，按 'q' 退出）
        canvas = np.hstack((curr_frame, stabilized_frame))
        cv2.imshow("Original vs Stabilized", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n用户中断处理")
            break

        prev_frame = curr_frame

    # --------------------------
    # 清理资源
    # --------------------------
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n 视频处理完成！")
    print(f"   输出文件: {VIDEO_OUTPUT}")
    print(f"   处理帧数: {frame_count}/{total_frames}")

    # --------------------------
    # 绘制轨迹对比图
    # --------------------------
    print("\n绘制轨迹对比图...")
    frames = range(len(original_dx))
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(frames, original_dx, 'r-', label="原始 dx", alpha=0.7)
    plt.plot(frames, smoothed_dx, 'b-', label="平滑 dx", linewidth=2)
    plt.ylabel("dx (像素)")
    plt.legend()
    plt.title("累计平移与旋转数据对比")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(frames, original_dy, 'r-', label="原始 dy", alpha=0.7)
    plt.plot(frames, smoothed_dy, 'b-', label="平滑 dy", linewidth=2)
    plt.ylabel("dy (像素)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(frames, original_da, 'r-', label="原始 da", alpha=0.7)
    plt.plot(frames, smoothed_da, 'b-', label="平滑 da", linewidth=2)
    plt.xlabel("帧数")
    plt.ylabel("da (弧度)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150)
    print(" 轨迹对比图已保存: trajectory_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
