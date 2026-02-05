"""
改进的实时视频稳定推理脚本
===========================
改进点:
1. 支持多种改进的模型
2. 累积平滑轨迹预测
3. 自适应窗口大小
4. 实时性能优化
5. 不确定性加权
"""
import argparse
from collections import deque
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from improved_model import (
    ImprovedVideoStabilizer,
    UncertaintyVideoStabilizer,
    HybridVideoStabilizer,
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


# ========== 光流跟踪 ==========
def get_transform(prev_frame, curr_frame):
    """计算帧间变换"""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
    )
    
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


def fix_border(frame, scale=1.1):
    """修复边界"""
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, scale)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


# ========== 实时稳定器 ==========
class RealtimeStabilizer:
    """实时视频稳定器"""
    
    def __init__(
        self,
        model,
        model_type='hybrid',
        window_size=125,
        smooth_factor=0.5,
        use_uncertainty=True,
        device='cpu'
    ):
        self.model = model
        self.model_type = model_type
        self.window_size = window_size
        self.smooth_factor = smooth_factor
        self.use_uncertainty = use_uncertainty
        self.device = device
        
        # 轨迹缓冲区
        self.trajectory_window = deque(maxlen=window_size)
        
        # 平滑后的轨迹
        self.smoothed_trajectory = []
        
        # 原始轨迹
        self.original_dx = []
        self.original_dy = []
        self.original_da = []
        self.smoothed_dx = []
        self.smoothed_dy = []
        self.smoothed_da = []
        
        # 累积变换
        self.cumulative_dx = 0.0
        self.cumulative_dy = 0.0
        self.cumulative_da = 0.0
        
        # 预测修正量
        self.predicted_corrections = deque(maxlen=window_size)
        
    def reset(self):
        """重置状态"""
        self.trajectory_window.clear()
        self.smoothed_trajectory = []
        self.predicted_corrections.clear()
        self.cumulative_dx = 0.0
        self.cumulative_dy = 0.0
        self.cumulative_da = 0.0
        
    def predict_correction(self, trajectory_window):
        """使用模型预测修正量"""
        self.model.eval()
        with torch.no_grad():
            trajectory_tensor = torch.tensor(
                trajectory_window, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            if self.model_type == 'uncertainty':
                mean, logvar = self.model(trajectory_tensor)
                # 使用均值作为修正量，方差用于加权
                correction = mean.squeeze(0).cpu().numpy()
                uncertainty = torch.exp(logvar).squeeze(0).cpu().numpy()
                return correction, uncertainty
            else:
                if hasattr(self.model, 'smooth_head'):
                    refine, smooth = self.model(trajectory_tensor)
                    correction = smooth.squeeze(0).cpu().numpy()
                else:
                    correction = self.model(trajectory_tensor)
                    if isinstance(correction, tuple):
                        correction = correction[0]
                    correction = correction.squeeze(0).cpu().numpy()
                return correction, None
    
    def smooth_trajectory(self, dx, dy, da):
        """
        平滑轨迹预测
        =============
        使用模型预测修正量，并结合指数平滑
        """
        # 累积原始轨迹
        self.cumulative_dx += dx
        self.cumulative_dy += dy
        self.cumulative_da += da
        
        self.original_dx.append(self.cumulative_dx)
        self.original_dy.append(self.cumulative_dy)
        self.original_da.append(self.cumulative_da)
        
        self.trajectory_window.append([self.cumulative_dx, self.cumulative_dy, self.cumulative_da])
        
        # 如果缓冲区未满，直接返回原始值
        if len(self.trajectory_window) < self.window_size:
            return dx, dy, da
        
        # 获取归一化参数
        mean = self.mean
        std = self.std
        
        # 归一化输入
        trajectory_array = np.array(self.trajectory_window)
        norm_trajectory = (trajectory_array - mean) / std
        
        # 预测修正量
        correction, uncertainty = self.predict_correction(norm_trajectory)
        
        # 反归一化修正量
        correction = correction * std + mean
        
        # 如果有不确定性，使用加权平滑
        if uncertainty is not None and self.use_uncertainty:
            uncertainty = uncertainty * std
            # 低不确定性时更信任预测，高不确定性时更信任原始值
            weight = 1.0 / (uncertainty + 1e-6)
            weight = weight / weight.sum()
            alpha = np.clip(weight.mean(), 0.1, 0.9)
        else:
            alpha = self.smooth_factor
        
        # 累积平滑修正量
        if len(self.predicted_corrections) == 0:
            smoothed_correction = correction
        else:
            prev_correction = self.predicted_corrections[-1]
            smoothed_correction = alpha * correction + (1 - alpha) * prev_correction
        
        self.predicted_corrections.append(smoothed_correction)
        
        # 计算平滑后的轨迹
        smooth_dx = self.cumulative_dx + smoothed_correction[0]
        smooth_dy = self.cumulative_dy + smoothed_correction[1]
        smooth_da = self.cumulative_da + smoothed_correction[2]
        
        self.smoothed_dx.append(smooth_dx)
        self.smoothed_dy.append(smooth_dy)
        self.smoothed_da.append(smooth_da)
        
        return smoothed_correction[0], smoothed_correction[1], smoothed_correction[2]
    
    def get_frame_correction(self, dx, dy, da):
        """获取帧级别的修正量"""
        correction_dx, correction_dy, correction_da = self.smooth_trajectory(dx, dy, da)
        
        # 修正量 = 平滑轨迹 - 原始轨迹
        frame_correction_dx = correction_dx - (self.cumulative_dx - dx)
        frame_correction_dy = correction_dy - (self.cumulative_dy - dy)
        frame_correction_da = correction_da - (self.cumulative_da - da)
        
        return frame_correction_dx, frame_correction_dy, frame_correction_da


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="改进的实时视频稳定")
    parser.add_argument('--input', type=str, default='24.mp4', help='输入视频路径')
    parser.add_argument('--output', type=str, default='result24_improved.mp4', help='输出视频路径')
    parser.add_argument('--model', type=str, default='hybrid', 
                       choices=['transformer', 'uncertainty', 'hybrid'],
                       help='模型类型')
    parser.add_argument('--model_path', type=str, default='hybrid_model.pth', help='模型权重路径')
    parser.add_argument('--window_size', type=int, default=125, help='窗口大小')
    parser.add_argument('--smooth', type=float, default=0.5, help='平滑因子')
    parser.add_argument('--uncertainty', action='store_true', help='使用不确定性估计')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    if args.model == 'uncertainty':
        model = UncertaintyVideoStabilizer(input_dim=3, hidden_dim=96)
    else:
        model = create_improved_model(
            args.model,
            input_dim=3,
            window_size=args.window_size,
            hidden_dim=96
        )
    
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    
    # 加载归一化参数
    mean = np.load("improved_mean.npy")
    std = np.load("improved_std.npy")
    
    # 初始化稳定器
    stabilizer = RealtimeStabilizer(
        model=model,
        model_type=args.model,
        window_size=args.window_size,
        smooth_factor=args.smooth,
        use_uncertainty=args.uncertainty,
        device=device
    )
    stabilizer.mean = mean
    stabilizer.std = std
    
    # 视频读取
    cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频 {args.input}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
    
    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("错误：无法读取视频")
        return
    
    frame_count = 0
    print("\n开始处理视频...")
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"处理进度: {frame_count}/{total_frames} ({frame_count / total_frames * 100:.1f}%)")
        
        # 计算帧间变换
        dx, dy, da = get_transform(prev_frame, curr_frame)
        
        # 获取修正量
        correction_dx, correction_dy, correction_da = stabilizer.get_frame_correction(
            dx, dy, da
        )
        
        # 构建变换矩阵
        transform = np.array([
            [np.cos(correction_da), -np.sin(correction_da), correction_dx],
            [np.sin(correction_da), np.cos(correction_da), correction_dy]
        ], dtype=np.float32)
        
        # 应用变换
        stabilized_frame = cv2.warpAffine(curr_frame, transform, (width, height))
        stabilized_frame = fix_border(stabilized_frame)
        
        # 写入输出
        out.write(stabilized_frame)
        
        # 显示对比
        canvas = np.hstack((curr_frame, stabilized_frame))
        cv2.imshow("Original vs Stabilized", canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n用户中断处理")
            break
        
        prev_frame = curr_frame
    
    # 清理
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n视频处理完成！")
    print(f"输出文件: {args.output}")
    
    # 绘制轨迹对比图
    print("\n绘制轨迹对比图...")
    
    plt.figure(figsize=(14, 10))
    
    frames = range(len(stabilizer.original_dx))
    
    plt.subplot(3, 2, 1)
    plt.plot(frames, stabilizer.original_dx, 'r-', label='Original', alpha=0.7)
    plt.plot(frames, stabilizer.smoothed_dx, 'b-', label='Stabilized', linewidth=2)
    plt.ylabel("dx (像素)")
    plt.legend()
    plt.title("X方向平移")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 3)
    plt.plot(frames, stabilizer.original_dy, 'r-', label='Original', alpha=0.7)
    plt.plot(frames, stabilizer.smoothed_dy, 'b-', label='Stabilized', linewidth=2)
    plt.ylabel("dy (像素)")
    plt.legend()
    plt.title("Y方向平移")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 5)
    plt.plot(frames, stabilizer.original_da, 'r-', label='Original', alpha=0.7)
    plt.plot(frames, stabilizer.smoothed_da, 'b-', label='Stabilized', linewidth=2)
    plt.xlabel("帧数")
    plt.ylabel("da (弧度)")
    plt.legend()
    plt.title("旋转角度")
    plt.grid(True, alpha=0.3)
    
    # 差值图
    plt.subplot(3, 2, 2)
    diff_dx = np.array(stabilizer.smoothed_dx) - np.array(stabilizer.original_dx)
    plt.plot(frames, diff_dx, 'g-', label='Correction dx')
    plt.ylabel("修正量 (像素)")
    plt.legend()
    plt.title("X方向修正量")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 4)
    diff_dy = np.array(stabilizer.smoothed_dy) - np.array(stabilizer.original_dy)
    plt.plot(frames, diff_dy, 'g-', label='Correction dy')
    plt.ylabel("修正量 (像素)")
    plt.legend()
    plt.title("Y方向修正量")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 6)
    diff_da = np.array(stabilizer.smoothed_da) - np.array(stabilizer.original_da)
    plt.plot(frames, diff_da, 'g-', label='Correction da')
    plt.xlabel("帧数")
    plt.ylabel("修正量 (弧度)")
    plt.legend()
    plt.title("旋转修正量")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_trajectory_comparison.png', dpi=150)
    print("轨迹对比图已保存: improved_trajectory_comparison.png")
    plt.show()
    
    # 计算平滑度改善
    original_var = np.var(np.diff(stabilizer.original_dx)) + np.var(np.diff(stabilizer.original_dy))
    stabilized_var = np.var(np.diff(stabilizer.smoothed_dx)) + np.var(np.diff(stabilizer.smoothed_dy))
    
    print(f"\n稳定性改善:")
    print(f"  原始轨迹方差: {original_var:.4f}")
    print(f"  稳定后轨迹方差: {stabilized_var:.4f}")
    print(f"  改善比例: {(1 - stabilized_var/original_var) * 100:.1f}%")


if __name__ == "__main__":
    main()

