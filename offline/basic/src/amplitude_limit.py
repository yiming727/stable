import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Windows 下常用中文字体路径（微软雅黑）
font_path = "C:/Windows/Fonts/msyh.ttc"
font_prop = FontProperties(fname=font_path)

plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


def synthesize_shaky_video(img_path, save_dir, n_frames=100, max_shift=10, max_rotate=0):
    """合成带平移和旋转抖动的视频帧"""
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    for i in range(n_frames):
        dx = np.random.randint(-max_shift, max_shift+1)
        dy = np.random.randint(-max_shift, max_shift+1)
        angle = np.random.uniform(-max_rotate, max_rotate)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        M[:,2] += [dx, dy]
        shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        cv2.imwrite(os.path.join(save_dir, f"{i:03d}.jpg"), shifted)
    print(f"合成带抖动的视频帧已保存到 {save_dir}")

def extract_feature_trajectories(img_dir, p0=None, max_corners=100):
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
    prev = cv2.imread(img_files[0])
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    if p0 is None:
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=8)
        if p0 is None:
            raise ValueError("未检测到特征点，请换一张内容丰富的图片！")
    else:
        p0 = p0.copy()
    p0 = p0.astype(np.float32)  # 确保float32类型

    # trajectories 初始化，存每帧N个点的二维坐标，初始化为nan
    trajectories = np.full((len(img_files), p0.shape[0], 2), np.nan, dtype=np.float32)
    trajectories[0, :, :] = p0.reshape(-1, 2)

    for i, fname in enumerate(img_files[1:], start=1):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)

        if p1 is None or st is None:
            # 跟踪失败，保持上一帧点（或用nan）
            trajectories[i, :, :] = np.nan
            p0 = p0  # 不更新
        else:
            # 用mask保留跟踪成功的点，失败点用nan填充
            new_points = np.full_like(trajectories[i], np.nan)
            new_points[st.reshape(-1) == 1] = p1[st.reshape(-1) == 1].reshape(-1, 2)
            trajectories[i, :, :] = new_points
            # 更新p0，只有成功跟踪的点继续跟踪，失败的用nan对应点跳过
            p0 = new_points.reshape(-1, 1, 2).astype(np.float32)

        prev_gray = gray

    return trajectories


def calc_residual_motion(trajectories):
    # 只统计全程都有效的点
    valid_mask = ~np.isnan(trajectories).any(axis=(0, 2))
    if not np.any(valid_mask):
        return np.nan, np.nan
    traj_valid = trajectories[:, valid_mask, :]
    diffs = np.linalg.norm(np.diff(traj_valid, axis=0), axis=2)
    mean_motion = np.mean(diffs)
    std_motion = np.std(diffs)
    return mean_motion, std_motion


import os
import numpy as np
import cv2


def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), mode='edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    return curve_smoothed[radius:-radius]


def smooth_trajectory(trajectory, radius):
    smoothed = np.copy(trajectory)
    for i in range(3):
        smoothed[:, i] = moving_average(trajectory[:, i], radius)
    return smoothed


def fix_border(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)
    return cv2.warpAffine(frame, T, (s[1], s[0]))


def stabilize_frames(input_dir, output_dir, smoothing_radius=30):
    """
    使用稀疏光流估计和移动平均对图像帧进行视频稳像处理。

    参数:
      - input_dir: 输入抖动帧文件夹（包含 .jpg 或 .png）
      - output_dir: 输出稳像帧文件夹
      - smoothing_radius: 平滑半径
    """
    os.makedirs(output_dir, exist_ok=True)
    img_files = sorted([os.path.join(input_dir, f)
                        for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])

    if len(img_files) < 2:
        raise ValueError("输入帧数量太少，无法稳像")

    # 读取第一帧
    prev = cv2.imread(img_files[0])
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    h, w = prev.shape[:2]
    n_frames = len(img_files)

    transforms = np.zeros((n_frames - 1, 3), np.float32)

    # 光流+变换估计
    for i in range(1, n_frames):
        curr = cv2.imread(img_files[i])
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                           minDistance=30, blockSize=3)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        if status is not None:
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

        if prev_pts.shape[0] < 4:
            m = np.eye(2, 3, dtype=np.float32)
        else:
            m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        if m is None:
            m = np.eye(2, 3, dtype=np.float32)

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i - 1] = [dx, dy, da]
        prev_gray = curr_gray

    # 平滑轨迹
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory, smoothing_radius)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    # 应用平滑后的变换
    for i in range(n_frames):
        frame = cv2.imread(img_files[i])
        if i == 0:
            stabilized = frame
        else:
            dx, dy, da = transforms_smooth[i - 1]
            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy

            stabilized = cv2.warpAffine(frame, m, (w, h))
            stabilized = fix_border(stabilized)

        out_path = os.path.join(output_dir, f"{i:03d}.jpg")
        cv2.imwrite(out_path, stabilized)

    print(f"稳像帧已保存到 {output_dir}")
    # 新增：返回变换参数
    return transforms, transforms_smooth


def main():
    img_path = '../Data/00001.jpg'
    n_frames = 100
    max_shifts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    max_rotate = 0
    residuals = []
    ratios = []

    for max_shift in max_shifts:
        # 1. 合成抖动视频
        shaky_dir = f'./shaky_{max_shift}px'
        synthesize_shaky_video(img_path, shaky_dir, n_frames=n_frames, max_shift=max_shift, max_rotate=max_rotate)

        # # 2. 稳像处理（你需要实现stabilize_video）
        # stabilized_dir = f'./stabilized_{max_shift}px'
        # stabilize_frames(shaky_dir, stabilized_dir, smoothing_radius=30)
        #
        # # 3. 提取轨迹
        # # traj_in = extract_feature_trajectories(shaky_dir)
        # # traj_out = extract_feature_trajectories(stabilized_dir)
        # traj_in = extract_feature_trajectories(shaky_dir)
        # traj_out = extract_feature_trajectories(stabilized_dir, p0=traj_in[0])  # 关键修复
        #
        # # 4. 计算指标
        # mean_in, _ = calc_residual_motion(traj_in)
        # mean_out, _ = calc_residual_motion(traj_out)
        # residuals.append(mean_out)
        # ratios.append(mean_in / mean_out if mean_out > 1e-5 else np.nan)
        # print(f'抖动幅度: {max_shift:2d}px | 输入均值: {mean_in:.3f} | 稳像后: {mean_out:.3f} | 抑制比: {ratios[-1]:.2f}')

        # 2. 稳像处理，并获取变换参数
        stabilized_dir = f'./stabilized_{max_shift}px'
        transforms, transforms_smooth = stabilize_frames(shaky_dir, stabilized_dir, smoothing_radius=30)

        # 3. 统计仿射参数的均值/方差
        motion_in = np.mean(np.linalg.norm(transforms, axis=1))
        motion_out = np.mean(np.linalg.norm(transforms_smooth, axis=1))
        residuals.append(motion_out)
        ratios.append(motion_in / motion_out if motion_out > 1e-5 else np.nan)
        print(f'抖动幅度: {max_shift:2d}px | 输入均值: {motion_in:.3f} | 稳像后: {motion_out:.3f} | 抑制比: {ratios[-1]:.2f}')

    # 5. 画图
    plt.figure()
    plt.plot(max_shifts, residuals, marker='o', label='残余运动')
    plt.xlabel('输入最大抖动幅度 (px)')
    plt.ylabel('稳像后残余运动 (px)')
    plt.title('防抖极限实验')
    plt.grid(True)
    # plt.xlim(0, max(max_shifts) + 2)  # 横坐标范围
    # plt.ylim(0, max(residuals) * 1.1)  # 纵坐标范围
    # plt.axhline(y=1, color='red', linestyle='--', label='y=1')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(max_shifts, ratios, marker='s', label='抑制比')
    plt.xlabel('输入最大抖动幅度 (px)')
    plt.ylabel('稳像抑制比')
    plt.title('防抖极限实验-抑制比')
    plt.grid(True)
    plt.xlim(0, max(max_shifts) + 2)  # 横坐标范围
    plt.ylim(0, max(ratios) * 1.1)  # 纵坐标范围
    plt.axhline(y=1, color='red', linestyle='--', label='y=1')  # 画y=1红线
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
