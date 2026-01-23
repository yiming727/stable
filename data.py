import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def extract_video_trajectory_with_plot(video_path, save_path, show_plot=True):
    """
    从视频中提取轨迹，并保存为 numpy 文件
    :param video_path: 视频路径
    :param save_path: 保存轨迹的 numpy 文件路径
    :param show_plot: 是否显示轨迹曲线
    :return: 轨迹数据
    """
    # --- 参数设置 ---
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频：{video_path}")

    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    cumulative_dx = 0.0
    cumulative_dy = 0.0
    cumulative_da = 0.0

    trajectory = []  # 每帧相对第一帧的累计 dx, dy, da
    traj_points = [(0, 0)]  # 用于可视化路径轨迹

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

        if next_pts is None or len(next_pts) < 10:
            print("特征点不足，跳过帧")
            prev_gray = curr_gray
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            continue

        good_old = prev_pts[status == 1]
        good_new = next_pts[status == 1]

        m, _ = cv2.estimateAffinePartial2D(good_old, good_new)
        if m is None:
            print("无法估计变换矩阵，跳过帧")
            continue

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        # 累计变换
        cumulative_dx += dx
        cumulative_dy += dy
        cumulative_da += da

        trajectory.append([cumulative_dx, cumulative_dy, cumulative_da])

        # 轨迹点用于可视化
        last_x, last_y = traj_points[-1]
        new_x = last_x + dx
        new_y = last_y + dy
        traj_points.append((new_x, new_y))

        # 绘制轨迹路径
        draw = frame.copy()
        for i in range(1, len(traj_points)):
            cv2.line(draw,
                     (int(traj_points[i-1][0]) + 300, int(traj_points[i-1][1]) + 300),
                     (int(traj_points[i][0]) + 300, int(traj_points[i][1]) + 300),
                     (0, 255, 0), 2)

        cv2.imshow("Trajectory", draw)
        if cv2.waitKey(10) & 0xFF == 27:
            break

        prev_gray = curr_gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    cap.release()
    cv2.destroyAllWindows()

    trajectory = np.array(trajectory)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, trajectory)
    print(f"✅ 累加轨迹提取完成，保存至 {save_path}，共 {len(trajectory)} 帧")

    # --- 可选：绘制轨迹三维曲线 dx/dy/da ---
    if show_plot:
        frames = list(range(len(trajectory)))
        labels = ['累计 dx', '累计 dy', '累计 da']
        plt.figure(figsize=(12, 4))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.plot(frames, trajectory[:, i])
            plt.title(labels[i])
        plt.suptitle("提取的累计轨迹")
        plt.tight_layout()
        plt.show()

    return trajectory


# 示例调用
if __name__ == "__main__":
    video_file = "../data/Zooming/28stb.avi"
    output_npy = "./trajectories/video_74_stable.npy"
    extract_video_trajectory_with_plot(video_file, output_npy)
