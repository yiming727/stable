"""
轨迹平滑图
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed


def smooth_trajectory(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory


def fix_border(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


# ── 新增：绘制平滑前后轨迹对比图 ──
def plot_trajectory(trajectory, smoothed_trajectory):
    frames  = np.arange(len(trajectory))
    titles  = ['水平位移 dx（累计）', '垂直位移 dy（累计）', '旋转角 da（累计）']
    ylabels = ['像素 (px)', '像素 (px)', '弧度 (rad)']

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('轨迹平滑前后对比', fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes):
        ax.plot(frames, trajectory[:, i],
                color='#1a6fc4', linewidth=1.0, alpha=0.75,
                label='平滑前')          # 蓝色
        ax.plot(frames, smoothed_trajectory[:, i],
                color='#d62728', linewidth=2.0, alpha=0.95,
                label='平滑后')          # 红色
        ax.set_title(titles[i], fontsize=11)
        ax.set_ylabel(ylabels[i], fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

    axes[-1].set_xlabel('帧序号', fontsize=10)
    plt.tight_layout()
    plt.show()


# ── 主流程 ──
SMOOTHING_RADIUS = 50

cap = cv2.VideoCapture(r'./24.mp4')
if not cap.isOpened():
    print("Error opening video file")
    exit()

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter('stabilized_video.mp4', fourcc, fps, (2 * w, h))

_, prev = cap.read()
if prev is None:
    print("Error reading video file")
    cap.release()
    exit()

prev_gray  = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
transforms = np.zeros((n_frames - 1, 3), np.float32)

# ── 第一遍：提取帧间变换 ──
for i in range(n_frames - 2):
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200, qualityLevel=0.01,
                                       minDistance=30, blockSize=3)
    success, curr = cap.read()
    if not success:
        break

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    idx      = np.where(status == 1)[0]
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

    transforms[i] = [dx, dy, da]
    prev_gray = curr_gray
    print(f"Frame: {i}/{n_frames - 2} - Tracked points: {len(prev_pts)}")

# ── 轨迹累积与平滑 ──
trajectory          = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth_trajectory(trajectory)
difference          = smoothed_trajectory - trajectory
transforms_smooth   = transforms + difference

# ── 绘制平滑前后对比图（蓝=原始，红=平滑后）──
plot_trajectory(trajectory, smoothed_trajectory)

# ── 第二遍：应用平滑变换输出视频 ──
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in range(n_frames - 2):
    success, frame = cap.read()
    if not success:
        break

    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da);  m[0, 1] = -np.sin(da);  m[0, 2] = dx
    m[1, 0] = np.sin(da);  m[1, 1] =  np.cos(da);  m[1, 2] = dy

    frame_stabilized = cv2.warpAffine(frame, m, (w, h))
    frame_stabilized = fix_border(frame_stabilized)

    frame_out = cv2.hconcat([frame, frame_stabilized])

    if frame_out.shape[1] > 1920:
        frame_out = cv2.resize(frame_out,
                               (frame_out.shape[1] // 2, frame_out.shape[0] // 2))
    out.write(frame_out)

cap.release()
out.release()
cv2.destroyAllWindows()