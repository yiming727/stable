import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

"""
卡尔曼滤波
"""
# class KalmanFilter:
#     def __init__(self, dt=1.0):
#         # 状态向量: [dx, dy, da, vx, vy, va]，其中 dx, dy, da 分别为平移和旋转；vx, vy, va 为速度项
#         self.state = np.zeros((6, 1), dtype=np.float32)
#         self.dt = dt
#
#         # 状态转移矩阵
#         self.A = np.array([
#             [1, 0, 0, dt, 0,  0],
#             [0, 1, 0, 0,  dt, 0],
#             [0, 0, 1, 0,  0,  dt],
#             [0, 0, 0, 1,  0,  0],
#             [0, 0, 0, 0,  1,  0],
#             [0, 0, 0, 0,  0,  1]
#         ], dtype=np.float32)
#
#         # 观测矩阵
#         self.H = np.array([
#             [1, 0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0, 0],
#             [0, 0, 1, 0, 0, 0]
#         ], dtype=np.float32)
#
#         # 过程噪声协方差
#         self.Q = np.eye(6, dtype=np.float32) * 0.01
#
#         # 观测噪声协方差
#         self.R = np.eye(3, dtype=np.float32) * 0.1
#
#         # 误差协方差矩阵
#         self.P = np.eye(6, dtype=np.float32)
#
#     def predict(self):
#         # 状态预测
#         self.state = np.dot(self.A, self.state)
#         # 误差协方差预测
#         self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
#         return self.state.copy()
#
#     def update(self, z):
#         # 将测量向量转换为 (3,1) 形状
#         z = z.reshape((3, 1))
#         # 计算卡尔曼增益
#         S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
#         K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
#
#         # 更新状态
#         y = z - np.dot(self.H, self.state)
#         self.state += np.dot(K, y)
#
#         # 动态调整 IIR 平滑系数
#         residual_norm = np.linalg.norm(y)
#         alpha = 0.2 if residual_norm < 1.0 else 0.5  # 根据残差调整平滑系数
#
#         # IIR 平滑
#         self.state[0] = alpha * self.state[0] + (1 - alpha) * self.state[0]
#         self.state[1] = alpha * self.state[1] + (1 - alpha) * self.state[1]
#         self.state[2] = alpha * self.state[2] + (1 - alpha) * self.state[2]
#
#         # 更新误差协方差
#         I = np.eye(self.P.shape[0], dtype=np.float32)
#         self.P = np.dot(I - np.dot(K, self.H), self.P)
#         return self.state.copy()

class KalmanFilter:
    def __init__(self, dt=1.0):
        # 状态向量: [dx, dy, da, vx, vy, va, ax, ay, aa]，增加加速度项
        self.state = np.zeros((9, 1), dtype=np.float32)
        self.dt = dt

        # 状态转移矩阵，加入加速度预测
        self.A = np.array([
            [1, 0, 0, dt, 0,  0, 0.5 * dt**2, 0, 0],
            [0, 1, 0, 0,  dt, 0, 0, 0.5 * dt**2, 0],
            [0, 0, 1, 0,  0,  dt, 0, 0, 0.5 * dt**2],
            [0, 0, 0, 1,  0,  0, dt, 0, 0],
            [0, 0, 0, 0,  1,  0, 0, dt, 0],
            [0, 0, 0, 0,  0,  1, 0, 0, dt],
            [0, 0, 0, 0,  0,  0, 1, 0, 0],
            [0, 0, 0, 0,  0,  0, 0, 1, 0],
            [0, 0, 0, 0,  0,  0, 0, 0, 1]
        ], dtype=np.float32)

        # 观测矩阵
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32)

        # 过程噪声协方差（降低 Q 增强平滑）
        self.Q = np.eye(9, dtype=np.float32) * 0.001

        # 观测噪声协方差（提高 R 增强平滑）
        self.R = np.eye(3, dtype=np.float32) * 1.0

        # 误差协方差矩阵
        self.P = np.eye(9, dtype=np.float32)

    def predict(self):
        # 预测下一状态
        self.state = np.dot(self.A, self.state)
        # 误差协方差预测
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.state.copy()

    def update(self, z):
        # 观测更新
        z = z.reshape((3, 1))
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 更新状态
        y = z - np.dot(self.H, self.state)
        self.state += np.dot(K, y)

        # IIR 平滑
        alpha = 0.2  # 平滑系数
        self.state[0] = alpha * self.state[0] + (1 - alpha) * self.state[0]
        self.state[1] = alpha * self.state[1] + (1 - alpha) * self.state[1]
        self.state[2] = alpha * self.state[2] + (1 - alpha) * self.state[2]

        # 更新误差协方差
        I = np.eye(self.P.shape[0], dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.state.copy()

# class LowPassFilter:
#     def __init__(self, alpha):
#         self.alpha = alpha
#         self.filtered_value = None
#
#     def update(self, value):
#         if self.filtered_value is None:
#             self.filtered_value = value
#         else:
#             self.filtered_value = self.alpha * value + (1 - self.alpha) * self.filtered_value
#         return self.filtered_value

# 定义一个函数，用于修复由于变换导致的边界问题
# def fix_border(frame):
#     s = frame.shape  # 获取帧的尺寸
#     T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)  # 创建一个旋转矩阵
#     frame = cv2.warpAffine(frame, T, (s[1], s[0]))  # 应用旋转和缩放
#     return frame

# 替代原有的 fix_border 函数：修复变换后的边界
def fix_border(frame, transform):
    """自动裁剪由于稳定化变换而引起的边界问题

    :param frame: 当前视频帧
    :param transforms: 应用的变换历史记录（变换参数列表）
    :return: 自动裁剪后的帧
    """
    # 获取帧的尺寸
    h, w = frame.shape[:2]

    # 定义初始帧的四个角点
    frame_corners = np.array([[0, 0],  # top left
                              [0, h - 1],  # bottom left
                              [w - 1, 0],  # top right
                              [w - 1, h - 1]],  # bottom right
                             dtype='float32')

    # 计算因稳定化变换导致的极值点
    min_x = min_y = max_x = max_y = 0

    # 遍历所有变换，应用到四个角点
    # for enhancedEdition in transforms:
    transform_mat = build_transformation_matrix(transform)  # 假设这是构造变换矩阵的函数
    transformed_corners = cv2.transform(np.array([frame_corners]), transform_mat)

    # 计算每个角点的位移
    delta_corners = transformed_corners - frame_corners

    # 获取位移的 x 和 y 值
    delta_x_corners = delta_corners[:, 0]
    delta_y_corners = delta_corners[:, 1]

    # 更新极值点
    min_x = min(min_x, np.min(delta_x_corners))
    min_y = min(min_y, np.min(delta_y_corners))
    max_x = max(max_x, np.max(delta_x_corners))
    max_y = max(max_y, np.max(delta_y_corners))

    # 计算裁剪区域的边界
    min_x = int(np.floor(min_x))
    min_y = int(np.floor(min_y))
    max_x = int(np.ceil(max_x))
    max_y = int(np.ceil(max_y))

    # 确保裁剪区域在帧尺寸内
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, w)
    max_y = min(max_y, h)

    # 裁剪掉边界的黑色区域
    cropped_frame = frame[min_y:h - max_y, min_x:w - max_x]
    # cropped_frame = frame[max_x:w - max_x, max_y:h - max_y]

    return cropped_frame


def build_transformation_matrix(transform):
    """根据变换参数构建变换矩阵

    :param transform: 包含 [dx, dy, da]（平移和旋转）的变换向量
    :return: 2x3 仿射变换矩阵
    """
    transform_matrix = np.zeros((2, 3))

    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]
    return transform_matrix

# 使用队列来管理帧
class FrameQueue:
    def __init__(self, max_len):
        self.frames = deque(maxlen=max_len)  # 使用deque来缓存视频帧
        self.max_len = max_len

    def append(self, frame):
        """添加帧到队列中"""
        self.frames.append(frame)

    def pop_append(self, frame):
        """如果队列已满，则将队头帧弹出并返回，然后将新帧添加到队尾"""
        popped_frame = None
        # 如果队列已满，弹出队头元素并返回
        if len(self.frames) == self.max_len:
            popped_frame = self.frames.popleft()
        # 将新的帧添加到队尾
        self.frames.append(frame)
        return popped_frame

    def pop_appenddd(self, frame):
        """将新帧与队列中的所有帧做计算后加入队列，如果队列已满，弹出最旧的帧并返回"""
        popped_frame = None

        # 如果队列已满，弹出队头元素并返回
        if len(self.frames) == self.max_len:
            popped_frame = self.frames.popleft()

        # 将新帧添加到队尾
        self.frames.append(frame)

        # 如果队列已经有元素，进行一次计算（比如求平均值）
        if len(self.frames) > 1:
            # 这里可以根据需要计算其他的值，假设我们是做平均
            avg_frame = np.mean(np.array(self.frames), axis=0)  # 对所有帧求平均
            # 将平均值更新到队列中（你可以决定如何处理这些值）
            self.frames = deque([avg_frame] * len(self.frames), maxlen=self.max_len)  # 替换为平均后的值

        return popped_frame

    def get_all(self):
        """获取队列中的所有帧"""
        return list(self.frames)

"""
参数设置
"""
# 设置平滑半径
SMOOTHING_RADIUS = 50
# 打开摄像头（或者在线视频流）
cap = cv2.VideoCapture(0)  # 使用默认摄像头，或者替换为在线视频流 URL
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# frame_skip = 3  # 处理 1 帧，跳过 2 帧
# frame_count = 0
# 检查视频是否成功打开
if not cap.isOpened():
    print("Error opening video")
    exit()
# 获取帧率，用于确定 dt
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  # 默认帧率
dt = 1.0 / fps
# 初始化卡尔曼滤波器
kf = KalmanFilter(dt=dt)
# low_pass_filter = LowPassFilter(alpha=0.2)
# 获取视频的总帧数、宽、高和帧率
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 读取视频的第一帧
_, prev = cap.read()
if prev is None:
    print("Error reading video file")
    cap.release()
    exit()
# 将第一帧转换为灰度图
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
# 创建队列
frame_queue = FrameQueue(max_len=25)
time.sleep(1)
# 卡尔曼滤波时定义累计变换参数，初始为零
cumulative_dx = cumulative_dy = cumulative_da = 0.0
"""
测试
"""
# 用于存储每帧滤波前和滤波后的累计变换数据（便于后续画图）
original_dx = []
original_dy = []
original_da = []
smoothed_dx = []
smoothed_dy = []
smoothed_da = []
# # 初始化视频写入器
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 选择 XVID 编码格式（.avi）
# out = cv2.VideoWriter('stabilized_output.avi', fourcc, 30, (w, h))  # 帧率设为 10，分辨率设为 (w, h)
# # 视频参数设置
# output_filename_original = "original_output.mp4"
# output_filename_stabilized = "stabilized_output.mp4"
# frame_size1 = None  # 在第一帧时确定
# frame_size2 = None
# fps = 30  # 假设视频帧率为30
# codec = cv2.VideoWriter_fourcc(*"mp4v")  # 选择MP4编码格式
# # VideoWriter 初始化（延迟到第一帧确定大小）
# video_writer_original = None
# video_writer_stabilized = None


# 主循环，处理视频流
while True:
    success, curr = cap.read()
    if not success:
        break

    # frame_count += 1
    # if frame_count % frame_skip != 0:
    #     continue  # 跳过帧

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    # # Shi-Tomasi特征点检测
    # prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    # #FAST特征点检测
    # fast = cv2.FastFeatureDetector_create()
    # keypoints = fast.detect(prev_gray, None)
    # prev_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    # ORB 特征点检测
    orb = cv2.ORB_create(nfeatures=100)  # 增加特征点数量
    keypoints = orb.detect(prev_gray, None)
    prev_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # 筛选出成功跟踪的点
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # 如果跟踪的点太少，则使用单位矩阵
    if prev_pts.shape[0] < 4:
        m = np.eye(2, 3, dtype=np.float32)
    else:
        # m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)  # 估计仿射变换矩阵
        m, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
        # m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)

    if m is None:
        m = np.eye(2, 3, dtype=np.float32)

    # 提取变换参数
    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])

    # # 如果跟踪的点太少，则使用单位矩阵
    # if prev_pts.shape[0] < 4:
    #     m = np.eye(3, 3, dtype=np.float32)  # 使用3x3单位矩阵
    # else:
    #     # 使用 cv2.findHomography 计算投射变换矩阵（3x3矩阵）
    #     m, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
    #
    # # 如果变换矩阵是空的，则使用单位矩阵
    # if m is None:
    #     m = np.eye(3, 3, dtype=np.float32)
    #
    # # 提取变换参数（平移和旋转部分）
    # dx = m[0, 2]
    # dy = m[1, 2]
    # # 透视修正可以通过更复杂的方式获得，如通过卡尔曼滤波计算累计修正值
    # da = np.arctan2(m[1, 0], m[0, 0])  # 如果需要旋转角度的话

    """
    卡尔曼滤波
    """
    # # 累加当前帧的变换，得到历来的累计运动
    cumulative_dx += dx
    cumulative_dy += dy
    cumulative_da += da

    # 构造累计变换的测量向量，传入卡尔曼滤波器
    measurement = np.array([cumulative_dx, cumulative_dy, cumulative_da], dtype=np.float32)

    # 卡尔曼滤波：先预测，再更新，得到平滑后的累计变换
    kf.predict()
    smoothed_state = kf.update(measurement)
    # # 低通滤波
    # smoothed_state = low_pass_filter.update(smoothed_state)
    smooth_cumulative_dx = smoothed_state[0, 0]
    smooth_cumulative_dy = smoothed_state[1, 0]
    smooth_cumulative_da = smoothed_state[2, 0]

    # 记录数据（用于后续画图）
    original_dx.append(cumulative_dx)
    original_dy.append(cumulative_dy)
    original_da.append(cumulative_da)
    smoothed_dx.append(smooth_cumulative_dx)
    smoothed_dy.append(smooth_cumulative_dy)
    smoothed_da.append(smooth_cumulative_da)

    # 计算累计轨迹与平滑累计轨迹之间的差值（即需要修正的量）
    diff_dx = smooth_cumulative_dx - cumulative_dx
    diff_dy = smooth_cumulative_dy - cumulative_dy
    diff_da = smooth_cumulative_da - cumulative_da

    # 将当前帧相对变换加上修正量，得到稳定化后的相对变换
    corrected_dx = dx + diff_dx
    corrected_dy = dy + diff_dy
    corrected_da = da + diff_da

    # 将修正后的变换参数组合成变换向量
    transform = np.array([corrected_dx, corrected_dy, corrected_da], dtype=np.float32)

    # 构造平滑后的仿射变换矩阵（这里只处理平移和旋转，可根据需要加入缩放）
    m_smooth = np.zeros((2, 3), dtype=np.float32)
    m_smooth[0, 0] = np.cos(corrected_da)
    m_smooth[0, 1] = -np.sin(corrected_da)
    m_smooth[1, 0] = np.sin(corrected_da)
    m_smooth[1, 1] = np.cos(corrected_da)
    m_smooth[0, 2] = corrected_dx
    m_smooth[1, 2] = corrected_dy

    # 应用变换到当前帧
    frame_stabilized = cv2.warpAffine(curr, m_smooth, (w, h))
    # frame_stabilized = cv2.warpPerspective(curr, m_smooth, (w, h))
    # out.write(frame_stabilized)

    # 修复变换后的边界问题
    frame_stabilized = fix_border(frame_stabilized, transform)
    # frame_stabilized = fix_border(frame_stabilized)

    # 确保两张图像大小一致
    frame_stabilized_resized = cv2.resize(frame_stabilized, (curr.shape[1], curr.shape[0]))
    # 将原始帧和平滑帧并排放置
    frame_out = cv2.hconcat([curr, frame_stabilized_resized])

    # # 初始化 VideoWriter（仅第一次）
    # if frame_size1 is None:
    #     frame_size1 = (curr.shape[1], curr.shape[0])
    #     video_writer_original = cv2.VideoWriter(output_filename_original, codec, fps, frame_size1)
    # # 初始化 VideoWriter（仅第一次）
    # if frame_size2 is None:
    #     frame_size2 = (frame_stabilized.shape[1], frame_stabilized.shape[0])
    #     video_writer_stabilized = cv2.VideoWriter(output_filename_stabilized, codec, fps, frame_size2)
    #
    # # 保存帧到视频
    # video_writer_original.write(curr)
    # video_writer_stabilized.write(frame_stabilized)

    """
    采用队列
    """
    # popped_frame = frame_queue.pop_append(frame_out)
    #
    # # 如果队列有足够的帧，可以开始延迟输出
    # if popped_frame is not None:
    #     # 显示输出
    #     cv2.imshow("Original vs Stabilized", popped_frame)

    # print(frame_stabilized.shape)
    cv2.imshow("Original vs Stabilized", frame_out)

    # 按‘q’键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = curr_gray

# 释放所有资源
cap.release()
cv2.destroyAllWindows()
# video_writer_original.release()
# video_writer_stabilized.release()

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
