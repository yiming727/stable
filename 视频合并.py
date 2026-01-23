import cv2

# 打开两个视频文件
video1_path = './07.mp4'  # 替换为你的第一个视频路径
video2_path = './DIFRINT07.mp4'  # 替换为你的第二个视频路径

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# 获取视频的基本信息
frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap1.get(cv2.CAP_PROP_FPS)

# 创建一个 VideoWriter 对象以保存合并后的视频
output_path = './DIFRINTCombine07.mp4'  # 输出视频路径
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))

while True:
    # 读取帧
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # 如果任一视频结束，则退出循环
    if not ret1 or not ret2:
        break

    # 获取每个帧的右半边
    right_half1 = frame1[:, frame_width // 2:]  # 第一个视频的右半边
    right_half2 = frame2[:, frame_width // 2:]  # 第二个视频的右半边

    # 合并两个右半边
    combined_frame = cv2.hconcat([frame1, frame2])

    # 写入合并后的视频
    out.write(combined_frame)

# 释放资源
cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()

print("视频合并完成，输出文件:", output_path)
