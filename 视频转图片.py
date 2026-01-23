import cv2
import os

def extract_frames(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取视频文件名（不带路径和扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 从 0 开始计数
    frame_count = 0

    # 逐帧读取视频并保存为图像文件
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_filename = os.path.join(output_folder, f"{frame_count:05d}.png")
        cv2.imwrite(frame_filename, frame)

    # 释放视频流
    cap.release()

if __name__ == '__main__':
    video_path = './result24.mp4'
    output_folder = './result24'
    extract_frames(video_path, output_folder)
