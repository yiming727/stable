import cv2
import numpy as np
import os
import argparse


def compute_adjacent_diff(cap, output_dir, label):
    """
    读取视频，计算相邻帧差分，保存原始帧和差分帧
    cap        : cv2.VideoCapture 对象
    output_dir : 根输出目录
    label      : 子目录前缀，如 "before" / "after"
    """
    frames_dir = os.path.join(output_dir, f"frames_{label}")
    diff_dir   = os.path.join(output_dir, f"diff_{label}")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(diff_dir,   exist_ok=True)

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:05d}.png"), frame)
        frames.append(frame)
        idx += 1

    print(f"[{label}] 共读取 {len(frames)} 帧，已保存至 {frames_dir}")

    # 相邻帧差分：diff_i = |frame_{i+1} - frame_i|
    diff_count = 0
    for i in range(len(frames) - 1):
        f1 = frames[i]
        f2 = frames[i + 1]

        # 尺寸对齐（以前一帧为基准）
        h, w = f1.shape[:2]
        f2 = cv2.resize(f2, (w, h))

        diff = cv2.absdiff(f1, f2)
        diff_enhanced = np.clip(diff.astype(np.float32) * 3.0, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(diff_dir, f"diff_{i:05d}_{i+1:05d}.png"), diff_enhanced)
        diff_count += 1

    print(f"[{label}] 共生成 {diff_count} 张差分图，已保存至 {diff_dir}")
    return frames, diff_dir


def video_diff_validator(before_path, after_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    cap_before = cv2.VideoCapture(before_path)
    cap_after  = cv2.VideoCapture(after_path)

    if not cap_before.isOpened():
        raise FileNotFoundError(f"无法打开稳定前视频: {before_path}")
    if not cap_after.isOpened():
        raise FileNotFoundError(f"无法打开稳定后视频: {after_path}")

    print("=== 处理稳定前视频 ===")
    frames_before, diff_before_dir = compute_adjacent_diff(cap_before, output_dir, "before")

    print("\n=== 处理稳定后视频 ===")
    frames_after, diff_after_dir = compute_adjacent_diff(cap_after, output_dir, "after")

    cap_before.release()
    cap_after.release()

    print(f"""
========== 完成 ==========
输出目录        : {output_dir}
稳定前原始帧    : {output_dir}/frames_before/
稳定前差分帧    : {output_dir}/diff_before/
稳定后原始帧    : {output_dir}/frames_after/
稳定后差分帧    : {output_dir}/diff_after/
==========================
差分越暗 → 相邻帧变化越小 → 视频越稳定
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频稳定有效性验证（相邻帧差分法）")
    parser.add_argument("--before", default="./8.mp4", help="稳定前视频路径")
    parser.add_argument("--after", default="./result8.mp4", help="稳定后视频路径")
    parser.add_argument("--output", default="./output", help="输出根目录（默认: ./output）")
    args = parser.parse_args()

    video_diff_validator(args.before, args.after, args.output)