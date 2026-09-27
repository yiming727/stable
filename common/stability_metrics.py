"""
稳像质量评估指标：CR / DV / SS + 帧间 PSNR / SSIM。

来源分两组，**输入形态不同，两组都保留**：

A. 目录对目录（逐张 .jpg 对比），来自 2 份**字节完全相同**的副本：
       offline/basic/src/evaluate.py
       offline/enhanced/evaluate.py
   → compute_cropping_ratio / compute_distortion_value / compute_stability_score

B. 单帧对单帧，来自 3 份**只差 argparse 默认路径**的副本：
       offline/basic/src/ssimAndPsnr.py        (默认 ./9.mp4)
       online/enhanced/method/ssimAndPsnr.py   (默认 ./18.mp4)
       online/enhanced/method1/ssimAndPsnr.py  (默认 ./11.mp4)
   → compute_psnr / compute_ssim

另有一个视频对视频的完整 OOP 实现 `tools/metrics.py::VideoStabilityEvaluator`
（490 行，比 A 更健壮，但接口是"视频路径进、指标出"），**故意没有并入这里**，
见 README 的"已知问题"。三者的指标定义是同一套：
    CR = 1/scale            （裁剪比，越大越好）
    DV = 特征值比           （畸变值，越小越好）
    SS = 低频 FFT 能量占比  （稳定度，越大越好）

⚠️ 继承自原实现的两个隐患，未做修改：
  - compute_distortion_value 在 DV_seq 为空时 np.min 会抛 ValueError
  - compute_stability_score 在 FFT 全为零时除以零
这两个是既有行为，改动会掩盖真实问题，故保留。
"""

import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


# ---------------------------------------------------------------------------
# A. 目录级：CR / DV / SS
# ---------------------------------------------------------------------------

def compute_cropping_ratio(original_dir, pred_dir, sift, bf,
                           ratio=0.7, min_match_count=10, thresh=5.0):
    """裁剪比 CR：由单应矩阵的尺度分量取倒数，返回 (均值, 最小值) 并截断到 1.0。"""
    CR_seq = []

    image_paths = sorted([p for p in os.listdir(pred_dir) if p.endswith('.jpg')])

    for path in image_paths:
        img1 = cv2.imread(os.path.join(original_dir, path), 0)
        img2 = cv2.imread(os.path.join(pred_dir, path), 0)

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < ratio * n.distance]

        if len(good) >= min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, thresh)

            if M is not None:
                scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
                CR_seq.append(1 / scale)

    return min(np.mean(CR_seq), 1.0), min(np.min(CR_seq), 1.0)


def compute_distortion_value(original_dir, pred_dir, sift, bf,
                             ratio=0.7, min_match_count=10, thresh=5.0):
    """畸变值 DV：单应矩阵线性部分两个特征值的比值（排序后取小/大）的绝对值最小值。"""
    DV_seq = []

    image_paths = sorted([p for p in os.listdir(pred_dir) if p.endswith('.jpg')])

    for path in image_paths:
        img1 = cv2.imread(os.path.join(original_dir, path), 0)
        img2 = cv2.imread(os.path.join(pred_dir, path), 0)

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < ratio * n.distance]

        if len(good) >= min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, thresh)

            if M is not None:
                w, _ = np.linalg.eig(M[:2, :2])
                w = np.sort(w)[::-1]
                DV_seq.append(w[1] / w[0])

    return np.abs(np.min(DV_seq))


def compute_stability_score(pred_dir, sift, bf, ratio=0.7, min_match_count=10, thresh=5.0):
    """稳定度 SS：累积变换的平移/旋转序列做 FFT，取低频前 5 个 bin 的能量占比。

    返回 (平移与旋转的平均, 平移, 旋转)。
    """
    P_seq = []
    Pt = np.eye(3)
    image_paths = sorted([p for p in os.listdir(pred_dir) if p.endswith('.jpg')])

    for i in range(len(image_paths) - 1):
        img1 = cv2.imread(os.path.join(pred_dir, image_paths[i]), 0)
        img2 = cv2.imread(os.path.join(pred_dir, image_paths[i + 1]), 0)

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < ratio * n.distance]

        if len(good) >= min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, thresh)

            if M is not None:
                P_seq.append(Pt @ M)
                Pt = Pt @ M

    P_seq_t = [np.sqrt(M[0, 2] ** 2 + M[1, 2] ** 2) for M in P_seq]
    P_seq_r = [np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi for M in P_seq]

    fft_t = np.abs(np.fft.fft(P_seq_t)[1:]) ** 2
    fft_r = np.abs(np.fft.fft(P_seq_r)[1:]) ** 2

    fft_t = fft_t[:len(fft_t) // 2]
    fft_r = fft_r[:len(fft_r) // 2]

    SS_t = np.sum(fft_t[:5]) / np.sum(fft_t)
    SS_r = np.sum(fft_r[:5]) / np.sum(fft_r)

    return (SS_t + SS_r) / 2, SS_t, SS_r


def default_matcher(kind="orb"):
    """构造 (检测器, 匹配器) 二元组。

    原 evaluate.py 在 __main__ 里用的是 ORB + BFMatcher；
    tools/metrics.py 用的是 SIFT。
    """
    if kind.lower() == "sift":
        return cv2.SIFT_create(), cv2.BFMatcher()
    return cv2.ORB_create(), cv2.BFMatcher()


# ---------------------------------------------------------------------------
# B. 帧级：PSNR / SSIM
# ---------------------------------------------------------------------------

def compute_psnr(frame1, frame2):
    """相邻两帧的 PSNR（dB）。两帧完全相同时返回 100.0 以免除零。

    帧间 PSNR 越高 → 相邻帧越相似 → 画面越平滑 → 抖动越小。
    """
    mse = np.mean((frame1.astype(np.float64) - frame2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return _psnr(frame1, frame2, data_range=255)


def compute_ssim(frame1, frame2):
    """相邻两帧的 SSIM（转灰度后计算），范围 [-1, 1]。"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return _ssim(gray1, gray2, data_range=255)
