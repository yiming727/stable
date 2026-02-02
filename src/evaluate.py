import os
import sys
import numpy as np
import cv2

def compute_cropping_ratio(original_dir, pred_dir, sift, bf, ratio=0.7, min_match_count=10, thresh=5.0):
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
                scale = np.sqrt(M[0,0]**2 + M[0,1]**2)
                CR_seq.append(1 / scale)

    return min(np.mean(CR_seq), 1.0), min(np.min(CR_seq), 1.0)

def compute_distortion_value(original_dir, pred_dir, sift, bf, ratio=0.7, min_match_count=10, thresh=5.0):
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

    P_seq_t = [np.sqrt(M[0,2]**2 + M[1,2]**2) for M in P_seq]
    P_seq_r = [np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi for M in P_seq]

    fft_t = np.abs(np.fft.fft(P_seq_t)[1:])**2
    fft_r = np.abs(np.fft.fft(P_seq_r)[1:])**2

    fft_t = fft_t[:len(fft_t)//2]
    fft_r = fft_r[:len(fft_r)//2]

    SS_t = np.sum(fft_t[:5]) / np.sum(fft_t)
    SS_r = np.sum(fft_r[:5]) / np.sum(fft_r)

    return (SS_t + SS_r) / 2, SS_t, SS_r

if __name__ == '__main__':
    original_dir = '../Data/2/'
    pred_dir = '../Data/result2/'

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher()

    avg_cr, min_cr = compute_cropping_ratio(original_dir, pred_dir, orb, bf)
    print('***Cropping ratio (Avg, Min):')
    print(f'{avg_cr:.4f} | {min_cr:.4f}')

    min_dv = compute_distortion_value(original_dir, pred_dir, orb, bf)
    print('***Distortion value:')
    print(f'{min_dv:.4f}')

    ss_avg_pred, ss_trans_pred, ss_rot_pred = compute_stability_score(pred_dir, orb, bf)
    ss_avg_orig, ss_trans_orig, ss_rot_orig = compute_stability_score(original_dir, orb, bf)

    print('***Stability Score Comparison:')
    print(f'Original  : {ss_avg_orig:.4f} | {ss_trans_orig:.4f} | {ss_rot_orig:.4f}')
    print(f'Predicted : {ss_avg_pred:.4f} | {ss_trans_pred:.4f} | {ss_rot_pred:.4f}')
    print(f'Improvement: {ss_avg_pred - ss_avg_orig:.4f} | {ss_trans_pred - ss_trans_orig:.4f} | {ss_rot_pred - ss_rot_orig:.4f}')



