import os
import glob
import cv2
import torch
import numpy as np
import re
from models.matching import Matching
from models.utils import read_image

def video_to_images(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_path = os.path.join(output_dir, f"{idx}.jpg")
        cv2.imwrite(img_path, frame)
        idx += 1
    cap.release()
    print(f"共保存{idx-1}帧到{output_dir}")

def detect_and_match(img_path0, img_path1, matching, device='cuda'):
    image0, inp0, _ = read_image(img_path0, device, [320, 240], 0, False)
    image1, inp1, _ = read_image(img_path1, device, [320, 240], 0, False)
    if image0 is None or image1 is None:
        raise ValueError(f"读取图片失败: {img_path0}, {img_path1}")
    with torch.no_grad():
        pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    return {
        'keypoints0': kpts0,
        'keypoints1': kpts1,
        'matches': matches,
        'match_confidence': conf,
        'matched_keypoints0': mkpts0,
        'matched_keypoints1': mkpts1,
        'matched_confidence': mconf
    }

def save_match_to_npz(save_path, result):
    np.savez(
        save_path,
        keypoints0=result['keypoints0'],
        keypoints1=result['keypoints1'],
        matches=result['matches'],
        match_confidence=result['match_confidence']
    )

def natural_sort_key(file):
    # 只对文件名部分做分割，防止路径中有数字
    filename = os.path.basename(file)
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

def process_video_to_npz(video_path, img_dir='./video_frames', save_dir='./dump_match_pairs'):
    # 1. 视频转图片
    video_to_images(video_path, img_dir)

    # 2. 初始化SuperGlue模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval().to(device)

    os.makedirs(save_dir, exist_ok=True)

    # 3. 获取图片列表并自然排序
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')), key=natural_sort_key)
    N = len(img_files)
    print(f'共{N}帧')

    # 4. 滑动窗口配对并保存
    for i in range(N-1):
        img0 = img_files[i]
        img1 = img_files[i+1]
        print(f'匹配: {os.path.basename(img0)} <-> {os.path.basename(img1)}')
        result = detect_and_match(img0, img1, matching, device=device)
        npz_name = f'{os.path.splitext(os.path.basename(img0))[0]}_{os.path.splitext(os.path.basename(img1))[0]}.npz'
        save_path = os.path.join(save_dir, npz_name)
        save_match_to_npz(save_path, result)
        print(f'保存: {save_path}')

    # 5. 返回所有npz文件路径
    npz_files = sorted(glob.glob(os.path.join(save_dir, '*.npz')), key=natural_sort_key)

    print('所有npz文件:', npz_files)
    return npz_files

# if  __name__ == '__main__':
#     video_path = '../Data/11.mp4'
#     process_video_to_npz(video_path)
