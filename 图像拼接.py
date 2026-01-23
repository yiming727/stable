import cv2
import numpy as np
import os
from PIL import Image


def image_stitching(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img = np.zeros_like(img1)
    ind1 = np.where(gray1 != 0)
    img[ind1] = img1[ind1]
    ind2 = np.where(gray2 != 0)
    img[ind2] = img2[ind2]
    return img


def process_folders(folder1, folder2, output_folder_stitched, output_folder_cropped):
    # 获取文件夹中的图片文件名
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    # 确保两个文件夹中图片数量相同
    if len(files1) != len(files2):
        raise ValueError("两个文件夹中的图片数量不同")

    # 遍历每对图片
    for file1, file2 in zip(files1, files2):
        # 读取图片
        img1 = cv2.imread(os.path.join(folder1, file1))
        img2 = cv2.imread(os.path.join(folder2, file2))

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.ORB_create()
        kp1, desc1 = sift.detectAndCompute(gray1, None)
        kp2, desc2 = sift.detectAndCompute(gray2, None)

        bf = cv2.BFMatcher(crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        good = sorted(good, key=lambda x: x.distance)[:len(good) // 3]

        pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pt2, pt1, cv2.RANSAC, ransacReprojThreshold=4.0)

        height = img1.shape[0] + img2.shape[0]
        width = img1.shape[1] + img2.shape[1]
        canvas = np.zeros((height, width, 3), np.uint8)
        canvas = cv2.warpPerspective(img2, H, (canvas.shape[1], canvas.shape[0]))

        final_image = image_stitching(canvas, img1)

        # 保存拼接后的图片
        output_file_stitched = os.path.join(output_folder_stitched, f'stitched_{file1}')
        cv2.imwrite(output_file_stitched, final_image)

        # 裁剪图片
        img = Image.open(output_file_stitched)
        box = (0, 0, 320, 240)  # 需要裁剪的区域
        cropped_img = img.crop(box)

        # 保存裁剪后的图片
        output_file_cropped = os.path.join(output_folder_cropped, f'cropped_{file1}')
        cropped_img.save(output_file_cropped)

# 设置文件夹路径
folder1 = './01'
folder2 = './02'
output_folder_stitched = './output_stitched'
output_folder_cropped = './output_cropped'

# 创建输出文件夹
if not os.path.exists(output_folder_stitched):
    os.makedirs(output_folder_stitched)
if not os.path.exists(output_folder_cropped):
    os.makedirs(output_folder_cropped)

process_folders(folder1, folder2, output_folder_stitched, output_folder_cropped)