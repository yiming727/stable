import numpy as np
import cv2
from matplotlib import pyplot as plt

img2 = cv2.imread('1.png', 0)
img1 = cv2.imread('2.png', 0)
src_img1 = cv2.imread('1.png', 1)
src_img2 = cv2.imread('2.png', 1)

match_num = 600
threshold = 1000
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

# 根据距离排序
matches_list = sorted(matches, key=lambda x: x.distance)

#以下m_data为一个 cv2.DMatch对象
# m_data.queryIdx  代表特征模板（也就是这里的s.bmp）特征点的索引
# m_data.trainIdx  代表特征点在另一张图像(这里的b.bmp)中相匹配的特征点的索引
# m_data.distance 代表对应特征点之间的欧氏距离，越小表明匹配度越高
# m_data.imgIdx  代表进行匹配图像的索引，如已知一幅图像的sift描述子，与其他十幅图像的描述子进行匹配，找最相似的图像，则imgIdx此时就有用了。
#以下 为一个 cv2.KeyPoint 对象， pt返回坐标
# (int(cnt[0]), int(cnt[1])) 就是得到的对应点的坐标

for m_data in matches[:match_num]:
    if m_data.distance < threshold and m_data.trainIdx < len(kp1):
        cnt1 = kp1[m_data.queryIdx].pt  # 第一张图像中的特征点坐标
        cnt2 = kp2[m_data.trainIdx].pt  # 第二张图像中的特征点坐标
        cv2.circle(src_img1, (int(cnt1[0]), int(cnt1[1])), 5, (0, 0, 255), -1)
        cv2.circle(src_img2, (int(cnt2[0]), int(cnt2[1])), 5, (0, 0, 255), -1)
        print("Point in img1: ", cnt1)
        print("Point in img2: ", cnt2)

cv2.imshow("result1",src_img1)
cv2.imwrite("res_b1.bmp", src_img1)
cv2.imshow("result2",src_img2)
cv2.imwrite("res_b2.bmp", src_img2)
cv2.waitKey(0)

