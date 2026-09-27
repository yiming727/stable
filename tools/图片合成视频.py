import cv2
import os

def natural_sort_key(filename):
    # 将文件名按数字和字母分开
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]

def frame2vid(src, vidDir):
    images = [img for img in os.listdir(src) if img.endswith(".png")]

    # 使用自定义的自然数排序
    images.sort(key=natural_sort_key)

    # 打印排序后的图像列表
    print("排序后的图像列表:")
    print(images)

    video = cv2.VideoWriter(vidDir, cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))

    for image in images:
        video.write(cv2.imread(os.path.join(src, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    import re  # 导入正则表达式模块
    frame2vid(src='./result19', vidDir='./9.mp4')
