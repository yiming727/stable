"""
图片/视频互转与文件名自然排序。

来源（3 份 natural_sort_key 副本，**三种变体都不同**）：

    offlineTool/图片合成视频.py   整串分割，不转小写
    offlineTool/图片排序.py       整串分割，字母转小写
    offlineEnhanced/superglue.py  先取 basename 再分割，不转小写

这里合并成一个带开关的实现，默认行为 = 图片合成视频.py 那一版，
另外两个的差异用 basename / lower 参数还原。

⚠️ 原 `offlineTool/图片合成视频.py` 有个隐患：`natural_sort_key` 里用了
`re.split`，但 `re` 只写在了 `if __name__ == '__main__':` 块里（第 27 行）。
以脚本方式运行没问题，但 `import 图片合成视频` 后再调用该函数会 NameError。
本模块在文件顶部正常导入 re，不存在该问题。
"""

import os
import re

import cv2


def natural_sort_key(name, basename=False, lower=False):
    """把文件名按"数字段当作整数"的方式切分，用于自然排序。

    参数:
        name:     文件名或路径
        basename: 是否先取 basename（防止路径中的数字干扰）→ superglue.py 用 True
        lower:    字母段是否转小写 → 图片排序.py 用 True
    返回:
        可直接作为 sort(key=...) 使用的列表
    """
    if basename:
        name = os.path.basename(name)
    if lower:
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', name)]
    return [int(part) if part.isdigit() else part
            for part in re.split(r'(\d+)', name)]


def list_images(dirpath, exts=('.png', '.jpg', '.jpeg', '.bmp'), basename=True):
    """列出目录下的图片文件并按自然序排列。"""
    files = [f for f in os.listdir(dirpath) if f.lower().endswith(exts)]
    files.sort(key=lambda f: natural_sort_key(f, basename=basename))
    return files


def extract_frames(video_path, output_folder, ext='.png'):
    """把视频逐帧导出为图片，命名为 00001.ext、00002.ext……

    来源：offlineTool/视频转图片.py
    返回:
        实际导出的帧数
    """
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_filename = os.path.join(output_folder, f"{frame_count:05d}{ext}")
        cv2.imwrite(frame_filename, frame)

    cap.release()
    return frame_count


def frame2vid(src, vidDir, fps=30, size=(320, 240)):
    """把目录下的 PNG 按自然序合成为视频。

    来源：offlineTool/图片合成视频.py
    ⚠️ 原实现把分辨率**硬编码**为 (320, 240)，与源图不一致时会静默拉伸；
    这里保留同样的默认值，需要时显式传 size。
    """
    images = [img for img in os.listdir(src) if img.endswith(".png")]
    images.sort(key=lambda f: natural_sort_key(f, basename=False))

    video = cv2.VideoWriter(vidDir, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for image in images:
        video.write(cv2.imread(os.path.join(src, image)))

    cv2.destroyAllWindows()
    video.release()
    return len(images)


def video_to_images(video_path, img_dir='./video_frames', ext='.jpg'):
    """把视频拆成图片并返回自然序的文件名列表。

    来源：offlineEnhanced/superglue.py::video_to_images
    （那边是给 SuperGlue 配对用的，默认导出 jpg）
    """
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        cv2.imwrite(os.path.join(img_dir, f"{count:05d}{ext}"), frame)
    cap.release()

    return sorted(
        (os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(ext)),
        key=lambda p: natural_sort_key(p, basename=True)
    )
