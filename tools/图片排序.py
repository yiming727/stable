import os
import re

def natural_sort_key(s):
    """
    自然排序关键字函数，用于将字符串中的数字作为整体进行排序
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def rename_images(folder_path, start_num):
    """
    重命名指定文件夹中的图像文件
    """
    file_list = os.listdir(folder_path)
    file_list = sorted([file for file in file_list if file.endswith('.png')], key=natural_sort_key) # 自然排序文件列表
    for i, file_name in enumerate(file_list, start=start_num):
        new_name = f'{i:05}.png'
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f'Renamed "{file_name}" to "{new_name}"')

folder_path = r'Data/train2'  # 指定文件夹路径
start_num = 1 # 指定起始数
rename_images(folder_path, start_num)
