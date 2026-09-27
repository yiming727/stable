import os
import cv2

# 输入文件夹路径
input_folder = '05'

# 输出文件夹路径
output_folder = 'output_folder'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有文件
file_list = os.listdir(input_folder)

# 遍历文件夹中的所有文件
for file_name in file_list:
    # 检查文件扩展名是否为 jpg
    if file_name.endswith('.jpg'):
        # 构建输入文件路径
        input_path = os.path.join(input_folder, file_name)

        # 构建输出文件路径
        output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.png')

        # 读取 JPG 图像
        jpg_image = cv2.imread(input_path)

        # 将 JPG 图像转换为 PNG 格式并保存
        cv2.imwrite(output_path, jpg_image)

print("文件夹内所有 JPG 图像已成功转换为 PNG 格式")
