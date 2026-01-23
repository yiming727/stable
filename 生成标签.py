import os

def generate_image_pairs(directory):
    # 获取目录中所有 .jpg 文件
    images = [img for img in os.listdir(directory) if img.endswith(".jpg")]

    # 对文件名进行自然排序，确保数字顺序正确
    images.sort(key=lambda f: int(f.split('.')[0]))

    # 遍历图像列表，生成配对
    for i in range(len(images) - 1):
        print(f"{images[i]} {images[i + 1]}")

if __name__ == '__main__':
    # 设置图像文件所在的目录
    image_directory = './Data/doudong6'  # 替换为你的实际路径

    # 调用生成配对的函数
    generate_image_pairs(image_directory)
