import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Windows 下常用中文字体路径（微软雅黑）
font_path = "C:/Windows/Fonts/msyh.ttc"
font_prop = FontProperties(fname=font_path)

plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 示例数据
x = [3, 2, 1]  # 横坐标
y = [1.12, 1.05, 0.97] # 纵坐标

plt.plot(x, y, marker='o')  # marker='o' 表示每个点画一个圆圈

# 添加红色的纵坐标为1的横线
plt.axhline(y=1, color='red', linestyle='--', label='y=1')

plt.xlabel('像素亮度')
plt.ylabel('抑制比')
plt.title('折线图示例')
plt.grid(True)
plt.legend()
plt.show()
