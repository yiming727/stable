"""
matplotlib 中文字体配置。

来源：约 15 处重复，两种写法混用：

    写法 A（rcParams 全局设置）——出现于
        online/basic/stable*.py（全部 5 个）
        offline/basic/src/detection.py、huatu1.py、huatu2.py、ssimAndPsnr.py
        offline/basic/u16/stable.py
        online/enhanced/method/*.py、method1/*.py
        offlineTool/PSNR.py
      用的是 plt.rcParams['font.sans-serif'] = ['SimHei']

    写法 B（显式字体文件）——出现于
        offline/basic/src/amplitude_limit.py
        offlineTool/画折线图.py
      用的是 FontProperties(fname="C:/Windows/Fonts/msyh.ttc")

两种都收录。⚠️ 不装 SimHei / 没有 msyh.ttc 的机器（尤其 Linux）上，
写法 A 会退化成方框、写法 B 会直接抛异常——所以 setup_cjk_font 里
保留了 DejaVu Sans 作为兜底字体名。
"""

import matplotlib
from matplotlib.font_manager import FontProperties

#: 按优先级排列的候选中文字体名
CJK_FONTS = ('SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS')

#: 写法 B 里硬编码的微软雅黑字体文件（仅 Windows）
MSYH_PATH = "C:/Windows/Fonts/msyh.ttc"


def setup_cjk_font(fonts=CJK_FONTS):
    """全局配置 matplotlib 以正确显示中文（同时修掉负号变方框的问题）。

    对应写法 A。在任何绘图之前调用一次即可。
    """
    matplotlib.rcParams['font.sans-serif'] = list(fonts)
    matplotlib.rcParams['axes.unicode_minus'] = False


def cjk_font_properties(path=MSYH_PATH):
    """返回指向具体字体文件的 FontProperties。

    对应写法 B，用于只想给单个元素指定字体的场景（不改全局状态）。
    """
    return FontProperties(fname=path)
