"""
视频抖动检测（薄壳）。

本文件与 offline/enhanced/detection.py 原本是**字节完全相同**的两份副本，
现已统一到 common/shake.py，此处只做转发，保持 `from detection import detect_shake`
这类既有用法不变。

函数签名与原来一致；额外多了一个可选参数 `plot`（原本固定会画图并 show()）。

⚠️ 与合并前的一处行为差异：原文件在第 122 行**裸执行**
`detect_shake('../Data/test1.mp4')`，导致任何 import 都会立刻读视频、画图并阻塞。
现在那段演示只在本文件被直接运行时才执行（见文件末尾）。
"""

import pathlib
import sys

# 定位仓库根（含 common/ 的那一层），使本文件从任意 cwd 运行都能 import common
for _p in pathlib.Path(__file__).resolve().parents:
    if (_p / "common" / "__init__.py").is_file():
        sys.path.insert(0, str(_p))
        break

from common.shake import (  # noqa: E402  基础库导入必须在本文件顶部
    analyze_shake_per_frame,
    calculate_optical_flow_with_ransac,
    detect_peaks_and_widths,
    detect_shake,
)

__all__ = [
    "calculate_optical_flow_with_ransac",
    "detect_peaks_and_widths",
    "analyze_shake_per_frame",
    "detect_shake",
]


if __name__ == '__main__':
    detect_shake('../Data/test1.mp4')
