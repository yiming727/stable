"""
稳像质量评估（薄壳）—— 目录级 CR / DV / SS 指标。

本文件与 offline/enhanced/evaluate.py 原本是**字节完全相同**的两份副本，
现已统一到 common/stability_metrics.py，此处只做转发，
保持 `from evaluate import compute_cropping_ratio` 这类既有用法不变。

三个函数签名与原来完全一致（含默认参数）。

⚠️ 注意：另有一个 `tools/metrics.py::VideoStabilityEvaluator`（490 行）算的是同一套
CR/DV/SS，但接口是"视频路径进、指标出"，与本文件的"目录进"不同，**没有**并入这里。
"""

import pathlib
import sys

# 定位仓库根（含 common/ 的那一层），使本文件从任意 cwd 运行都能 import common
for _p in pathlib.Path(__file__).resolve().parents:
    if (_p / "common" / "__init__.py").is_file():
        sys.path.insert(0, str(_p))
        break

import cv2  # noqa: E402

from common.stability_metrics import (  # noqa: E402
    compute_cropping_ratio,
    compute_distortion_value,
    compute_stability_score,
)

__all__ = [
    "compute_cropping_ratio",
    "compute_distortion_value",
    "compute_stability_score",
]


if __name__ == '__main__':
    original_dir = '../Data/2/'
    pred_dir = '../Data/result2/'

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher()

    avg_cr, min_cr = compute_cropping_ratio(original_dir, pred_dir, orb, bf)
    print('***Cropping ratio (Avg, Min):')
    print(f'{avg_cr:.4f} | {min_cr:.4f}')

    min_dv = compute_distortion_value(original_dir, pred_dir, orb, bf)
    print('***Distortion value:')
    print(f'{min_dv:.4f}')

    ss_avg_pred, ss_trans_pred, ss_rot_pred = compute_stability_score(pred_dir, orb, bf)
    ss_avg_orig, ss_trans_orig, ss_rot_orig = compute_stability_score(original_dir, orb, bf)

    print('***Stability Score Comparison:')
    print(f'Original  : {ss_avg_orig:.4f} | {ss_trans_orig:.4f} | {ss_rot_orig:.4f}')
    print(f'Predicted : {ss_avg_pred:.4f} | {ss_trans_pred:.4f} | {ss_rot_pred:.4f}')
    print(f'Improvement: {ss_avg_pred - ss_avg_orig:.4f} | {ss_trans_pred - ss_trans_orig:.4f} | {ss_rot_pred - ss_rot_orig:.4f}')
