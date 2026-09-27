"""
红外/可见光视频稳像项目的共享模块。

子模块按需导入——这里刻意**不**导入任何子模块，
否则 `import common.trajectory` 会连带拖入 cv2 / scipy / matplotlib，
让只想用一个纯 numpy 工具的脚本付出全部依赖代价。

    from common.trajectory import TrajectorySmoother
    from common.borders    import fix_border, build_transformation_matrix
    from common.features   import GridCalculator, fb_check, estimate_transform_ecc
    from common.shake      import detect_shake
    from common.image_io   import natural_sort_key, extract_frames, frame2vid

各模块的 docstring 里记录了它的来源（哪几个原文件重复实现了它）
以及各处**行为差异**是如何参数化的——重构时务必保持各调用点的原行为。
"""

__version__ = "0.1.0"
