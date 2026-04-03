"""
运行速度测试
"""

import time
import numpy as np
import cv2
import argparse
from scipy.ndimage import gaussian_filter1d
import math

# ============================================================================
# 把你的视频稳定代码全部 import 进来（或直接粘贴在同一文件）
# 这里假设你的代码保存为 video_stabilizer.py
# ============================================================================
from stable1 import VideoStabilizer  # 如果是独立文件则用此行

# ============================================================================
# 运行速度测试类
# ============================================================================

class SpeedBenchmark:
    """
    测量 VideoStabilizer.stabilize() 的运行速度
    指标：
      - 总耗时 (s)
      - 处理速度 (fps)
      - 平均每帧耗时 (ms)
    """

    def __init__(self, input_path, output_path, num_runs=1, **stabilizer_kwargs):
        """
        参数:
            input_path:        待测试视频路径
            output_path:       输出视频路径（每次覆盖）
            num_runs:          重复测试次数，取均值（默认1次）
            stabilizer_kwargs: 传给 VideoStabilizer 的参数
        """
        self.input_path        = input_path
        self.output_path       = output_path
        self.num_runs          = num_runs
        self.stabilizer_kwargs = stabilizer_kwargs

    def _get_frame_count(self):
        cap = cv2.VideoCapture(self.input_path)
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return n, fps, w, h

    def run(self):
        n_frames, src_fps, w, h = self._get_frame_count()
        print("=" * 60)
        print("         视频稳定算法 运行速度测试")
        print("=" * 60)
        print(f"  视频分辨率 : {w} × {h}")
        print(f"  视频帧数   : {n_frames} 帧")
        print(f"  原始帧率   : {src_fps:.2f} fps")
        print(f"  重复次数   : {self.num_runs} 次")
        print("=" * 60)

        results = []

        for run_idx in range(self.num_runs):
            print(f"\n▶ 第 {run_idx + 1} / {self.num_runs} 次测试...")

            stabilizer = VideoStabilizer(**self.stabilizer_kwargs)

            t_start = time.perf_counter()
            stabilizer.stabilize(self.input_path, self.output_path)
            t_end   = time.perf_counter()

            elapsed      = t_end - t_start          # 总耗时 (s)
            proc_fps     = n_frames / elapsed        # 处理速度 (fps)
            ms_per_frame = elapsed / n_frames * 1000 # 每帧耗时 (ms)

            results.append({
                'elapsed'     : elapsed,
                'proc_fps'    : proc_fps,
                'ms_per_frame': ms_per_frame,
            })

            print(f"  ✅ 耗时        : {elapsed:.3f} s")
            print(f"  ✅ 处理速度    : {proc_fps:.2f} fps")
            print(f"  ✅ 平均每帧    : {ms_per_frame:.2f} ms")

        # ── 汇总统计 ──────────────────────────────────────────────────
        elapsed_arr  = np.array([r['elapsed']      for r in results])
        fps_arr      = np.array([r['proc_fps']      for r in results])
        ms_arr       = np.array([r['ms_per_frame']  for r in results])

        print("\n" + "=" * 60)
        print("                   汇总统计")
        print("=" * 60)
        print(f"  {'指标':<16} {'处理速度 (fps)':<20} {'每帧耗时 (ms)'}")
        print(f"  {'-'*56}")
        print(f"  {'均值':<16} {fps_arr.mean():<20.2f} {ms_arr.mean():.2f}")
        print(f"  {'标准差':<16} {fps_arr.std():<20.4f} {ms_arr.std():.4f}")
        print(f"  {'最小值':<16} {fps_arr.min():<20.2f} {ms_arr.min():.2f}")
        print(f"  {'最大值':<16} {fps_arr.max():<20.2f} {ms_arr.max():.2f}")
        print(f"  {'变异系数CV(%)':<16} {fps_arr.std()/fps_arr.mean()*100:<20.2f}"
              f" {ms_arr.std()/ms_arr.mean()*100:.2f}")
        print("=" * 60)

        # ── 实时性判断 ────────────────────────────────────────────────
        print("\n  实时性判断（基准: >25 fps / <40 ms）:")
        fps_ok = fps_arr.min() > 25
        ms_ok  = ms_arr.max()  < 40
        print(f"  处理速度最低值 {fps_arr.min():.2f} fps {'✅ 满足' if fps_ok else '❌ 不满足'} 实时基准(25fps)")
        print(f"  单帧最大耗时   {ms_arr.max():.2f} ms  {'✅ 满足' if ms_ok  else '❌ 不满足'} 实时基准(40ms)")
        print("=" * 60)

        return results


# ============================================================================
# 命令行入口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="视频稳定算法运行速度测试",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input',         type=str,   default='./8.mp4')
    parser.add_argument('--output',        type=str,   default='./result8.mp4')
    parser.add_argument('--num_runs',      type=int,   default=1,    help='重复测试次数')
    parser.add_argument('--smoothing',     type=int,   default=50)
    parser.add_argument('--bucket_size',   type=int,   default=120)
    parser.add_argument('--quality_level', type=float, default=0.01)
    parser.add_argument('--min_distance',  type=int,   default=30)
    parser.add_argument('--fb_threshold',  type=float, default=1.0)
    parser.add_argument('--min_inliers',   type=int,   default=10)
    parser.add_argument('--fov_sigma',     type=float, default=1000.0)
    return parser.parse_args()


def main():
    args = parse_args()

    bench = SpeedBenchmark(
        input_path    = args.input,
        output_path   = args.output,
        num_runs      = args.num_runs,
        smoothing_radius = args.smoothing,
        bucket_size      = args.bucket_size,
        quality_level    = args.quality_level,
        min_distance     = args.min_distance,
        fb_threshold     = args.fb_threshold,
        min_inliers      = args.min_inliers,
        fov_sigma        = args.fov_sigma,
    )
    bench.run()


if __name__ == '__main__':
    main()
