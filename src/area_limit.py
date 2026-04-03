"""
视频稳像算法遮挡鲁棒性测试工具（稳定性分数版）
核心指标：稳定性分数差值 = 稳像后分数 − 稳像前分数
  > 0 表示稳像有效（分数越高越好）
  = 0 表示无改善
  < 0 表示稳像引入了额外抖动
保留所有中间视频文件
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List
import json
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 遮挡生成器
# ============================================================================

class OcclusionGenerator:
    """遮挡生成器"""

    @staticmethod
    def apply_static_occlusion(frame, ratio, position='top'):
        occluded = frame.copy()
        h, w = frame.shape[:2]

        if position == 'top':
            mask_h = int(h * ratio)
            occluded[:mask_h, :] = 0
        elif position == 'bottom':
            mask_h = int(h * ratio)
            occluded[h - mask_h:, :] = 0
        elif position == 'left':
            mask_w = int(w * ratio)
            occluded[:, :mask_w] = 0
        elif position == 'right':
            mask_w = int(w * ratio)
            occluded[:, w - mask_w:] = 0
        elif position == 'center':
            total_area    = h * w
            occluded_area = total_area * ratio
            block_size    = int(np.sqrt(occluded_area))
            if block_size > h or block_size > w:
                if h < w:
                    block_size_h = min(block_size, h)
                    block_size_w = min(int(occluded_area / block_size_h), w)
                else:
                    block_size_w = min(block_size, w)
                    block_size_h = min(int(occluded_area / block_size_w), h)
            else:
                block_size_h = block_size_w = block_size
            y1 = max(0, (h - block_size_h) // 2)
            x1 = max(0, (w - block_size_w) // 2)
            y2 = min(h, y1 + block_size_h)
            x2 = min(w, x1 + block_size_w)
            occluded[y1:y2, x1:x2] = 0

        return occluded

    @staticmethod
    def apply_moving_occlusion(frame, ratio, position, velocity=(10, 5)):
        occluded = frame.copy()
        h, w = frame.shape[:2]

        total_area    = h * w
        occluded_area = total_area * ratio
        block_size    = int(np.sqrt(occluded_area))
        block_size    = min(block_size, h, w)

        x, y   = position
        vx, vy = velocity
        x += vx
        y += vy

        if x < 0 or x + block_size > w:
            vx = -vx
            x  = np.clip(x, 0, w - block_size)
        if y < 0 or y + block_size > h:
            vy = -vy
            y  = np.clip(y, 0, h - block_size)

        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + block_size)), min(h, int(y + block_size))
        occluded[y1:y2, x1:x2] = 0

        return occluded, [x, y], [vx, vy]


# ============================================================================
# 视频对比生成器
# ============================================================================

class VideoComparator:
    """视频对比生成器"""

    @staticmethod
    def create_side_by_side_video(occluded_path, stabilized_path,
                                  output_path, max_frames=None):
        cap_occl = cv2.VideoCapture(occluded_path)
        cap_stab = cv2.VideoCapture(stabilized_path)

        w   = int(cap_occl.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap_occl.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_occl.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

        frame_count = 0
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break
            ret_occl, frame_occl = cap_occl.read()
            ret_stab, frame_stab = cap_stab.read()
            if not (ret_occl and ret_stab):
                break
            canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
            canvas[:, 0:w]   = frame_occl
            canvas[:, w:2*w] = frame_stab
            out.write(canvas)
            frame_count += 1

        cap_occl.release()
        cap_stab.release()
        out.release()
        print(f"    对比视频已保存: {output_path}")


# ============================================================================
# 稳定性分数计算器
# ============================================================================

class StabilityScoreCalculator:
    """
    稳定性分数计算器

    核心指标：稳定性分数差值 diff = score_after − score_before
      diff > 0：稳像有效，数值越大改善越明显
      diff = 0：稳像无改善
      diff < 0：稳像引入了额外抖动，效果恶化

    稳定性分数定义：
      score = 1 / (1 + std(motion))
      motion_i = mean|frame_i − frame_{i-1}|（逐帧平均绝对差）
      取值 (0, 1]，越接近 1 表示帧间抖动越小
    """

    @staticmethod
    def compute_stability_score(video_path, max_frames=100):
        cap         = cv2.VideoCapture(video_path)
        prev_frame  = None
        motions     = []
        frame_count = 0

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                motion = np.mean(np.abs(gray.astype(float) - prev_frame.astype(float)))
                motions.append(motion)
            prev_frame  = gray
            frame_count += 1

        cap.release()

        if len(motions) == 0:
            return 1.0

        return 1.0 / (1.0 + float(np.std(motions)))

    @staticmethod
    def compute_stability_comparison(original_path, stabilized_path, max_frames=100):
        score_before = StabilityScoreCalculator.compute_stability_score(
            original_path,   max_frames
        )
        score_after  = StabilityScoreCalculator.compute_stability_score(
            stabilized_path, max_frames
        )
        diff = score_after - score_before

        return {
            'stability_score_before':        float(score_before),
            'stability_score_after':         float(score_after),
            'stability_diff':                float(diff),
            'stability_improvement':         float(diff),
            'stability_improvement_percent': float(diff * 100),
        }


# ============================================================================
# 遮挡鲁棒性测试器
# ============================================================================

class OcclusionRobustnessTest:
    """遮挡鲁棒性测试器"""

    def __init__(self, stabilizer, output_dir='./occlusion_test_results'):
        self.stabilizer    = stabilizer
        self.output_dir    = output_dir
        self.occlusion_gen = OcclusionGenerator()
        self.score_calc    = StabilityScoreCalculator()
        self.comparator    = VideoComparator()

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'occluded_videos'),   exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'stabilized_videos'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'comparison_videos'), exist_ok=True)

    # ------------------------------------------------------------------
    # ✅ 静态遮挡测试（统一为 top 位置，去掉 position 分类）
    # ------------------------------------------------------------------

    def test_static_occlusion(self,
                              video_path: str,
                              ratios: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4,
                                                     0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                              num_frames: int = 100):
        print("\n" + "=" * 70)
        print("测试静态遮挡对稳像质量的影响")
        print("=" * 70)
        print("核心指标: 稳定性分数差值 = 稳像后分数 − 稳像前分数")
        print("  差值 > 0 → 稳像有效   差值 < 0 → 稳像引入额外抖动")
        print("=" * 70)

        # ✅ 结果直接以 ratio 为键，不再嵌套 position
        results = {}

        for ratio in ratios:
            print(f"\n  遮挡比例: {ratio * 100:.0f}%")

            occluded_path = os.path.join(
                self.output_dir, 'occluded_videos',
                f'occluded_static_{int(ratio * 100)}percent.mp4'
            )
            print(f"    正在生成遮挡视频...")
            self._create_occluded_video(
                video_path, ratio, 'top', num_frames, occluded_path
            )
            print(f"    遮挡视频已保存: {occluded_path}")

            stabilized_path = os.path.join(
                self.output_dir, 'stabilized_videos',
                f'stabilized_static_{int(ratio * 100)}percent.mp4'
            )

            try:
                print(f"    正在稳像...")
                self.stabilizer.stabilize(occluded_path, stabilized_path)
                print(f"    稳像视频已保存: {stabilized_path}")

                print(f"    正在计算稳定性分数差值...")
                score_metrics = self.score_calc.compute_stability_comparison(
                    occluded_path, stabilized_path, max_frames=num_frames
                )

                is_success = score_metrics['stability_diff'] > 0

                comparison_path = os.path.join(
                    self.output_dir, 'comparison_videos',
                    f'comparison_static_{int(ratio * 100)}percent.mp4'
                )
                print(f"    正在生成对比视频...")
                self.comparator.create_side_by_side_video(
                    occluded_path, stabilized_path,
                    comparison_path, max_frames=num_frames
                )

                results[ratio] = {
                    'success':          is_success,
                    'metrics':          score_metrics,
                    'occluded_video':   occluded_path,
                    'stabilized_video': stabilized_path,
                    'comparison_video': comparison_path
                }

                m   = score_metrics
                tag = "✓ 稳像有效" if is_success else "✗ 稳像恶化"
                print(f"    {tag}")
                print(f"      稳像前分数:     {m['stability_score_before']:.4f}")
                print(f"      稳像后分数:     {m['stability_score_after']:.4f}")
                sign = "+" if m['stability_diff'] >= 0 else ""
                print(f"      分数差值(后−前): {sign}{m['stability_diff']:.4f}")

            except Exception as e:
                results[ratio] = {
                    'success':          False,
                    'error':            f'算法异常: {str(e)}',
                    'metrics':          None,
                    'occluded_video':   occluded_path,
                    'stabilized_video': None,
                    'comparison_video': None
                }
                print(f"    ✗ 算法异常: {e}")

        self._save_results(results, 'static_occlusion_test_stability.json')
        self._plot_static_results(results)
        return results

    # ------------------------------------------------------------------
    # 运动遮挡测试（不变）
    # ------------------------------------------------------------------

    def test_moving_occlusion(self,
                              video_path: str,
                              ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                              num_frames: int = 100):
        print("\n" + "=" * 70)
        print("测试运动遮挡对稳像质量的影响")
        print("=" * 70)
        print("核心指标: 稳定性分数差值 = 稳像后分数 − 稳像前分数")
        print("  差值 > 0 → 稳像有效   差值 < 0 → 稳像引入额外抖动")
        print("=" * 70)

        results = {}

        for ratio in ratios:
            print(f"\n遮挡比例: {ratio * 100:.0f}%")

            occluded_path = os.path.join(
                self.output_dir, 'occluded_videos',
                f'occluded_moving_{int(ratio * 100)}percent.mp4'
            )
            print(f"  正在生成运动遮挡视频...")
            self._create_moving_occluded_video(
                video_path, ratio, num_frames, occluded_path
            )
            print(f"  遮挡视频已保存: {occluded_path}")

            stabilized_path = os.path.join(
                self.output_dir, 'stabilized_videos',
                f'stabilized_moving_{int(ratio * 100)}percent.mp4'
            )

            try:
                print(f"  正在稳像...")
                self.stabilizer.stabilize(occluded_path, stabilized_path)
                print(f"  稳像视频已保存: {stabilized_path}")

                print(f"  正在计算稳定性分数差值...")
                score_metrics = self.score_calc.compute_stability_comparison(
                    occluded_path, stabilized_path, max_frames=num_frames
                )

                is_success = score_metrics['stability_diff'] > 0

                comparison_path = os.path.join(
                    self.output_dir, 'comparison_videos',
                    f'comparison_moving_{int(ratio * 100)}percent.mp4'
                )
                print(f"  正在生成对比视频...")
                self.comparator.create_side_by_side_video(
                    occluded_path, stabilized_path,
                    comparison_path, max_frames=num_frames
                )

                results[ratio] = {
                    'success':          is_success,
                    'metrics':          score_metrics,
                    'occluded_video':   occluded_path,
                    'stabilized_video': stabilized_path,
                    'comparison_video': comparison_path
                }

                m   = score_metrics
                tag = "✓ 稳像有效" if is_success else "✗ 稳像恶化"
                print(f"  {tag}")
                print(f"    稳像前分数:     {m['stability_score_before']:.4f}")
                print(f"    稳像后分数:     {m['stability_score_after']:.4f}")
                sign = "+" if m['stability_diff'] >= 0 else ""
                print(f"    分数差值(后−前): {sign}{m['stability_diff']:.4f}")

            except Exception as e:
                results[ratio] = {
                    'success':          False,
                    'error':            f'算法异常: {str(e)}',
                    'metrics':          None,
                    'occluded_video':   occluded_path,
                    'stabilized_video': None,
                    'comparison_video': None
                }
                print(f"  ✗ 算法异常: {e}")

        self._save_results(results, 'moving_occlusion_test_stability.json')
        self._plot_moving_results(results)
        return results

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _create_occluded_video(self, video_path, ratio, position, num_frames, output_path):
        cap    = cv2.VideoCapture(video_path)
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(self.occlusion_gen.apply_static_occlusion(frame, ratio, position))
        cap.release()
        out.release()

    def _create_moving_occluded_video(self, video_path, ratio, num_frames, output_path):
        cap      = cv2.VideoCapture(video_path)
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps      = cap.get(cv2.CAP_PROP_FPS)
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        out      = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        position = [w // 2, h // 2]
        velocity = [10, 5]
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            occluded, position, velocity = self.occlusion_gen.apply_moving_occlusion(
                frame, ratio, position, velocity
            )
            out.write(occluded)
        cap.release()
        out.release()

    def _save_results(self, results, filename):
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in results.items()},
                      f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")

    def _plot_static_results(self, results):
        """
        ✅ 静态遮挡可视化（统一，不再区分 position）
        左图：稳定性分数差值随遮挡比例变化，红色零线分隔正负，填色区分有效/恶化
        右图：成功/失败标志
        """
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        ratios = []
        diffs  = []
        flags  = []

        for ratio, result in sorted(results.items()):
            ratios.append(ratio * 100)
            if result.get('metrics'):
                diffs.append(result['metrics']['stability_diff'])
            else:
                diffs.append(0.0)
            flags.append(1 if result['success'] else 0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：差值曲线 + 填色
        axes[0].plot(ratios, diffs, 's-', color='steelblue',
                     linewidth=2, markersize=10, label='分数差值（后减前）')
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='零线')
        axes[0].fill_between(ratios, diffs, 0,
                             where=[d > 0 for d in diffs],
                             alpha=0.15, color='green', label='稳像有效区域')
        axes[0].fill_between(ratios, diffs, 0,
                             where=[d <= 0 for d in diffs],
                             alpha=0.15, color='red',   label='稳像恶化区域')
        axes[0].set_xlabel('遮挡比例 (%)', fontsize=12)
        axes[0].set_ylabel('稳定性分数差值（后减前）', fontsize=12)
        axes[0].set_title('稳定性分数差值 vs 静态遮挡比例', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # 右图：有效性标志
        axes[1].plot(ratios, flags, 's-', color='green',
                     linewidth=2, markersize=10)
        axes[1].axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
        axes[1].set_xlabel('遮挡比例 (%)', fontsize=12)
        axes[1].set_ylabel('稳像有效 (1=是, 0=否)', fontsize=12)
        axes[1].set_title('稳像有效性 vs 静态遮挡比例', fontsize=14, fontweight='bold')
        axes[1].set_ylim([-0.1, 1.1])
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, 'static_occlusion_stability_diff.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {out_path}")
        plt.close()

    def _plot_moving_results(self, results):
        """运动遮挡可视化（不变）"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        ratios = []
        diffs  = []
        flags  = []

        for ratio, result in sorted(results.items()):
            ratios.append(ratio * 100)
            if result.get('metrics'):
                diffs.append(result['metrics']['stability_diff'])
            else:
                diffs.append(0.0)
            flags.append(1 if result['success'] else 0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(ratios, diffs, 's-', color='steelblue',
                     linewidth=2, markersize=10, label='分数差值（后减前）')
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='零线')
        axes[0].fill_between(ratios, diffs, 0,
                             where=[d > 0 for d in diffs],
                             alpha=0.15, color='green', label='稳像有效区域')
        axes[0].fill_between(ratios, diffs, 0,
                             where=[d <= 0 for d in diffs],
                             alpha=0.15, color='red',   label='稳像恶化区域')
        axes[0].set_xlabel('遮挡比例 (%)', fontsize=12)
        axes[0].set_ylabel('稳定性分数差值（后减前）', fontsize=12)
        axes[0].set_title('稳定性分数差值 vs 运动遮挡比例', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ratios, flags, 's-', color='green',
                     linewidth=2, markersize=10)
        axes[1].axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
        axes[1].set_xlabel('遮挡比例 (%)', fontsize=12)
        axes[1].set_ylabel('稳像有效 (1=是, 0=否)', fontsize=12)
        axes[1].set_title('稳像有效性 vs 运动遮挡比例', fontsize=14, fontweight='bold')
        axes[1].set_ylim([-0.1, 1.1])
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, 'moving_occlusion_stability_diff.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {out_path}")
        plt.close()


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    from stable1 import VideoStabilizer

    stabilizer = VideoStabilizer(
        smoothing_radius=50,
        bucket_size=120,
        quality_level=0.01,
        min_distance=5,
        fb_threshold=1.0,
        min_inliers=8,
        fov_sigma=1000.0
    )

    tester = OcclusionRobustnessTest(
        stabilizer=stabilizer,
        output_dir='./occlusion_test_results'
    )

    video_path = './11.mp4'

    # 1. 静态遮挡测试（✅ 去掉 positions 参数）
    print("\n开始测试静态遮挡...")
    static_results = tester.test_static_occlusion(
        video_path=video_path,
        ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        num_frames=100
    )

    # 2. 运动遮挡测试
    print("\n开始测试运动遮挡...")
    moving_results = tester.test_moving_occlusion(
        video_path=video_path,
        ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        num_frames=100
    )

    # ── 汇总输出 ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("测试总结（核心指标：稳定性分数差值 = 稳像后 − 稳像前）")
    print("=" * 70)

    # ✅ 静态结果：直接遍历，不再按 position 分组
    print("\n静态遮挡:")
    failure_point = None
    for ratio, result in sorted(static_results.items()):
        m = result.get('metrics')
        if result['success']:
            diff_str = f"+{m['stability_diff']:.4f}" if m else "N/A"
            status   = f"✓ 有效  差值={diff_str}"
        else:
            if m:
                diff_str = f"{m['stability_diff']:.4f}"
                status   = f"✗ 恶化  差值={diff_str}"
            else:
                status   = f"✗ 异常  ({result.get('error', '')})"
            if failure_point is None:
                failure_point = ratio
        print(f"  {ratio * 100:3.0f}%: {status}")

    boundary = (str(int(failure_point * 100)) + '%') if failure_point else '未失效'
    print(f"  → 失效临界点: {boundary}")

    print("\n运动遮挡:")
    failure_point = None
    for ratio, result in sorted(moving_results.items()):
        m = result.get('metrics')
        if result['success']:
            diff_str = f"+{m['stability_diff']:.4f}" if m else "N/A"
            status   = f"✓ 有效  差值={diff_str}"
        else:
            if m:
                diff_str = f"{m['stability_diff']:.4f}"
                status   = f"✗ 恶化  差值={diff_str}"
            else:
                status   = f"✗ 异常  ({result.get('error', '')})"
            if failure_point is None:
                failure_point = ratio
        print(f"  {ratio * 100:3.0f}%: {status}")

    boundary = (str(int(failure_point * 100)) + '%') if failure_point else '未失效'
    print(f"  → 失效临界点: {boundary}")

    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print(f"结果保存在: ./occlusion_test_results/")
    print(f"  - 遮挡视频:  ./occlusion_test_results/occluded_videos/")
    print(f"  - 稳像视频:  ./occlusion_test_results/stabilized_videos/")
    print(f"  - 对比视频:  ./occlusion_test_results/comparison_videos/")
    print(f"  - 静态图表:  static_occlusion_stability_diff.png")
    print(f"  - 运动图表:  moving_occlusion_stability_diff.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
