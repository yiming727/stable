"""
视频稳像算法遮挡鲁棒性测试工具（纯抖动检测版）
仅使用抖动减少率作为评估指标
抖动减少 > 0 即为成功
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
        """
        应用静态遮挡

        参数:
            frame: 输入帧
            ratio: 遮挡比例 (0.0-1.0)
            position: 遮挡位置 ('top', 'bottom', 'left', 'right', 'center')

        返回:
            occluded_frame: 带遮挡的帧
        """
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
            total_area = h * w
            occluded_area = total_area * ratio

            block_size = int(np.sqrt(occluded_area))

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
        """
        应用运动遮挡

        参数:
            frame: 输入帧
            ratio: 遮挡比例
            position: 遮挡块位置 [x, y]
            velocity: 速度 [vx, vy]

        返回:
            occluded_frame: 带遮挡的帧
            new_position: 更新后的位置
            new_velocity: 更新后的速度
        """
        occluded = frame.copy()
        h, w = frame.shape[:2]

        total_area = h * w
        occluded_area = total_area * ratio
        block_size = int(np.sqrt(occluded_area))
        block_size = min(block_size, h, w)

        x, y = position
        vx, vy = velocity
        x += vx
        y += vy

        if x < 0 or x + block_size > w:
            vx = -vx
            x = np.clip(x, 0, w - block_size)
        if y < 0 or y + block_size > h:
            vy = -vy
            y = np.clip(y, 0, h - block_size)

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
        """
        创建遮挡视频和稳像视频并排对比（无文字标签）

        参数:
            occluded_path: 遮挡视频路径
            stabilized_path: 稳像视频路径
            output_path: 输出视频路径
            max_frames: 最大帧数
        """
        cap_occl = cv2.VideoCapture(occluded_path)
        cap_stab = cv2.VideoCapture(stabilized_path)

        # 获取视频属性
        w = int(cap_occl.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_occl.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_occl.get(cv2.CAP_PROP_FPS)

        # 计算输出视频尺寸（两个视频横向排列，无标签区域）
        output_w = w * 2
        output_h = h

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))

        frame_count = 0

        while True:
            if max_frames is not None and frame_count >= max_frames:
                break

            ret_occl, frame_occl = cap_occl.read()
            ret_stab, frame_stab = cap_stab.read()

            if not (ret_occl and ret_stab):
                break

            # 创建画布（简单拼接）
            canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)

            # 放置两个视频帧（左右拼接）
            canvas[:, 0:w] = frame_occl
            canvas[:, w:2 * w] = frame_stab

            out.write(canvas)
            frame_count += 1

        cap_occl.release()
        cap_stab.release()
        out.release()

        print(f"    对比视频已保存: {output_path}")


# ============================================================================
# 抖动检测器
# ============================================================================

class JitterDetector:
    """视频抖动检测器"""

    @staticmethod
    def compute_jitter(video_path, max_frames=100):
        """
        计算视频的抖动程度

        参数:
            video_path: 视频路径
            max_frames: 最大评估帧数

        返回:
            jitter: 抖动分数（越小越稳定）
        """
        cap = cv2.VideoCapture(video_path)

        prev_frame = None
        diffs = []

        frame_count = 0

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # 计算帧间差异
                diff = np.mean(np.abs(gray.astype(float) - prev_frame.astype(float)))
                diffs.append(diff)

            prev_frame = gray
            frame_count += 1

        cap.release()

        if len(diffs) == 0:
            return 0.0

        # 抖动 = 帧间差异的标准差
        jitter = float(np.std(diffs))
        return jitter

    @staticmethod
    def compute_jitter_reduction(original_path, stabilized_path, max_frames=100):
        """
        计算抖动减少率

        参数:
            original_path: 原始视频路径
            stabilized_path: 稳像视频路径
            max_frames: 最大评估帧数

        返回:
            字典包含原始抖动、稳像抖动和减少率
        """
        # 计算原始视频的抖动
        original_jitter = JitterDetector.compute_jitter(original_path, max_frames)

        # 计算稳像视频的抖动
        stabilized_jitter = JitterDetector.compute_jitter(stabilized_path, max_frames)

        # 计算减少率
        if original_jitter > 0:
            jitter_reduction = (original_jitter - stabilized_jitter) / original_jitter
        else:
            jitter_reduction = 0.0

        return {
            'original_jitter': float(original_jitter),
            'stabilized_jitter': float(stabilized_jitter),
            'jitter_reduction': float(jitter_reduction),
            'jitter_reduction_percent': float(jitter_reduction * 100)
        }


# ============================================================================
# 遮挡鲁棒性测试器（纯抖动检测版）
# ============================================================================

class OcclusionRobustnessTest:
    """遮挡鲁棒性测试器（仅使用抖动检测）"""

    def __init__(self, stabilizer, output_dir='./occlusion_test_results'):
        """
        初始化测试器

        参数:
            stabilizer: VideoStabilizer 实例
            output_dir: 测试结果输出目录
        """
        self.stabilizer = stabilizer
        self.output_dir = output_dir
        self.occlusion_gen = OcclusionGenerator()
        self.jitter_detector = JitterDetector()
        self.comparator = VideoComparator()

        # 创建子目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'occluded_videos'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'stabilized_videos'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'comparison_videos'), exist_ok=True)

    def test_static_occlusion(self,
                              video_path: str,
                              ratios: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                              positions: List[str] = ['top', 'center'],
                              num_frames: int = 100):
        """
        测试静态遮挡的影响（仅使用抖动检测）

        参数:
            video_path: 输入视频路径
            ratios: 遮挡比例列表
            positions: 遮挡位置列表
            num_frames: 测试帧数

        返回:
            测试结果字典
        """
        print("\n" + "=" * 70)
        print("测试静态遮挡对稳像质量的影响（纯抖动检测）")
        print("=" * 70)
        print("评判标准: 抖动减少 > 0 即为成功")
        print("保留所有中间视频文件")
        print("=" * 70)

        results = {}

        for position in positions:
            print(f"\n测试位置: {position}")
            results[position] = {}

            for ratio in ratios:
                print(f"\n  遮挡比例: {ratio * 100:.0f}%")

                # 生成带遮挡的视频（保存到 occluded_videos 目录）
                occluded_path = os.path.join(
                    self.output_dir,
                    'occluded_videos',
                    f'occluded_{position}_{int(ratio * 100)}percent.mp4'
                )

                print(f"    正在生成遮挡视频...")
                self._create_occluded_video(
                    video_path, ratio, position, num_frames, occluded_path
                )
                print(f"    遮挡视频已保存: {occluded_path}")

                # 稳像后视频路径
                stabilized_path = os.path.join(
                    self.output_dir,
                    'stabilized_videos',
                    f'stabilized_{position}_{int(ratio * 100)}percent.mp4'
                )

                try:
                    # 稳像处理
                    print(f"    正在稳像...")
                    self.stabilizer.stabilize(occluded_path, stabilized_path)
                    print(f"    稳像视频已保存: {stabilized_path}")

                    # 评估抖动减少率
                    print(f"    正在评估抖动...")
                    jitter_metrics = self.jitter_detector.compute_jitter_reduction(
                        occluded_path, stabilized_path, max_frames=num_frames
                    )

                    # 判断成功：抖动减少 > 0
                    is_success = jitter_metrics['jitter_reduction'] > 0

                    # 创建对比视频（简单拼接，无文字）
                    comparison_path = os.path.join(
                        self.output_dir,
                        'comparison_videos',
                        f'comparison_{position}_{int(ratio * 100)}percent.mp4'
                    )

                    print(f"    正在生成对比视频...")
                    self.comparator.create_side_by_side_video(
                        occluded_path, stabilized_path,
                        comparison_path, max_frames=num_frames
                    )

                    results[position][ratio] = {
                        'success': is_success,
                        'metrics': jitter_metrics,
                        'occluded_video': occluded_path,
                        'stabilized_video': stabilized_path,
                        'comparison_video': comparison_path
                    }

                    if is_success:
                        print(f"    ✓ 稳像成功")
                        print(f"      原始抖动: {jitter_metrics['original_jitter']:.2f}")
                        print(f"      稳像抖动: {jitter_metrics['stabilized_jitter']:.2f}")
                        print(f"      抖动减少: {jitter_metrics['jitter_reduction_percent']:.1f}%")
                    else:
                        print(f"    ✗ 稳像失败")
                        print(f"      原始抖动: {jitter_metrics['original_jitter']:.2f}")
                        print(f"      稳像抖动: {jitter_metrics['stabilized_jitter']:.2f}")
                        print(f"      抖动减少: {jitter_metrics['jitter_reduction_percent']:.1f}% (未减少)")

                except Exception as e:
                    results[position][ratio] = {
                        'success': False,
                        'error': f'算法异常: {str(e)}',
                        'metrics': None,
                        'occluded_video': occluded_path,
                        'stabilized_video': None,
                        'comparison_video': None
                    }
                    print(f"    ✗ 稳像失败: 算法异常 - {e}")

        # 保存结果
        self._save_results(results, 'static_occlusion_test_jitter_only.json')

        # 可视化结果
        self._plot_static_results(results)

        return results

    def test_moving_occlusion(self,
                              video_path: str,
                              ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                              num_frames: int = 100):
        """
        测试运动遮挡的影响（仅使用抖动检测）

        参数:
            video_path: 输入视频路径
            ratios: 遮挡比例列表
            num_frames: 测试帧数

        返回:
            测试结果字典
        """
        print("\n" + "=" * 70)
        print("测试运动遮挡对稳像质量的影响（纯抖动检测）")
        print("=" * 70)
        print("评判标准: 抖动减少 > 0 即为成功")
        print("保留所有中间视频文件")
        print("=" * 70)

        results = {}

        for ratio in ratios:
            print(f"\n遮挡比例: {ratio * 100:.0f}%")

            # 生成带运动遮挡的视频（保存到 occluded_videos 目录）
            occluded_path = os.path.join(
                self.output_dir,
                'occluded_videos',
                f'occluded_moving_{int(ratio * 100)}percent.mp4'
            )

            print(f"  正在生成运动遮挡视频...")
            self._create_moving_occluded_video(
                video_path, ratio, num_frames, occluded_path
            )
            print(f"  遮挡视频已保存: {occluded_path}")

            # 稳像后视频路径
            stabilized_path = os.path.join(
                self.output_dir,
                'stabilized_videos',
                f'stabilized_moving_{int(ratio * 100)}percent.mp4'
            )

            try:
                # 稳像处理
                print(f"  正在稳像...")
                self.stabilizer.stabilize(occluded_path, stabilized_path)
                print(f"  稳像视频已保存: {stabilized_path}")

                # 评估抖动减少率
                print(f"  正在评估抖动...")
                jitter_metrics = self.jitter_detector.compute_jitter_reduction(
                    occluded_path, stabilized_path, max_frames=num_frames
                )

                # 判断成功：抖动减少 > 0
                is_success = jitter_metrics['jitter_reduction'] > 0

                # 创建对比视频（简单拼接，无文字）
                comparison_path = os.path.join(
                    self.output_dir,
                    'comparison_videos',
                    f'comparison_moving_{int(ratio * 100)}percent.mp4'
                )

                print(f"  正在生成对比视频...")
                self.comparator.create_side_by_side_video(
                    occluded_path, stabilized_path,
                    comparison_path, max_frames=num_frames
                )

                results[ratio] = {
                    'success': is_success,
                    'metrics': jitter_metrics,
                    'occluded_video': occluded_path,
                    'stabilized_video': stabilized_path,
                    'comparison_video': comparison_path
                }

                if is_success:
                    print(f"  ✓ 稳像成功")
                    print(f"    原始抖动: {jitter_metrics['original_jitter']:.2f}")
                    print(f"    稳像抖动: {jitter_metrics['stabilized_jitter']:.2f}")
                    print(f"    抖动减少: {jitter_metrics['jitter_reduction_percent']:.1f}%")
                else:
                    print(f"  ✗ 稳像失败")
                    print(f"    原始抖动: {jitter_metrics['original_jitter']:.2f}")
                    print(f"    稳像抖动: {jitter_metrics['stabilized_jitter']:.2f}")
                    print(f"    抖动减少: {jitter_metrics['jitter_reduction_percent']:.1f}% (未减少)")

            except Exception as e:
                results[ratio] = {
                    'success': False,
                    'error': f'算法异常: {str(e)}',
                    'metrics': None,
                    'occluded_video': occluded_path,
                    'stabilized_video': None,
                    'comparison_video': None
                }
                print(f"  ✗ 稳像失败: 算法异常 - {e}")

        # 保存结果
        self._save_results(results, 'moving_occlusion_test_jitter_only.json')

        # 可视化结果
        self._plot_moving_results(results)

        return results

    def _create_occluded_video(self, video_path, ratio, position, num_frames, output_path):
        """创建带静态遮挡的视频"""
        cap = cv2.VideoCapture(video_path)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            occluded = self.occlusion_gen.apply_static_occlusion(
                frame, ratio, position
            )
            out.write(occluded)

        cap.release()
        out.release()

    def _create_moving_occluded_video(self, video_path, ratio, num_frames, output_path):
        """创建带运动遮挡的视频"""
        cap = cv2.VideoCapture(video_path)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        position = [w // 2, h // 2]
        velocity = [10, 5]

        for i in range(num_frames):
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
        """保存测试结果到 JSON 文件"""
        output_path = os.path.join(self.output_dir, filename)

        serializable_results = {}
        for key, value in results.items():
            if isinstance(key, (int, float)):
                key = str(key)
            serializable_results[key] = value

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: {output_path}")

    def _plot_static_results(self, results):
        """可视化静态遮挡测试结果"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for position, data in results.items():
            ratios = []
            jitter_reductions = []
            success_flags = []

            for ratio, result in sorted(data.items()):
                ratios.append(ratio * 100)

                if result.get('metrics'):
                    jitter_reductions.append(result['metrics']['jitter_reduction_percent'])
                else:
                    jitter_reductions.append(0)

                success_flags.append(1 if result['success'] else 0)

            # 抖动减少率曲线
            axes[0].plot(ratios, jitter_reductions, 'o-', label=position,
                         linewidth=2, markersize=8)
            axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5,
                            linewidth=2, label='成功阈值 (0%)')
            axes[0].set_xlabel('遮挡比例 (%)', fontsize=12)
            axes[0].set_ylabel('抖动减少率 (%)', fontsize=12)
            axes[0].set_title('抖动减少率 vs 遮挡比例', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 成功率
            axes[1].plot(ratios, success_flags, 's-', label=position,
                         linewidth=2, markersize=8)
            axes[1].set_xlabel('遮挡比例 (%)', fontsize=12)
            axes[1].set_ylabel('成功 (1=是, 0=否)', fontsize=12)
            axes[1].set_title('稳像成功率', fontsize=14, fontweight='bold')
            axes[1].set_ylim([-0.1, 1.1])
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'static_occlusion_jitter_only.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
        plt.close()

    def _plot_moving_results(self, results):
        """可视化运动遮挡测试结果"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        ratios = []
        jitter_reductions = []
        success_flags = []

        for ratio, result in sorted(results.items()):
            ratios.append(ratio * 100)

            if result.get('metrics'):
                jitter_reductions.append(result['metrics']['jitter_reduction_percent'])
            else:
                jitter_reductions.append(0)

            success_flags.append(1 if result['success'] else 0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 抖动减少率
        axes[0].plot(ratios, jitter_reductions, 'o-', color='blue',
                     linewidth=2, markersize=10)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5,
                        linewidth=2, label='成功阈值 (0%)')
        axes[0].set_xlabel('遮挡比例 (%)', fontsize=12)
        axes[0].set_ylabel('抖动减少率 (%)', fontsize=12)
        axes[0].set_title('抖动减少率 vs 运动遮挡', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 成功率
        axes[1].plot(ratios, success_flags, 's-', color='green',
                     linewidth=2, markersize=10)
        axes[1].set_xlabel('遮挡比例 (%)', fontsize=12)
        axes[1].set_ylabel('成功 (1=是, 0=否)', fontsize=12)
        axes[1].set_title('稳像成功率', fontsize=14, fontweight='bold')
        axes[1].set_ylim([-0.1, 1.1])
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'moving_occlusion_jitter_only.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
        plt.close()


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主测试函数"""
    # 导入稳像器
    from stable import VideoStabilizer

    # 创建稳像器实例
    stabilizer = VideoStabilizer(
        smoothing_radius=50,
        target_grid_size=100,
        min_grids=3,
        max_grids=8
    )

    # 创建测试器（纯抖动检测）
    tester = OcclusionRobustnessTest(
        stabilizer=stabilizer,
        output_dir='./occlusion_test_results'
    )

    # 测试视频路径
    video_path = './11.mp4'

    # 1. 测试静态遮挡
    print("\n开始测试静态遮挡...")
    static_results = tester.test_static_occlusion(
        video_path=video_path,
        ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        positions=['top', 'center'],
        num_frames=100
    )

    # 2. 测试运动遮挡
    print("\n开始测试运动遮挡...")
    moving_results = tester.test_moving_occlusion(
        video_path=video_path,
        ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        num_frames=100
    )

    # 3. 分析结果
    print("\n" + "=" * 70)
    print("测试总结（纯抖动检测）")
    print("=" * 70)

    for position, data in static_results.items():
        print(f"\n{position} 位置静态遮挡:")
        failure_point = None

        for ratio, result in sorted(data.items()):
            if result['success']:
                status = "✓ 成功"
                if result.get('metrics'):
                    m = result['metrics']
                    status += f" (抖动减少 {m['jitter_reduction_percent']:.1f}%)"
            else:
                status = "✗ 失败"
                if result.get('metrics'):
                    m = result['metrics']
                    status += f" (抖动减少 {m['jitter_reduction_percent']:.1f}%)"
                elif result.get('error'):
                    status += f" ({result['error']})"

                if failure_point is None:
                    failure_point = ratio

            print(f"  {ratio * 100:3.0f}%: {status}")

        if failure_point is not None:
            print(f"  → 失效临界点: {failure_point * 100:.0f}%")
        else:
            print(f"  → 所有测试通过")

    print("\n运动遮挡:")
    failure_point = None
    for ratio, result in sorted(moving_results.items()):
        if result['success']:
            status = "✓ 成功"
            if result.get('metrics'):
                m = result['metrics']
                status += f" (抖动减少 {m['jitter_reduction_percent']:.1f}%)"
        else:
            status = "✗ 失败"
            if result.get('metrics'):
                m = result['metrics']
                status += f" (抖动减少 {m['jitter_reduction_percent']:.1f}%)"
            elif result.get('error'):
                status += f" ({result['error']})"

            if failure_point is None:
                failure_point = ratio

        print(f"  {ratio * 100:3.0f}%: {status}")

    if failure_point is not None:
        print(f"  → 失效临界点: {failure_point * 100:.0f}%")
    else:
        print(f"  → 所有测试通过")

    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print(f"结果保存在: ./occlusion_test_results/")
    print(f"  - 遮挡视频: ./occlusion_test_results/occluded_videos/")
    print(f"  - 稳像视频: ./occlusion_test_results/stabilized_videos/")
    print(f"  - 对比视频: ./occlusion_test_results/comparison_videos/")
    print("=" * 70)


if __name__ == '__main__':
    main()
