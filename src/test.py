"""
视频稳像算法性能测试工具（无需 psutil）
测试指标:
1. SSIM (结构相似性) - 评估稳像质量
2. PSNR (峰值信噪比) - 评估图像保真度
3. 运行速度 (FPS) - 评估处理效率
4. 抖动减少率 - 评估稳像效果
"""

import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 质量评估指标
# ============================================================================

class QualityMetrics:
    """质量评估指标计算器"""

    @staticmethod
    def compute_ssim(img1, img2):
        """
        计算两张图像的 SSIM

        参数:
            img1: 第一张图像（灰度图）
            img2: 第二张图像（灰度图）

        返回:
            SSIM 值 (0-1)
        """
        try:
            from skimage.metrics import structural_similarity

            if img1.shape != img2.shape:
                # 调整尺寸
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            ssim_value = structural_similarity(img1, img2)
            return float(ssim_value)

        except ImportError:
            # 使用 OpenCV 实现
            return QualityMetrics.compute_ssim_opencv(img1, img2)

    @staticmethod
    def compute_ssim_opencv(img1, img2):
        """使用 OpenCV 计算 SSIM（备用方案）"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.mean(ssim_map))

    @staticmethod
    def compute_psnr(img1, img2):
        """
        计算两张图像的 PSNR

        参数:
            img1: 第一张图像
            img2: 第二张图像

        返回:
            PSNR 值（dB）
        """
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)

        if mse == 0:
            return 100.0

        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        return float(psnr)

    @staticmethod
    def compute_jitter(cap):
        """
        计算视频的抖动程度

        参数:
            cap: cv2.VideoCapture 对象

        返回:
            jitter: 抖动分数（越小越稳定）
        """
        prev_frame = None
        diffs = []

        frame_count = 0
        max_frames = 100  # 只评估前100帧

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

        if len(diffs) == 0:
            return 0.0

        # 抖动 = 帧间差异的标准差
        jitter = float(np.std(diffs))
        return jitter


# ============================================================================
# 性能测试器
# ============================================================================

class StabilizerPerformanceTester:
    """稳像算法性能测试器"""

    def __init__(self, stabilizer, output_dir='./performance_test_results'):
        """
        初始化测试器

        参数:
            stabilizer: VideoStabilizer 实例
            output_dir: 测试结果输出目录
        """
        self.stabilizer = stabilizer
        self.output_dir = output_dir
        self.metrics = QualityMetrics()

        os.makedirs(output_dir, exist_ok=True)

    def test_performance(self, video_path: str, num_frames: int = 100):
        """
        测试稳像算法的性能

        参数:
            video_path: 输入视频路径
            num_frames: 测试帧数

        返回:
            测试结果字典
        """
        print("\n" + "=" * 70)
        print("视频稳像算法性能测试")
        print("=" * 70)
        print(f"测试视频: {video_path}")
        print(f"测试帧数: {num_frames}")
        print("=" * 70)

        # 1. 运行稳像算法并测量时间
        print("\n步骤 1/4: 运行稳像算法...")
        stabilized_path = os.path.join(self.output_dir, 'stabilized_test.mp4')

        start_time = time.time()

        try:
            self.stabilizer.stabilize(video_path, stabilized_path)
        except Exception as e:
            print(f"❌ 稳像失败: {e}")
            import traceback
            traceback.print_exc()
            return None

        end_time = time.time()

        processing_time = end_time - start_time

        # 2. 计算 SSIM 和 PSNR
        print("\n步骤 2/4: 计算 SSIM 和 PSNR...")
        ssim_values, psnr_values = self._compute_quality_metrics(
            video_path, stabilized_path, num_frames
        )

        # 3. 计算抖动减少率
        print("\n步骤 3/4: 计算抖动减少率...")
        jitter_reduction = self._compute_jitter_reduction(
            video_path, stabilized_path
        )

        # 4. 计算处理速度
        print("\n步骤 4/4: 计算处理速度...")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_original = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        processing_fps = total_frames / processing_time
        speedup_ratio = processing_fps / fps_original

        # 计算文件大小
        original_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        stabilized_size = os.path.getsize(stabilized_path) / (1024 * 1024)  # MB

        # 汇总结果
        results = {
            'video_info': {
                'path': video_path,
                'total_frames': total_frames,
                'fps': fps_original,
                'tested_frames': num_frames,
                'original_size_mb': float(original_size),
                'stabilized_size_mb': float(stabilized_size)
            },
            'quality_metrics': {
                'avg_ssim': float(np.mean(ssim_values)),
                'std_ssim': float(np.std(ssim_values)),
                'min_ssim': float(np.min(ssim_values)),
                'max_ssim': float(np.max(ssim_values)),
                'avg_psnr': float(np.mean(psnr_values)),
                'std_psnr': float(np.std(psnr_values)),
                'min_psnr': float(np.min(psnr_values)),
                'max_psnr': float(np.max(psnr_values))
            },
            'jitter_metrics': {
                'original_jitter': jitter_reduction['original_jitter'],
                'stabilized_jitter': jitter_reduction['stabilized_jitter'],
                'jitter_reduction': jitter_reduction['jitter_reduction'],
                'jitter_reduction_percent': jitter_reduction['jitter_reduction'] * 100
            },
            'performance_metrics': {
                'processing_time': processing_time,
                'processing_fps': processing_fps,
                'original_fps': fps_original,
                'speedup_ratio': speedup_ratio,
                'realtime_capable': speedup_ratio >= 1.0,
                'time_per_frame': processing_time / total_frames
            }
        }

        # 打印结果
        self._print_results(results)

        # 保存结果
        self._save_results(results)

        # 可视化结果
        self._plot_results(results, ssim_values, psnr_values)

        return results

    def _compute_quality_metrics(self, original_path, stabilized_path, num_frames):
        """计算 SSIM 和 PSNR"""
        cap_orig = cv2.VideoCapture(original_path)
        cap_stab = cv2.VideoCapture(stabilized_path)

        ssim_values = []
        psnr_values = []

        for i in range(num_frames):
            ret1, frame_orig = cap_orig.read()
            ret2, frame_stab = cap_stab.read()

            if not (ret1 and ret2):
                break

            # 转换为灰度图
            gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
            gray_stab = cv2.cvtColor(frame_stab, cv2.COLOR_BGR2GRAY)

            # 计算 SSIM
            ssim = self.metrics.compute_ssim(gray_orig, gray_stab)
            ssim_values.append(ssim)

            # 计算 PSNR
            psnr = self.metrics.compute_psnr(gray_orig, gray_stab)
            psnr_values.append(psnr)

            if (i + 1) % 20 == 0:
                print(f"  已处理 {i + 1}/{num_frames} 帧")

        cap_orig.release()
        cap_stab.release()

        return np.array(ssim_values), np.array(psnr_values)

    def _compute_jitter_reduction(self, original_path, stabilized_path):
        """计算抖动减少率"""
        # 计算原始视频的抖动
        cap_orig = cv2.VideoCapture(original_path)
        original_jitter = self.metrics.compute_jitter(cap_orig)
        cap_orig.release()

        # 计算稳像视频的抖动
        cap_stab = cv2.VideoCapture(stabilized_path)
        stabilized_jitter = self.metrics.compute_jitter(cap_stab)
        cap_stab.release()

        # 计算减少率
        if original_jitter > 0:
            jitter_reduction = (original_jitter - stabilized_jitter) / original_jitter
        else:
            jitter_reduction = 0.0

        return {
            'original_jitter': float(original_jitter),
            'stabilized_jitter': float(stabilized_jitter),
            'jitter_reduction': float(jitter_reduction)
        }

    def _print_results(self, results):
        """打印测试结果"""
        print("\n" + "=" * 70)
        print("测试结果")
        print("=" * 70)

        # 视频信息
        print("\n【视频信息】")
        print(f"  视频路径: {results['video_info']['path']}")
        print(f"  总帧数: {results['video_info']['total_frames']}")
        print(f"  帧率: {results['video_info']['fps']:.2f} fps")
        print(f"  测试帧数: {results['video_info']['tested_frames']}")
        print(f"  原始大小: {results['video_info']['original_size_mb']:.2f} MB")
        print(f"  稳像大小: {results['video_info']['stabilized_size_mb']:.2f} MB")

        # 质量指标
        print("\n【质量指标】")
        print(f"  平均 SSIM: {results['quality_metrics']['avg_ssim']:.4f} "
              f"(范围: {results['quality_metrics']['min_ssim']:.4f} - "
              f"{results['quality_metrics']['max_ssim']:.4f})")
        print(f"  标准差 SSIM: {results['quality_metrics']['std_ssim']:.4f}")
        print(f"  平均 PSNR: {results['quality_metrics']['avg_psnr']:.2f} dB "
              f"(范围: {results['quality_metrics']['min_psnr']:.2f} - "
              f"{results['quality_metrics']['max_psnr']:.2f} dB)")
        print(f"  标准差 PSNR: {results['quality_metrics']['std_psnr']:.2f} dB")

        # 抖动指标
        print("\n【抖动指标】")
        print(f"  原始抖动: {results['jitter_metrics']['original_jitter']:.2f}")
        print(f"  稳像抖动: {results['jitter_metrics']['stabilized_jitter']:.2f}")
        print(f"  抖动减少: {results['jitter_metrics']['jitter_reduction_percent']:.1f}%")

        # 性能指标
        print("\n【性能指标】")
        print(f"  处理时间: {results['performance_metrics']['processing_time']:.2f} 秒")
        print(f"  处理速度: {results['performance_metrics']['processing_fps']:.2f} fps")
        print(f"  每帧耗时: {results['performance_metrics']['time_per_frame'] * 1000:.2f} ms")
        print(f"  加速比: {results['performance_metrics']['speedup_ratio']:.2f}x")
        print(f"  实时处理: {'✓ 是' if results['performance_metrics']['realtime_capable'] else '✗ 否'}")

        # 综合评价
        print("\n【综合评价】")
        avg_ssim = results['quality_metrics']['avg_ssim']
        jitter_reduction = results['jitter_metrics']['jitter_reduction']
        realtime = results['performance_metrics']['realtime_capable']

        if avg_ssim >= 0.85 and jitter_reduction >= 0.5 and realtime:
            grade = "优秀 ⭐⭐⭐⭐⭐"
            comment = "质量高，抖动减少明显，可实时处理"
        elif avg_ssim >= 0.75 and jitter_reduction >= 0.3:
            grade = "良好 ⭐⭐⭐⭐"
            comment = "质量较好，抖动减少显著"
        elif avg_ssim >= 0.65 and jitter_reduction >= 0.2:
            grade = "中等 ⭐⭐⭐"
            comment = "质量一般，有一定稳像效果"
        elif avg_ssim >= 0.50 and jitter_reduction >= 0.1:
            grade = "及格 ⭐⭐"
            comment = "质量偏低，稳像效果有限"
        else:
            grade = "不及格 ⭐"
            comment = "质量差，稳像效果不明显"

        print(f"  综合评分: {grade}")
        print(f"  评价: {comment}")

        print("=" * 70)

    def _save_results(self, results):
        """保存测试结果到 JSON 文件"""
        output_path = os.path.join(self.output_dir, 'performance_test_results.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: {output_path}")

    def _plot_results(self, results, ssim_values, psnr_values):
        """可视化测试结果"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig = plt.figure(figsize=(16, 10))

        # 1. SSIM 曲线
        ax1 = plt.subplot(2, 3, 1)
        frames = np.arange(len(ssim_values))
        ax1.plot(frames, ssim_values, 'b-', linewidth=1.5, alpha=0.7)
        ax1.axhline(y=results['quality_metrics']['avg_ssim'], color='r',
                    linestyle='--', label=f"平均值: {results['quality_metrics']['avg_ssim']:.4f}")
        ax1.set_xlabel('帧数', fontsize=11)
        ax1.set_ylabel('SSIM', fontsize=11)
        ax1.set_title('SSIM 随时间变化', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. PSNR 曲线
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(frames, psnr_values, 'g-', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=results['quality_metrics']['avg_psnr'], color='r',
                    linestyle='--', label=f"平均值: {results['quality_metrics']['avg_psnr']:.2f} dB")
        ax2.set_xlabel('帧数', fontsize=11)
        ax2.set_ylabel('PSNR (dB)', fontsize=11)
        ax2.set_title('PSNR 随时间变化', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. SSIM 分布直方图
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(ssim_values, bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax3.axvline(x=results['quality_metrics']['avg_ssim'], color='r',
                    linestyle='--', linewidth=2, label='平均值')
        ax3.set_xlabel('SSIM', fontsize=11)
        ax3.set_ylabel('频数', fontsize=11)
        ax3.set_title('SSIM 分布', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 抖动对比
        ax4 = plt.subplot(2, 3, 4)
        jitter_data = [
            results['jitter_metrics']['original_jitter'],
            results['jitter_metrics']['stabilized_jitter']
        ]
        bars = ax4.bar(['原始视频', '稳像视频'], jitter_data, color=['red', 'green'], alpha=0.7)
        ax4.set_ylabel('抖动程度', fontsize=11)
        ax4.set_title('抖动对比', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontsize=10)

        # 5. 性能指标
        ax5 = plt.subplot(2, 3, 5)
        perf_data = [
            results['performance_metrics']['original_fps'],
            results['performance_metrics']['processing_fps']
        ]
        bars = ax5.bar(['原始帧率', '处理速度'], perf_data, color=['blue', 'orange'], alpha=0.7)
        ax5.set_ylabel('FPS', fontsize=11)
        ax5.set_title('处理速度对比', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontsize=10)

        # 6. 综合指标雷达图
        ax6 = plt.subplot(2, 3, 6, projection='polar')

        # 归一化指标（0-1）
        ssim_norm = results['quality_metrics']['avg_ssim']
        psnr_norm = min(results['quality_metrics']['avg_psnr'] / 40, 1.0)  # 假设 40dB 为满分
        jitter_norm = results['jitter_metrics']['jitter_reduction']
        speed_norm = min(results['performance_metrics']['speedup_ratio'], 1.0)

        categories = ['SSIM', 'PSNR', '抖动减少', '处理速度']
        values = [ssim_norm, psnr_norm, jitter_norm, speed_norm]

        # 闭合雷达图
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax6.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax6.fill(angles, values, alpha=0.25, color='blue')
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories, fontsize=10)
        ax6.set_ylim(0, 1)
        ax6.set_title('综合性能雷达图', fontsize=13, fontweight='bold', pad=20)
        ax6.grid(True)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'performance_test_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
        plt.close()


# ============================================================================
# 批量测试（多个视频）
# ============================================================================

class BatchPerformanceTester:
    """批量性能测试器"""

    def __init__(self, stabilizer, output_dir='./batch_performance_test'):
        """
        初始化批量测试器

        参数:
            stabilizer: VideoStabilizer 实例
            output_dir: 测试结果输出目录
        """
        self.stabilizer = stabilizer
        self.output_dir = output_dir
        self.tester = StabilizerPerformanceTester(stabilizer, output_dir)

        os.makedirs(output_dir, exist_ok=True)

    def test_multiple_videos(self, video_paths: List[str], num_frames: int = 100):
        """
        测试多个视频

        参数:
            video_paths: 视频路径列表
            num_frames: 每个视频测试的帧数

        返回:
            所有测试结果的列表
        """
        print("\n" + "=" * 70)
        print(f"批量性能测试（共 {len(video_paths)} 个视频）")
        print("=" * 70)

        all_results = []

        for i, video_path in enumerate(video_paths, 1):
            print(f"\n\n{'=' * 70}")
            print(f"测试视频 {i}/{len(video_paths)}: {video_path}")
            print(f"{'=' * 70}")

            try:
                result = self.tester.test_performance(video_path, num_frames)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                import traceback
                traceback.print_exc()

        # 生成汇总报告
        self._generate_summary_report(all_results)

        return all_results

    def _generate_summary_report(self, all_results):
        """生成汇总报告"""
        if not all_results:
            print("\n❌ 没有成功的测试结果")
            return

        print("\n" + "=" * 70)
        print("汇总报告")
        print("=" * 70)

        # 计算平均值
        avg_ssim = np.mean([r['quality_metrics']['avg_ssim'] for r in all_results])
        avg_psnr = np.mean([r['quality_metrics']['avg_psnr'] for r in all_results])
        avg_jitter_reduction = np.mean([r['jitter_metrics']['jitter_reduction'] for r in all_results])
        avg_processing_fps = np.mean([r['performance_metrics']['processing_fps'] for r in all_results])
        avg_time = np.mean([r['performance_metrics']['processing_time'] for r in all_results])

        print(f"\n测试视频数: {len(all_results)}")
        print(f"\n平均质量指标:")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"  PSNR: {avg_psnr:.2f} dB")
        print(f"\n平均抖动减少: {avg_jitter_reduction:.1%}")
        print(f"\n平均处理速度: {avg_processing_fps:.2f} fps")
        print(f"平均处理时间: {avg_time:.2f} 秒")

        # 保存汇总结果
        summary = {
            'total_videos': len(all_results),
            'average_metrics': {
                'ssim': float(avg_ssim),
                'psnr': float(avg_psnr),
                'jitter_reduction': float(avg_jitter_reduction),
                'processing_fps': float(avg_processing_fps),
                'processing_time': float(avg_time)
            },
            'individual_results': all_results
        }

        output_path = os.path.join(self.output_dir, 'batch_summary.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n汇总报告已保存到: {output_path}")
        print("=" * 70)


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

    # ========== 单个视频测试 ==========
    print("\n" + "=" * 70)
    print("单个视频性能测试")
    print("=" * 70)

    tester = StabilizerPerformanceTester(
        stabilizer=stabilizer,
        output_dir='./performance_test_results'
    )

    # 测试单个视频
    video_path = './19.mp4'
    result = tester.test_performance(video_path, num_frames=100)

    # ========== 批量视频测试（可选）==========
    # 如果有多个视频需要测试
    """
    print("\n\n" + "=" * 70)
    print("批量视频性能测试")
    print("=" * 70)

    batch_tester = BatchPerformanceTester(
        stabilizer=stabilizer,
        output_dir='./batch_performance_test'
    )

    video_paths = [
        './video1.mp4',
        './video2.mp4',
        './video3.mp4'
    ]

    batch_results = batch_tester.test_multiple_videos(video_paths, num_frames=100)
    """

    print("\n✅ 测试完成！")


if __name__ == '__main__':
    main()
