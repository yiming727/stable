"""
视频稳像算法遮挡鲁棒性测试工具
测试不同遮挡比例和类型对稳像质量的影响
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json


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
            mask: 遮挡掩码 (1=有效区域, 0=遮挡区域)
        """
        occluded = frame.copy()
        h, w = frame.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8)

        if position == 'top':
            mask_h = int(h * ratio)
            occluded[:mask_h, :] = 0
            mask[:mask_h, :] = 0

        elif position == 'bottom':
            mask_h = int(h * ratio)
            occluded[h - mask_h:, :] = 0
            mask[h - mask_h:, :] = 0

        elif position == 'left':
            mask_w = int(w * ratio)
            occluded[:, :mask_w] = 0
            mask[:, :mask_w] = 0

        elif position == 'right':
            mask_w = int(w * ratio)
            occluded[:, w - mask_w:] = 0
            mask[:, w - mask_w:] = 0

        elif position == 'center':
            block_size = int(np.sqrt(h * w * ratio))
            y1 = (h - block_size) // 2
            x1 = (w - block_size) // 2
            y2 = y1 + block_size
            x2 = x1 + block_size
            occluded[y1:y2, x1:x2] = 0
            mask[y1:y2, x1:x2] = 0

        return occluded, mask

    @staticmethod
    def apply_moving_occlusion(frame, ratio, position, velocity=(10, 5)):
        """
        应用运动遮挡

        参数:
            frame: 输入帧
            ratio: 遮挡比例
            position: 遮挡块位置 [x, y]
            velocity: 速度 (vx, vy)

        返回:
            occluded_frame: 带遮挡的帧
            mask: 遮挡掩码
            new_position: 更新后的位置
        """
        occluded = frame.copy()
        h, w = frame.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8)

        # 计算遮挡块大小
        block_size = int(np.sqrt(h * w * ratio))

        # 更新位置
        x, y = position
        x += velocity[0]
        y += velocity[1]

        # 边界处理（反弹）
        if x < 0 or x + block_size > w:
            velocity = (-velocity[0], velocity[1])
            x = np.clip(x, 0, w - block_size)
        if y < 0 or y + block_size > h:
            velocity = (velocity[0], -velocity[1])
            y = np.clip(y, 0, h - block_size)

        # 应用遮挡
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + block_size)), min(h, int(y + block_size))
        occluded[y1:y2, x1:x2] = 0
        mask[y1:y2, x1:x2] = 0

        return occluded, mask, [x, y]


# ============================================================================
# 质量评估指标
# ============================================================================

class QualityMetrics:
    """稳像质量评估指标"""

    @staticmethod
    def compute_psnr(original, processed, mask=None):
        """
        计算 PSNR（峰值信噪比）

        参数:
            original: 原始帧
            processed: 处理后的帧
            mask: 有效区域掩码

        返回:
            PSNR 值（dB）
        """
        if mask is not None:
            original = original[mask > 0]
            processed = processed[mask > 0]

        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)

        if mse == 0:
            return float('inf')

        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        return psnr

    @staticmethod
    def compute_ssim(original, processed, mask=None):
        """
        计算 SSIM（结构相似性）

        参数:
            original: 原始帧（灰度图）
            processed: 处理后的帧（灰度图）
            mask: 有效区域掩码

        返回:
            SSIM 值 (0-1)
        """
        from skimage.metrics import structural_similarity

        if mask is not None:
            # 只在有效区域计算
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) == 0:
                return 0.0

            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            original = original[y_min:y_max + 1, x_min:x_max + 1]
            processed = processed[y_min:y_max + 1, x_min:x_max + 1]

        if original.size == 0 or processed.size == 0:
            return 0.0

        ssim_value = structural_similarity(original, processed)

        return ssim_value

    @staticmethod
    def compute_temporal_smoothness(transforms):
        """
        计算时序平滑度（相邻帧变换的变化程度）

        参数:
            transforms: shape=(N, 3) 的变换数组

        返回:
            平滑度分数（越小越平滑）
        """
        if len(transforms) < 2:
            return 0.0

        # 计算相邻帧变换的差异
        diff = np.diff(transforms, axis=0)

        # 计算标准差（越小越平滑）
        smoothness = np.mean(np.std(diff, axis=0))

        return smoothness

    @staticmethod
    def compute_drift(trajectory):
        """
        计算累积漂移（轨迹偏离原点的程度）

        参数:
            trajectory: shape=(N, 3) 的累积轨迹

        返回:
            漂移距离（像素）
        """
        if len(trajectory) == 0:
            return 0.0

        # 计算最后一帧相对于第一帧的位移
        final_pos = trajectory[-1, :2]
        drift = np.linalg.norm(final_pos)

        return drift


# ============================================================================
# 遮挡鲁棒性测试器
# ============================================================================

class OcclusionRobustnessTest:
    """遮挡鲁棒性测试器"""

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
        self.metrics = QualityMetrics()

        os.makedirs(output_dir, exist_ok=True)

    def test_static_occlusion(self,
                              video_path: str,
                              ratios: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                              positions: List[str] = ['top', 'center'],
                              num_frames: int = 100):
        """
        测试静态遮挡的影响

        参数:
            video_path: 输入视频路径
            ratios: 遮挡比例列表
            positions: 遮挡位置列表
            num_frames: 测试帧数

        返回:
            测试结果字典
        """
        print("\n" + "=" * 70)
        print("测试静态遮挡对稳像质量的影响")
        print("=" * 70)

        results = {}

        for position in positions:
            print(f"\n测试位置: {position}")
            results[position] = {}

            for ratio in ratios:
                print(f"\n  遮挡比例: {ratio * 100:.0f}%")

                # 生成带遮挡的视频
                occluded_path = self._create_occluded_video(
                    video_path, ratio, position, num_frames
                )

                # 运行稳像算法
                output_path = os.path.join(
                    self.output_dir,
                    f'stabilized_{position}_{int(ratio * 100)}percent.mp4'
                )

                try:
                    # 稳像处理
                    self.stabilizer.stabilize(occluded_path, output_path)

                    # 评估质量
                    metrics = self._evaluate_stabilization(
                        video_path, output_path, occluded_path, num_frames
                    )

                    results[position][ratio] = {
                        'success': True,
                        'metrics': metrics
                    }

                    print(f"    ✓ 稳像成功")
                    print(f"      PSNR: {metrics['psnr']:.2f} dB")
                    print(f"      SSIM: {metrics['ssim']:.4f}")
                    print(f"      平滑度: {metrics['smoothness']:.4f}")

                except Exception as e:
                    results[position][ratio] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"    ✗ 稳像失败: {e}")

                # 清理临时文件
                if os.path.exists(occluded_path):
                    os.remove(occluded_path)

        # 保存结果
        self._save_results(results, 'static_occlusion_test.json')

        # 可视化结果
        self._plot_static_results(results)

        return results

    def test_moving_occlusion(self,
                              video_path: str,
                              ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
                              num_frames: int = 100):
        """
        测试运动遮挡的影响

        参数:
            video_path: 输入视频路径
            ratios: 遮挡比例列表
            num_frames: 测试帧数

        返回:
            测试结果字典
        """
        print("\n" + "=" * 70)
        print("测试运动遮挡对稳像质量的影响")
        print("=" * 70)

        results = {}

        for ratio in ratios:
            print(f"\n遮挡比例: {ratio * 100:.0f}%")

            # 生成带运动遮挡的视频
            occluded_path = self._create_moving_occluded_video(
                video_path, ratio, num_frames
            )

            # 运行稳像算法
            output_path = os.path.join(
                self.output_dir,
                f'stabilized_moving_{int(ratio * 100)}percent.mp4'
            )

            try:
                # 稳像处理
                self.stabilizer.stabilize(occluded_path, output_path)

                # 评估质量
                metrics = self._evaluate_stabilization(
                    video_path, output_path, occluded_path, num_frames
                )

                results[ratio] = {
                    'success': True,
                    'metrics': metrics
                }

                print(f"  ✓ 稳像成功")
                print(f"    PSNR: {metrics['psnr']:.2f} dB")
                print(f"    SSIM: {metrics['ssim']:.4f}")

            except Exception as e:
                results[ratio] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"  ✗ 稳像失败: {e}")

            # 清理临时文件
            if os.path.exists(occluded_path):
                os.remove(occluded_path)

        # 保存结果
        self._save_results(results, 'moving_occlusion_test.json')

        # 可视化结果
        self._plot_moving_results(results)

        return results

    def _create_occluded_video(self, video_path, ratio, position, num_frames):
        """创建带静态遮挡的视频"""
        cap = cv2.VideoCapture(video_path)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        temp_path = os.path.join(
            self.output_dir,
            f'temp_occluded_{position}_{int(ratio * 100)}.mp4'
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            occluded, _ = self.occlusion_gen.apply_static_occlusion(
                frame, ratio, position
            )
            out.write(occluded)

        cap.release()
        out.release()

        return temp_path

    def _create_moving_occluded_video(self, video_path, ratio, num_frames):
        """创建带运动遮挡的视频"""
        cap = cv2.VideoCapture(video_path)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        temp_path = os.path.join(
            self.output_dir,
            f'temp_moving_occluded_{int(ratio * 100)}.mp4'
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

        # 初始化遮挡位置
        position = [w // 2, h // 2]
        velocity = [10, 5]

        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            occluded, _, position = self.occlusion_gen.apply_moving_occlusion(
                frame, ratio, position, velocity
            )
            out.write(occluded)

        cap.release()
        out.release()

        return temp_path

    def _evaluate_stabilization(self, original_path, stabilized_path,
                                occluded_path, num_frames):
        """评估稳像质量"""
        cap_orig = cv2.VideoCapture(original_path)
        cap_stab = cv2.VideoCapture(stabilized_path)
        cap_occ = cv2.VideoCapture(occluded_path)

        psnr_values = []
        ssim_values = []

        for i in range(min(num_frames, 50)):  # 只评估前50帧
            ret1, frame_orig = cap_orig.read()
            ret2, frame_stab = cap_stab.read()
            ret3, frame_occ = cap_occ.read()

            if not (ret1 and ret2 and ret3):
                break

            # 转换为灰度图
            gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
            gray_stab = cv2.cvtColor(frame_stab, cv2.COLOR_BGR2GRAY)

            # 生成有效区域掩码
            _, mask = self.occlusion_gen.apply_static_occlusion(
                frame_occ, 0.0, 'top'
            )
            mask = (frame_occ.sum(axis=2) > 0).astype(np.uint8)

            # 计算 PSNR
            psnr = self.metrics.compute_psnr(gray_orig, gray_stab, mask)
            psnr_values.append(psnr)

            # 计算 SSIM
            ssim = self.metrics.compute_ssim(gray_orig, gray_stab, mask)
            ssim_values.append(ssim)

        cap_orig.release()
        cap_stab.release()
        cap_occ.release()

        return {
            'psnr': np.mean(psnr_values) if psnr_values else 0.0,
            'ssim': np.mean(ssim_values) if ssim_values else 0.0,
            'smoothness': 0.0,  # 需要从稳像器获取
            'drift': 0.0
        }

    def _save_results(self, results, filename):
        """保存测试结果到 JSON 文件"""
        output_path = os.path.join(self.output_dir, filename)

        # 转换为可序列化的格式
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
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for position, data in results.items():
            ratios = []
            psnr_values = []
            ssim_values = []
            success_flags = []

            for ratio, result in sorted(data.items()):
                ratios.append(ratio * 100)

                if result['success']:
                    psnr_values.append(result['metrics']['psnr'])
                    ssim_values.append(result['metrics']['ssim'])
                    success_flags.append(1)
                else:
                    psnr_values.append(0)
                    ssim_values.append(0)
                    success_flags.append(0)

            # PSNR 曲线
            axes[0, 0].plot(ratios, psnr_values, 'o-', label=position)
            axes[0, 0].set_xlabel('Occlusion Ratio (%)')
            axes[0, 0].set_ylabel('PSNR (dB)')
            axes[0, 0].set_title('PSNR vs Occlusion Ratio')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # SSIM 曲线
            axes[0, 1].plot(ratios, ssim_values, 's-', label=position)
            axes[0, 1].set_xlabel('Occlusion Ratio (%)')
            axes[0, 1].set_ylabel('SSIM')
            axes[0, 1].set_title('SSIM vs Occlusion Ratio')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # 成功率
            axes[1, 0].plot(ratios, success_flags, '^-', label=position)
            axes[1, 0].set_xlabel('Occlusion Ratio (%)')
            axes[1, 0].set_ylabel('Success (1=Yes, 0=No)')
            axes[1, 0].set_title('Stabilization Success Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # 综合评分
        axes[1, 1].text(0.5, 0.5, 'Occlusion Robustness Test\nStatic Occlusion',
                        ha='center', va='center', fontsize=16)
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'static_occlusion_results.png'), dpi=300)
        print(f"图表已保存到: {self.output_dir}/static_occlusion_results.png")
        plt.close()

    def _plot_moving_results(self, results):
        """可视化运动遮挡测试结果"""
        ratios = []
        psnr_values = []
        ssim_values = []
        success_flags = []

        for ratio, result in sorted(results.items()):
            ratios.append(ratio * 100)

            if result['success']:
                psnr_values.append(result['metrics']['psnr'])
                ssim_values.append(result['metrics']['ssim'])
                success_flags.append(1)
            else:
                psnr_values.append(0)
                ssim_values.append(0)
                success_flags.append(0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # PSNR
        axes[0].plot(ratios, psnr_values, 'o-', color='blue')
        axes[0].set_xlabel('Occlusion Ratio (%)')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('PSNR vs Moving Occlusion')
        axes[0].grid(True)

        # SSIM
        axes[1].plot(ratios, ssim_values, 's-', color='green')
        axes[1].set_xlabel('Occlusion Ratio (%)')
        axes[1].set_ylabel('SSIM')
        axes[1].set_title('SSIM vs Moving Occlusion')
        axes[1].grid(True)

        # 成功率
        axes[2].plot(ratios, success_flags, '^-', color='red')
        axes[2].set_xlabel('Occlusion Ratio (%)')
        axes[2].set_ylabel('Success')
        axes[2].set_title('Success Rate')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'moving_occlusion_results.png'), dpi=300)
        print(f"图表已保存到: {self.output_dir}/moving_occlusion_results.png")
        plt.close()


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主测试函数"""
    # 导入您的稳像器
    from stable import VideoStabilizer  # 替换为实际的导入路径

    # 创建稳像器实例
    stabilizer = VideoStabilizer(
        smoothing_radius=50,
        target_grid_size=100,
        min_grids=3,
        max_grids=8
    )

    # 创建测试器
    tester = OcclusionRobustnessTest(
        stabilizer=stabilizer,
        output_dir='./occlusion_test_results'
    )

    # 测试视频路径
    video_path = './19.mp4'

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
        ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        num_frames=100
    )

    # 3. 分析结果
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    # 找到失效临界点
    for position, data in static_results.items():
        print(f"\n{position} 位置静态遮挡:")
        for ratio, result in sorted(data.items()):
            status = "✓ 成功" if result['success'] else "✗ 失败"
            print(f"  {ratio * 100:3.0f}%: {status}")

    print("\n运动遮挡:")
    for ratio, result in sorted(moving_results.items()):
        status = "✓ 成功" if result['success'] else "✗ 失败"
        print(f"  {ratio * 100:3.0f}%: {status}")

    print("\n✅ 测试完成！结果已保存到 ./occlusion_test_results/")


if __name__ == '__main__':
    main()
