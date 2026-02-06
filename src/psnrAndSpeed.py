"""
视频稳像算法性能测试工具
测试指标：PSNR（峰值信噪比）和运行速度
"""

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
from stable import VideoStabilizer  # 导入您的稳像器


# ============================================================================
# PSNR 计算
# ============================================================================

def calculate_psnr(img1, img2):
    """
    计算两幅图像之间的 PSNR（峰值信噪比）

    参数:
        img1: 第一幅图像
        img2: 第二幅图像

    返回:
        PSNR 值（dB），值越大表示图像质量越好
    """
    # 转换为灰度图
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 确保尺寸一致
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 计算 MSE（均方误差）
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)

    if mse == 0:
        return float('inf')

    # 计算 PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


def calculate_video_psnr(original_path, stabilized_path, max_frames=None):
    """
    计算两个视频之间的逐帧 PSNR

    参数:
        original_path: 原始视频路径
        stabilized_path: 稳像后视频路径
        max_frames: 最大评估帧数（None 表示全部）

    返回:
        psnr_list: PSNR 值列表
    """
    cap_orig = cv2.VideoCapture(original_path)
    cap_stab = cv2.VideoCapture(stabilized_path)

    psnr_list = []
    frame_idx = 0

    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_stab, frame_stab = cap_stab.read()

        if not ret_orig or not ret_stab:
            break

        if max_frames is not None and frame_idx >= max_frames:
            break

        # 计算 PSNR
        psnr = calculate_psnr(frame_orig, frame_stab)
        psnr_list.append(psnr)

        frame_idx += 1

    cap_orig.release()
    cap_stab.release()

    return psnr_list


# ============================================================================
# 性能测试
# ============================================================================

class PerformanceTester:
    """视频稳像性能测试器"""

    def __init__(self, stabilizer, output_dir='./test_results'):
        """
        初始化测试器

        参数:
            stabilizer: VideoStabilizer 实例
            output_dir: 测试结果输出目录
        """
        self.stabilizer = stabilizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _sanitize_filename(self, filename):
        """
        清理文件名，移除不合法字符

        参数:
            filename: 原始文件名

        返回:
            清理后的文件名
        """
        # 移除路径分隔符和其他不合法字符
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '.']
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename

    def test_single_video(self, video_path, video_name, max_frames=None):
        """
        测试单个视频的性能

        参数:
            video_path: 视频路径
            video_name: 视频名称（用于保存结果）
            max_frames: 最大处理帧数

        返回:
            results: 测试结果字典
        """
        print(f"\n{'=' * 70}")
        print(f"测试视频: {video_name}")
        print(f"{'=' * 70}")

        # 清理视频名称
        clean_video_name = self._sanitize_filename(video_name)

        # 输出路径
        output_path = os.path.join(
            self.output_dir,
            f'stabilized_{clean_video_name}.mp4'
        )

        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"✗ 无法打开视频: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        print(f"总帧数: {total_frames}")
        print(f"帧率: {fps:.2f} FPS")

        # 测试运行时间
        print("\n开始稳像处理...")
        start_time = time.time()

        try:
            self.stabilizer.stabilize(video_path, output_path)
            end_time = time.time()

            runtime = end_time - start_time
            processing_fps = total_frames / runtime
            avg_time_per_frame = runtime / total_frames * 1000  # ms

            print(f"\n✓ 稳像完成")
            print(f"  运行时间: {runtime:.2f} 秒")
            print(f"  处理速度: {processing_fps:.2f} FPS")
            print(f"  平均每帧: {avg_time_per_frame:.2f} ms")

        except Exception as e:
            print(f"\n✗ 稳像失败: {e}")
            import traceback
            traceback.print_exc()
            return None

        # 计算 PSNR
        print("\n计算 PSNR...")
        psnr_before = self._calculate_psnr_original(video_path, max_frames)
        psnr_after = calculate_video_psnr(video_path, output_path, max_frames)

        if len(psnr_before) == 0 or len(psnr_after) == 0:
            print("✗ PSNR 计算失败")
            return None

        # 汇总结果
        results = {
            'video_name': clean_video_name,
            'total_frames': total_frames,
            'runtime': runtime,
            'processing_fps': processing_fps,
            'avg_time_per_frame': avg_time_per_frame,
            'psnr_before': psnr_before,
            'psnr_after': psnr_after,
            'avg_psnr_before': np.mean(psnr_before),
            'avg_psnr_after': np.mean(psnr_after),
            'psnr_improvement': np.mean(psnr_after) - np.mean(psnr_before)
        }

        print(f"\nPSNR 统计:")
        print(f"  稳像前平均: {results['avg_psnr_before']:.2f} dB")
        print(f"  稳像后平均: {results['avg_psnr_after']:.2f} dB")
        print(f"  PSNR 提升: {results['psnr_improvement']:.2f} dB")

        return results

    def _calculate_psnr_original(self, video_path, max_frames=None):
        """
        计算原始视频的帧间 PSNR（相邻帧对比）

        参数:
            video_path: 视频路径
            max_frames: 最大评估帧数

        返回:
            psnr_list: PSNR 值列表
        """
        cap = cv2.VideoCapture(video_path)

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return []

        psnr_list = []
        frame_idx = 1

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            if max_frames is not None and frame_idx >= max_frames:
                break

            # 计算相邻帧的 PSNR
            psnr = calculate_psnr(prev_frame, curr_frame)
            psnr_list.append(psnr)

            prev_frame = curr_frame
            frame_idx += 1

        cap.release()
        return psnr_list

    def test_multiple_videos(self, video_list, max_frames=None):
        """
        测试多个视频样本

        参数:
            video_list: 视频路径列表 [(path1, name1), (path2, name2), ...]
            max_frames: 最大处理帧数

        返回:
            all_results: 所有测试结果列表
        """
        all_results = []

        for video_path, video_name in video_list:
            if not os.path.exists(video_path):
                print(f"\n警告: 视频文件不存在: {video_path}")
                continue

            result = self.test_single_video(video_path, video_name, max_frames)
            if result is not None:
                all_results.append(result)

        return all_results

    def plot_psnr_comparison(self, results, scenario_name=''):
        """
        绘制 PSNR 对比图（类似论文中的图）

        参数:
            results: 测试结果字典
            scenario_name: 场景名称
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(8, 5))

        # 绘制稳像前后的 PSNR 曲线
        frames_before = range(len(results['psnr_before']))
        frames_after = range(len(results['psnr_after']))

        ax.plot(frames_before, results['psnr_before'],
                label='稳像前', color='blue', linewidth=1.5, alpha=0.7)
        ax.plot(frames_after, results['psnr_after'],
                label='稳像后', color='red', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('图像帧数', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title(f'{scenario_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 保存图片（使用清理后的文件名）
        clean_video_name = self._sanitize_filename(results['video_name'])
        output_path = os.path.join(
            self.output_dir,
            f'psnr_{clean_video_name}.png'
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPSNR 图表已保存: {output_path}")
        plt.close()

    def plot_all_scenarios(self, all_results, scenario_names):
        """
        绘制所有场景的 PSNR 对比图（3x2 子图布局，支持6个样本）

        参数:
            all_results: 所有测试结果列表
            scenario_names: 场景名称列表
        """
        if len(all_results) == 0:
            print("警告: 没有可绘制的结果")
            return

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # 计算子图布局（3行2列，最多6个样本）
        n_results = len(all_results)
        n_cols = 2
        n_rows = (n_results + 1) // 2  # 向上取整

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))

        # 如果只有一个结果，axes 不是数组
        if n_results == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, result in enumerate(all_results):
            ax = axes[idx]

            # 获取场景名称
            if idx < len(scenario_names):
                scenario_name = scenario_names[idx]
            else:
                scenario_name = f'场景 {idx + 1}'

            frames_before = range(len(result['psnr_before']))
            frames_after = range(len(result['psnr_after']))

            ax.plot(frames_before, result['psnr_before'],
                    label='稳像前', color='blue', linewidth=1.2, alpha=0.7)
            ax.plot(frames_after, result['psnr_after'],
                    label='稳像后', color='red', linewidth=1.2, alpha=0.7)

            ax.set_xlabel('图像帧数', fontsize=11)
            ax.set_ylabel('PSNR (dB)', fontsize=11)
            ax.set_title(f'({chr(97 + idx)}) {scenario_name}',
                         fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(n_results, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'psnr_all_scenarios.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n综合 PSNR 图表已保存: {output_path}")
        plt.close()

    def generate_performance_table(self, all_results):
        """
        生成性能统计表格（类似论文中的表格）

        参数:
            all_results: 所有测试结果列表

        返回:
            表格字符串
        """
        if len(all_results) == 0:
            print("警告: 没有可生成的结果")
            return

        print("\n" + "=" * 110)
        print("性能统计表")
        print("=" * 110)

        # 表头
        header = f"{'指标':<20}"
        for idx in range(len(all_results)):
            header += f"{'样本' + str(idx + 1):<15}"
        header += f"{'均值':<15}"
        print(header)
        print("-" * 110)

        # 总帧数
        row = f"{'总帧数':<20}"
        total_frames_list = []
        for result in all_results:
            frames = result['total_frames']
            row += f"{frames:<15}"
            total_frames_list.append(frames)
        row += f"{int(np.mean(total_frames_list)):<15}"
        print(row)

        # 运行时间
        row = f"{'运行时间 (s)':<20}"
        runtime_list = []
        for result in all_results:
            runtime = result['runtime']
            row += f"{runtime:<15.2f}"
            runtime_list.append(runtime)
        row += f"{np.mean(runtime_list):<15.2f}"
        print(row)

        # 处理速度
        row = f"{'处理速度 (FPS)':<20}"
        fps_list = []
        for result in all_results:
            fps = result['processing_fps']
            row += f"{fps:<15.2f}"
            fps_list.append(fps)
        row += f"{np.mean(fps_list):<15.2f}"
        print(row)

        # 平均每帧
        row = f"{'平均每帧 (ms)':<20}"
        time_per_frame_list = []
        for result in all_results:
            time_per_frame = result['avg_time_per_frame']
            row += f"{time_per_frame:<15.2f}"
            time_per_frame_list.append(time_per_frame)
        row += f"{np.mean(time_per_frame_list):<15.2f}"
        print(row)

        print("=" * 110)

        # 保存到文件
        output_path = os.path.join(self.output_dir, 'performance_table.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 110 + "\n")
            f.write("性能统计表\n")
            f.write("=" * 110 + "\n")
            f.write(header + "\n")
            f.write("-" * 110 + "\n")

            # 重新生成每一行
            metrics = [
                ('总帧数', 'total_frames', lambda x: f"{int(x):<15}"),
                ('运行时间 (s)', 'runtime', lambda x: f"{x:<15.2f}"),
                ('处理速度 (FPS)', 'processing_fps', lambda x: f"{x:<15.2f}"),
                ('平均每帧 (ms)', 'avg_time_per_frame', lambda x: f"{x:<15.2f}")
            ]

            for metric_name, metric_key, formatter in metrics:
                row = f"{metric_name:<20}"
                values = []
                for result in all_results:
                    value = result[metric_key]
                    row += formatter(value)
                    values.append(value)
                row += formatter(np.mean(values))
                f.write(row + "\n")

            f.write("=" * 110 + "\n")

        print(f"\n性能表格已保存: {output_path}")


# ============================================================================
# 主测试函数
# ============================================================================

def main():
    """主测试函数"""
    print("=" * 70)
    print("视频稳像算法性能测试")
    print("=" * 70)

    # 创建稳像器
    stabilizer = VideoStabilizer(
        smoothing_radius=50,
        target_grid_size=100,
        min_grids=3,
        max_grids=8
    )

    # 创建测试器
    tester = PerformanceTester(
        stabilizer=stabilizer,
        output_dir='./performance_test_results'
    )

    # 定义测试视频列表（根据您的实际情况修改）
    video_list = [
        ('./1.mp4', 'sample1'),
        ('./2.mp4', 'sample2'),
        ('./3.mp4', 'sample3'),
        ('./4.mp4', 'sample4'),
        ('./5.mp4', 'sample5'),
        ('./6.mp4', 'sample6'),
    ]

    # 场景名称（用于图表标题）
    scenario_names = [
        '样本1',
        '样本2',
        '样本3',
        '样本4',
        '样本5',
        '样本6'
    ]

    # 测试多个视频（测试全部6个样本）
    print("\n开始批量测试...")
    all_results = tester.test_multiple_videos(
        video_list,  # 测试全部6个视频
        max_frames=300  # 每个视频最多处理300帧
    )

    if len(all_results) == 0:
        print("\n✗ 没有成功的测试结果")
        return

    # 为每个视频绘制单独的 PSNR 图
    print("\n生成 PSNR 对比图...")
    for idx, result in enumerate(all_results):
        if idx < len(scenario_names):
            scenario_name = scenario_names[idx]
        else:
            scenario_name = f'场景 {idx + 1}'
        tester.plot_psnr_comparison(result, scenario_name)

    # 绘制综合对比图（3x2 布局，6个样本）
    print("\n生成综合 PSNR 对比图...")
    tester.plot_all_scenarios(all_results, scenario_names)

    # 生成性能统计表
    print("\n生成性能统计表...")
    tester.generate_performance_table(all_results)

    print("\n" + "=" * 70)
    print("✅ 所有测试完成！")
    print(f"结果保存在: ./performance_test_results/")
    print("=" * 70)


if __name__ == '__main__':
    main()
