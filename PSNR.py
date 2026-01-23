import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class VideoStabilizationPSNR:
    """视频稳像前后PSNR测量与可视化工具"""

    def __init__(self):
        pass

    def calculate_psnr(self, img1, img2, max_pixel_value=255.0):
        """
        计算两帧之间的PSNR

        参数:
            img1: 参考帧
            img2: 对比帧
            max_pixel_value: 像素最大值

        返回:
            psnr: 峰值信噪比（dB）
        """
        # 确保图像尺寸一致
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # 转换为float类型
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        # 计算均方误差（MSE）
        mse = np.mean((img1 - img2) ** 2)

        # 如果MSE为0，说明两图像完全相同
        if mse < 1e-10:
            return 100.0  # 返回一个很高的值

        # 计算PSNR
        psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))

        return psnr

    def calculate_video_psnr(self, original_video_path, stabilized_video_path):
        """
        计算稳像前后视频的PSNR（相邻帧对比，评估时域稳定性）

        参数:
            original_video_path: 稳像前视频路径
            stabilized_video_path: 稳像后视频路径

        返回:
            frame_numbers: 帧序号列表
            psnr_before: 稳像前相邻帧PSNR列表
            psnr_after: 稳像后相邻帧PSNR列表
        """
        cap_original = cv2.VideoCapture(original_video_path)
        cap_stabilized = cv2.VideoCapture(stabilized_video_path)

        if not cap_original.isOpened() or not cap_stabilized.isOpened():
            raise ValueError("无法打开视频文件，请检查路径是否正确！")

        psnr_before_list = []
        psnr_after_list = []
        frame_numbers = []

        # 读取第一帧
        ret_orig, prev_frame_orig = cap_original.read()
        ret_stab, prev_frame_stab = cap_stabilized.read()

        if not ret_orig or not ret_stab:
            raise ValueError("无法读取视频第一帧！")

        frame_count = 1

        print("开始处理视频...")
        print(f"视频分辨率: {prev_frame_orig.shape[1]}x{prev_frame_orig.shape[0]}")

        while True:
            ret_orig, curr_frame_orig = cap_original.read()
            ret_stab, curr_frame_stab = cap_stabilized.read()

            if not ret_orig or not ret_stab:
                break

            # 计算稳像前相邻帧的PSNR
            psnr_before = self.calculate_psnr(prev_frame_orig, curr_frame_orig)
            psnr_before_list.append(psnr_before)

            # 计算稳像后相邻帧的PSNR
            psnr_after = self.calculate_psnr(prev_frame_stab, curr_frame_stab)
            psnr_after_list.append(psnr_after)

            frame_numbers.append(frame_count)
            frame_count += 1

            # 更新前一帧
            prev_frame_orig = curr_frame_orig
            prev_frame_stab = curr_frame_stab

            if frame_count % 30 == 0:
                print(f"已处理 {frame_count} 帧 | 稳像前PSNR: {psnr_before:.2f} dB | 稳像后PSNR: {psnr_after:.2f} dB")

        cap_original.release()
        cap_stabilized.release()

        print(f"\n处理完成！共处理 {frame_count} 帧")

        return frame_numbers, psnr_before_list, psnr_after_list

    def plot_psnr_comparison(self, frame_numbers, psnr_before, psnr_after,
                             title="视频稳像PSNR对比",
                             output_path="psnr_comparison.png",
                             figsize=(10, 6),
                             show_stats=True):
        """
        绘制PSNR对比曲线图

        参数:
            frame_numbers: 帧序号列表
            psnr_before: 稳像前PSNR列表
            psnr_after: 稳像后PSNR列表
            title: 图表标题
            output_path: 输出图片路径
            figsize: 图表尺寸
            show_stats: 是否显示统计信息
        """
        plt.figure(figsize=figsize)

        # 绘制稳像前后的PSNR曲线
        plt.plot(frame_numbers, psnr_before, 'b-', linewidth=1.5, label='稳像前', alpha=0.8)
        plt.plot(frame_numbers, psnr_after, 'r-', linewidth=1.5, label='稳像后', alpha=0.8)

        # 设置图表属性
        plt.xlabel('图像帧数', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3, linestyle='--')

        # 添加统计信息
        if show_stats:
            avg_before = np.mean(psnr_before)
            avg_after = np.mean(psnr_after)
            improvement = avg_after - avg_before

            textstr = f'平均PSNR:\n稳像前: {avg_before:.2f} dB\n稳像后: {avg_after:.2f} dB\n提升: {improvement:+.2f} dB'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存至: {output_path}")
        plt.show()

    def generate_report(self, frame_numbers, psnr_before, psnr_after,
                        output_path='psnr_report.txt'):
        """
        生成详细的统计报告
        """
        avg_before = np.mean(psnr_before)
        avg_after = np.mean(psnr_after)
        std_before = np.std(psnr_before)
        std_after = np.std(psnr_after)
        max_before = np.max(psnr_before)
        max_after = np.max(psnr_after)
        min_before = np.min(psnr_before)
        min_after = np.min(psnr_after)
        improvement = avg_after - avg_before

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("视频稳像PSNR分析报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"总帧数: {len(frame_numbers)}\n\n")

            f.write("【稳像前统计】\n")
            f.write(f"  平均PSNR: {avg_before:.2f} dB\n")
            f.write(f"  标准差:   {std_before:.2f} dB\n")
            f.write(f"  最大值:   {max_before:.2f} dB (第{np.argmax(psnr_before) + 1}帧)\n")
            f.write(f"  最小值:   {min_before:.2f} dB (第{np.argmin(psnr_before) + 1}帧)\n\n")

            f.write("【稳像后统计】\n")
            f.write(f"  平均PSNR: {avg_after:.2f} dB\n")
            f.write(f"  标准差:   {std_after:.2f} dB\n")
            f.write(f"  最大值:   {max_after:.2f} dB (第{np.argmax(psnr_after) + 1}帧)\n")
            f.write(f"  最小值:   {min_after:.2f} dB (第{np.argmin(psnr_after) + 1}帧)\n\n")

            f.write("【改善效果】\n")
            f.write(f"  PSNR提升:     {improvement:+.2f} dB\n")
            f.write(f"  稳定性变化:   {std_after - std_before:+.2f} dB\n")

            if improvement > 2:
                quality = "显著提升"
            elif improvement > 0:
                quality = "有所提升"
            elif improvement > -2:
                quality = "基本持平"
            else:
                quality = "有所下降"

            f.write(f"  质量评价:     {quality}\n\n")

            # PSNR质量等级
            f.write("【质量等级说明】\n")
            f.write("  > 40 dB: 优秀\n")
            f.write("  35-40 dB: 良好\n")
            f.write("  30-35 dB: 一般\n")
            f.write("  < 30 dB: 较差\n")

        print(f"统计报告已保存至: {output_path}")

        # 同时打印到控制台
        print("\n" + "=" * 60)
        print("统计摘要")
        print("=" * 60)
        print(f"稳像前平均PSNR: {avg_before:.2f} dB")
        print(f"稳像后平均PSNR: {avg_after:.2f} dB")
        print(f"PSNR提升: {improvement:+.2f} dB ({quality})")
        print("=" * 60)

    def analyze(self, original_video_path, stabilized_video_path,
                title="视频稳像PSNR对比",
                output_image="psnr_comparison.png",
                output_report="psnr_report.txt"):
        """
        一键完成所有分析（计算+绘图+报告）

        参数:
            original_video_path: 稳像前视频路径
            stabilized_video_path: 稳像后视频路径
            title: 图表标题
            output_image: 输出图片路径
            output_report: 输出报告路径
        """
        # 计算PSNR
        frame_numbers, psnr_before, psnr_after = self.calculate_video_psnr(
            original_video_path,
            stabilized_video_path
        )

        # 绘制对比图
        self.plot_psnr_comparison(
            frame_numbers,
            psnr_before,
            psnr_after,
            title=title,
            output_path=output_image
        )

        # 生成报告
        self.generate_report(
            frame_numbers,
            psnr_before,
            psnr_after,
            output_path=output_report
        )


# ==================== 使用示例 ====================

def main():
    """主函数"""

    # 创建分析器
    analyzer = VideoStabilizationPSNR()

    # 方式1: 一键分析（推荐）
    analyzer.analyze(
        original_video_path="original_output.mp4",
        stabilized_video_path="stabilized_output.mp4",
        title="稳像PSNR对比",
        output_image="psnr_result.png",
        output_report="psnr_report.txt"
    )

    # 方式2: 分步操作（更灵活）
    # frame_numbers, psnr_before, psnr_after = analyzer.calculate_video_psnr(
    #     "original_video.mp4",
    #     "stabilized_video.mp4"
    # )
    #
    # analyzer.plot_psnr_comparison(
    #     frame_numbers, psnr_before, psnr_after,
    #     output_path="my_result.png"
    # )
    #
    # analyzer.generate_report(
    #     frame_numbers, psnr_before, psnr_after,
    #     output_path="my_report.txt"
    # )


if __name__ == "__main__":
    main()
