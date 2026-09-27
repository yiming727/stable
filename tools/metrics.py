"""
视频稳定性评估工具
基于 SIFT 特征匹配和单应性变换分析
评估指标：裁剪比例(CR)、失真值(DV)、稳定性分数(SS)
"""

import os
import sys
import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class StabilityMetrics:
    """存储评估指标的数据类"""
    cr_avg: float  # 平均裁剪比例
    cr_min: float  # 最小裁剪比例
    dv_min: float  # 最小失真值
    ss_avg: float  # 平均稳定性分数
    ss_trans: float  # 平移稳定性
    ss_rot: float  # 旋转稳定性


class VideoStabilityEvaluator:
    """视频稳定性评估器"""

    def __init__(self,
                 min_match_count: int = 10,
                 ratio_threshold: float = 0.7,
                 ransac_threshold: float = 5.0,
                 frame_mismatch_strategy: str = 'use_shorter'):
        """
        初始化评估器

        Args:
            min_match_count: 最小匹配点数量
            ratio_threshold: Lowe's ratio test 阈值
            ransac_threshold: RANSAC 重投影误差阈值
            frame_mismatch_strategy: 帧数不一致时的处理策略
                - 'use_shorter': 使用较短视频的帧数（默认）
                - 'use_longer': 使用较长视频的帧数（重复最后一帧）
                - 'align_start': 从开头对齐
                - 'align_end': 从结尾对齐
                - 'error': 抛出错误
        """
        self.min_match_count = min_match_count
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.frame_mismatch_strategy = frame_mismatch_strategy

        # 初始化特征检测器和匹配器
        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher()

    def read_video_frames(self, video_path: str) -> Tuple[List[np.ndarray], dict]:
        """
        读取视频的所有帧

        Args:
            video_path: 视频文件路径

        Returns:
            frames: 帧列表
            info: 视频信息字典
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

        frames = []
        print(f"正在读取视频: {os.path.basename(video_path)}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

            # 显示进度
            sys.stdout.write(f'\r读取帧: {len(frames)}/{info["frame_count"]}')
            sys.stdout.flush()

        cap.release()
        print(f"\n成功读取 {len(frames)} 帧")

        return frames, info

    def align_frame_sequences(self,
                              original_frames: List[np.ndarray],
                              stabilized_frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        对齐两个帧序列（处理帧数不一致的情况）

        Args:
            original_frames: 原始视频帧列表
            stabilized_frames: 稳定后视频帧列表

        Returns:
            aligned_original: 对齐后的原始帧列表
            aligned_stabilized: 对齐后的稳定帧列表
        """
        len_orig = len(original_frames)
        len_stab = len(stabilized_frames)

        if len_orig == len_stab:
            return original_frames, stabilized_frames

        print(f"\n⚠️  检测到帧数不一致: 原始视频 {len_orig} 帧, 稳定后视频 {len_stab} 帧")
        print(f"采用策略: {self.frame_mismatch_strategy}")

        if self.frame_mismatch_strategy == 'error':
            raise ValueError(
                f"视频帧数不一致: 原始视频 {len_orig} 帧, "
                f"稳定后视频 {len_stab} 帧"
            )

        elif self.frame_mismatch_strategy == 'use_shorter':
            # 策略1: 使用较短视频的帧数
            min_frames = min(len_orig, len_stab)
            print(f"✓ 使用前 {min_frames} 帧进行评估")
            return original_frames[:min_frames], stabilized_frames[:min_frames]

        elif self.frame_mismatch_strategy == 'use_longer':
            # 策略2: 使用较长视频的帧数（重复最后一帧）
            max_frames = max(len_orig, len_stab)

            aligned_orig = original_frames.copy()
            aligned_stab = stabilized_frames.copy()

            # 填充较短的序列
            if len_orig < max_frames:
                print(f"✓ 原始视频填充 {max_frames - len_orig} 帧（重复最后一帧）")
                last_frame = original_frames[-1]
                aligned_orig.extend([last_frame] * (max_frames - len_orig))

            if len_stab < max_frames:
                print(f"✓ 稳定视频填充 {max_frames - len_stab} 帧（重复最后一帧）")
                last_frame = stabilized_frames[-1]
                aligned_stab.extend([last_frame] * (max_frames - len_stab))

            return aligned_orig, aligned_stab

        elif self.frame_mismatch_strategy == 'align_start':
            # 策略3: 从开头对齐（截断较长的视频）
            min_frames = min(len_orig, len_stab)
            print(f"✓ 从开头对齐，使用前 {min_frames} 帧")
            return original_frames[:min_frames], stabilized_frames[:min_frames]

        elif self.frame_mismatch_strategy == 'align_end':
            # 策略4: 从结尾对齐（截断较长的视频）
            min_frames = min(len_orig, len_stab)
            print(f"✓ 从结尾对齐，使用后 {min_frames} 帧")
            return original_frames[-min_frames:], stabilized_frames[-min_frames:]

        else:
            raise ValueError(f"未知的对齐策略: {self.frame_mismatch_strategy}")

    def extract_and_match_features(self,
                                   img1: np.ndarray,
                                   img2: np.ndarray) -> Tuple[List[cv2.DMatch], List, List]:
        """
        提取特征并进行匹配

        Args:
            img1: 源图像
            img2: 目标图像

        Returns:
            good_matches: 优质匹配点列表
            kp1: 图像1的关键点
            kp2: 图像2的关键点
        """
        # 检测特征点和描述符
        kp1, desc1 = self.sift.detectAndCompute(img1, None)
        kp2, desc2 = self.sift.detectAndCompute(img2, None)

        if desc1 is None or desc2 is None:
            return [], kp1, kp2

        # KNN 匹配
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test 筛选优质匹配
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches, kp1, kp2

    def compute_homography(self,
                           img1: np.ndarray,
                           img2: np.ndarray) -> Optional[np.ndarray]:
        """
        计算两幅图像之间的单应性矩阵

        Args:
            img1: 源图像
            img2: 目标图像

        Returns:
            homography: 3x3 单应性矩阵，如果匹配失败则返回 None
        """
        good_matches, kp1, kp2 = self.extract_and_match_features(img1, img2)

        if len(good_matches) < self.min_match_count:
            print(f"\n警告: 匹配点数量不足 ({len(good_matches)} < {self.min_match_count})")
            return None

        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 使用 RANSAC 估计单应性矩阵
        homography, mask = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold
        )

        return homography

    def compute_cropping_ratio(self, homography: np.ndarray) -> float:
        """
        从单应性矩阵计算裁剪比例

        Args:
            homography: 单应性矩阵

        Returns:
            cropping_ratio: 裁剪比例
        """
        # 基于矩阵分解提取缩放因子
        scale = np.sqrt(homography[0, 0] ** 2 + homography[0, 1] ** 2)
        return 1.0 / scale

    def compute_distortion_value(self, homography: np.ndarray) -> float:
        """
        计算失真值（特征值比）

        Args:
            homography: 单应性矩阵

        Returns:
            distortion: 失真值
        """
        # 提取仿射部分的特征值
        affine_part = homography[0:2, 0:2]
        eigenvalues = np.linalg.eigvals(affine_part)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

        # 避免除零
        if eigenvalues[0] < 1e-6:
            return 0.0

        return eigenvalues[1] / eigenvalues[0]

    def extract_motion_parameters(self, homography: np.ndarray) -> Tuple[float, float]:
        """
        从单应性矩阵提取运动参数

        Args:
            homography: 单应性矩阵

        Returns:
            translation: 平移量
            rotation: 旋转角度（度）
        """
        # 平移量
        translation = np.sqrt(homography[0, 2] ** 2 + homography[1, 2] ** 2)

        # 旋转角度
        rotation = np.arctan2(homography[1, 0], homography[0, 0]) * 180.0 / np.pi

        return translation, rotation

    def compute_stability_score(self, motion_sequence: List[float]) -> float:
        """
        通过 FFT 分析计算稳定性分数

        Args:
            motion_sequence: 运动参数时间序列

        Returns:
            stability_score: 稳定性分数（低频能量占比）
        """
        if len(motion_sequence) < 2:
            return 1.0

        # FFT 变换
        fft_result = np.fft.fft(motion_sequence)

        # 计算功率谱
        power_spectrum = np.abs(fft_result) ** 2

        # 去除直流分量，保留正频率部分
        power_spectrum = power_spectrum[1:len(power_spectrum) // 2]

        if len(power_spectrum) == 0:
            return 1.0

        # 计算低频能量占比（前5个频率分量）
        num_low_freq = min(5, len(power_spectrum))
        low_freq_energy = np.sum(power_spectrum[:num_low_freq])
        total_energy = np.sum(power_spectrum)

        if total_energy < 1e-6:
            return 1.0

        return low_freq_energy / total_energy

    def evaluate(self,
                 original_video: str,
                 stabilized_video: str) -> StabilityMetrics:
        """
        评估视频稳定性

        Args:
            original_video: 原始视频路径
            stabilized_video: 稳定后视频路径

        Returns:
            metrics: 评估指标对象
        """
        print("=" * 60)
        print("视频稳定性评估")
        print("=" * 60)

        # 读取视频帧
        original_frames, orig_info = self.read_video_frames(original_video)
        stabilized_frames, stab_info = self.read_video_frames(stabilized_video)

        # 对齐帧序列（处理帧数不一致）
        original_frames, stabilized_frames = self.align_frame_sequences(
            original_frames, stabilized_frames
        )

        num_frames = len(original_frames)
        print(f"\n开始分析 {num_frames} 帧...")

        # 初始化存储
        cr_sequence = []  # 裁剪比例序列
        dv_sequence = []  # 失真值序列
        trans_sequence = []  # 平移序列
        rot_sequence = []  # 旋转序列

        cumulative_homography = np.eye(3)  # 累积单应性矩阵

        # 逐帧分析
        for i in range(num_frames):
            # 显示进度
            progress = (i + 1) / num_frames * 100
            sys.stdout.write(f'\r分析进度: {i + 1}/{num_frames} ({progress:.1f}%)')
            sys.stdout.flush()

            # 计算原始帧与稳定帧之间的单应性矩阵
            H = self.compute_homography(original_frames[i], stabilized_frames[i])

            if H is not None:
                # 计算裁剪比例和失真值
                cr = self.compute_cropping_ratio(H)
                dv = self.compute_distortion_value(H)

                cr_sequence.append(min(cr, 1.0))  # 限制最大值为1
                dv_sequence.append(dv)

            # 计算相邻稳定帧之间的单应性矩阵（用于稳定性分数）
            if i + 1 < num_frames:
                H_inter = self.compute_homography(
                    stabilized_frames[i],
                    stabilized_frames[i + 1]
                )

                if H_inter is not None:
                    # 累积变换
                    cumulative_homography = np.matmul(cumulative_homography, H_inter)

                    # 提取运动参数
                    trans, rot = self.extract_motion_parameters(cumulative_homography)
                    trans_sequence.append(trans)
                    rot_sequence.append(rot)

        print("\n\n计算稳定性分数...")

        # 计算稳定性分数
        ss_trans = self.compute_stability_score(trans_sequence)
        ss_rot = self.compute_stability_score(rot_sequence)
        ss_avg = (ss_trans + ss_rot) / 2.0

        # 汇总结果
        metrics = StabilityMetrics(
            cr_avg=np.mean(cr_sequence) if cr_sequence else 1.0,
            cr_min=np.min(cr_sequence) if cr_sequence else 1.0,
            dv_min=np.min(np.abs(dv_sequence)) if dv_sequence else 1.0,
            ss_avg=ss_avg,
            ss_trans=ss_trans,
            ss_rot=ss_rot
        )

        return metrics

    def print_results(self, metrics: StabilityMetrics):
        """
        打印评估结果

        Args:
            metrics: 评估指标对象
        """
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)

        print("\n 裁剪比例 (Cropping Ratio)")
        print(f"   平均值: {metrics.cr_avg:.4f}")
        print(f"   最小值: {metrics.cr_min:.4f}")
        print(f"   说明: 越接近 1.0 表示裁剪越少")

        print("\n 失真值 (Distortion Value)")
        print(f"   最小值: {metrics.dv_min:.4f}")
        print(f"   说明: 越接近 1.0 表示几何失真越小")

        print("\n 稳定性分数 (Stability Score)")
        print(f"   综合得分: {metrics.ss_avg:.4f}")
        print(f"   平移稳定性: {metrics.ss_trans:.4f}")
        print(f"   旋转稳定性: {metrics.ss_rot:.4f}")
        print(f"   说明: 越接近 1.0 表示运动越平滑")

        print("\n" + "=" * 60)


def main():
    """主函数"""
    # 配置参数
    original_video_path = "./07.mp4"
    stabilized_video_path = "./DIFRINT07.mp4"

    # 创建评估器
    evaluator = VideoStabilityEvaluator(
        min_match_count=10,
        ratio_threshold=0.7,
        ransac_threshold=5.0,
        frame_mismatch_strategy='use_shorter'  # 使用较短视频的帧数
    )

    try:
        # 执行评估
        metrics = evaluator.evaluate(original_video_path, stabilized_video_path)

        # 打印结果
        evaluator.print_results(metrics)

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 支持命令行参数
    if len(sys.argv) >= 3:
        original_video = sys.argv[1]
        stabilized_video = sys.argv[2]

        # 可选：从命令行指定策略
        strategy = sys.argv[3] if len(sys.argv) > 3 else 'use_shorter'

        evaluator = VideoStabilityEvaluator(frame_mismatch_strategy=strategy)
        metrics = evaluator.evaluate(original_video, stabilized_video)
        evaluator.print_results(metrics)
    else:
        main()
