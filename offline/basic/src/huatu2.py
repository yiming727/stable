"""
角点检测对比图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 📌 配置：修改此处为你的图片路径
# ============================================================
IMAGE_PATH  = "./img.png"   # ← 替换为实际图片路径
BUCKET_SIZE = 120                  # 每个桶的目标边长（像素）
SAVE_DIR    = "./output"           # 输出目录

# ============================================================
# 读取真实图片
# ============================================================
if not os.path.exists(IMAGE_PATH):
    print(f"❌ 图片文件不存在: {IMAGE_PATH}")
    print("   请将 IMAGE_PATH 修改为实际图片路径，例如：")
    print("   IMAGE_PATH = './test.jpg'")
    sys.exit(1)

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    print(f"❌ 无法读取图片（格式不支持或文件损坏）: {IMAGE_PATH}")
    sys.exit(1)

os.makedirs(SAVE_DIR, exist_ok=True)

H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

print(f"✅ 成功读取图片: {IMAGE_PATH}  ({W}x{H})")

# ============================================================
# 方法一：普通 Shi-Tomasi 角点检测
# ============================================================
shi_pts = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=500,
    qualityLevel=0.01,
    minDistance=10
)
shi_pts = shi_pts.reshape(-1, 2) if shi_pts is not None else np.empty((0, 2))

# ============================================================
# 方法二：本文分桶特征点提取
# ============================================================
def compute_bucket_grid(width, height, bucket_size=120):
    grid_cols = max(1, round(width  / bucket_size))
    grid_rows = max(1, round(height / bucket_size))
    return grid_rows, grid_cols

def detect_bucketing(img, grid_rows, grid_cols, quality=0.01, min_distance=10):
    h, w = img.shape
    bucket_h = h // grid_rows
    bucket_w = w // grid_cols
    selected = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            y1 = r * bucket_h
            y2 = (r + 1) * bucket_h if r < grid_rows - 1 else h
            x1 = c * bucket_w
            x2 = (c + 1) * bucket_w if c < grid_cols - 1 else w
            bucket_img = img[y1:y2, x1:x2]
            if bucket_img.size == 0:
                continue
            corners = cv2.goodFeaturesToTrack(
                bucket_img, maxCorners=0,
                qualityLevel=quality, minDistance=min_distance
            )
            if corners is not None:
                corners = corners.reshape(-1, 2)
                corners[:, 0] += x1
                corners[:, 1] += y1
                selected.extend(corners.tolist())
    if not selected:
        return np.empty((0, 2))
    return np.array(selected, dtype=np.float32)

grid_rows, grid_cols = compute_bucket_grid(W, H, bucket_size=BUCKET_SIZE)
bucket_pts = detect_bucketing(gray, grid_rows, grid_cols, quality=0.01, min_distance=5)

print(f"{'='*52}")
print(f"  普通 Shi-Tomasi  检测到: {len(shi_pts):>5} 个特征点")
print(f"  本文分桶提取     检测到: {len(bucket_pts):>5} 个特征点")
print(f"  分桶配置: {grid_rows} 行 × {grid_cols} 列（桶边长={BUCKET_SIZE}px）")
print(f"{'='*52}")

# ============================================================
# 通用绘图参数
# ============================================================
aspect  = W / H
fig_h   = 6
fig_w   = max(6, min(fig_h * aspect + 0.5, 14))
DPI     = 150
BG      = '#1e1e2e'

def base_fig():
    """创建统一背景的画布"""
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')
    return fig, ax

# ============================================================
# 图 1：原始图像
# ============================================================
fig, ax = base_fig()
ax.imshow(img_rgb)
ax.set_title(f"原始图像  ({W}×{H})", fontsize=14, color='white', pad=10)
save1 = os.path.join(SAVE_DIR, f"{img_name}_1_original.png")
plt.tight_layout()
plt.savefig(save1, dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print(f"✅ 已保存: {save1}")

# ============================================================
# 图 2：普通 Shi-Tomasi 角点检测
# ============================================================
fig, ax = base_fig()
ax.imshow(img_rgb)
if len(shi_pts) > 0:
    ax.scatter(shi_pts[:, 0], shi_pts[:, 1],
               s=18, c='#FF4444', edgecolors='white',
               linewidths=0.4, alpha=0.85, zorder=3)
save2 = os.path.join(SAVE_DIR, f"{img_name}_2_shi_tomasi.png")
plt.tight_layout()
plt.savefig(save2, dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print(f"✅ 已保存: {save2}")

# ============================================================
# 图 3：本文分桶特征点提取
# ============================================================
fig, ax = base_fig()
ax.imshow(img_rgb)
# 绘制桶格网
bucket_h_px = H // grid_rows
bucket_w_px = W // grid_cols
for r in range(grid_rows + 1):
    ax.axhline(r * bucket_h_px, color='#CCCCCC', linewidth=0.5, alpha=0.45)
for c in range(grid_cols + 1):
    ax.axvline(c * bucket_w_px, color='#CCCCCC', linewidth=0.5, alpha=0.45)
if len(bucket_pts) > 0:
    ax.scatter(bucket_pts[:, 0], bucket_pts[:, 1],
               s=18, c='#44AAFF', edgecolors='white',
               linewidths=0.4, alpha=0.85, zorder=3)
save3 = os.path.join(SAVE_DIR, f"{img_name}_3_bucketing.png")
plt.tight_layout()
plt.savefig(save3, dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.show()
print(f"✅ 已保存: {save3}")

print(f"\n🎉 全部完成！三张图已保存至目录: {SAVE_DIR}/")
print(f"   {os.path.basename(save1)}")
print(f"   {os.path.basename(save2)}")
print(f"   {os.path.basename(save3)}")