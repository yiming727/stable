# 红外 / 可见光视频稳像 Monorepo

原本是 5 个各自独立的 git 仓库，内容互相关联却彼此割裂：同一份代码散落在多个仓库里，
依赖没有任何一处声明，找东西不知道该去哪个仓库。此仓库把它们合并到一起，保留全部历史，
并抽出共享模块 `common/` 消除重复。

---

## 1. 目录结构

```
stable/
├── offline/
│   ├── basic/          离线稳像（传统 CV 路线）
│   │   ├── src/            成熟稳定器：网格 Harris + Shi-Tomasi + LK + FB-check + RANSAC + ECC
│   │   ├── u16/            16 位 KEII-IR 红外 (.IRV) 专用预处理与稳像
│   │   ├── 视频防抖.py      最早的单文件版本
│   │   └── README.md       原始设计说明（预处理/判抖/运动提取/滤波/合成）
│   └── enhanced/       离线稳像（SuperGlue/SuperPoint 特征匹配路线）
│       └── models/         SuperPoint + SuperGlue 实现与权重
├── online/
│   ├── basic/          流式稳像：5 个脚本 = 5 种平滑滤波的平行对照
│   └── enhanced/       流式稳像：训练好的 LSTM+Transformer+CNN 网络
│       ├── method/         旧架构（已被 method1 取代）
│       ├── method1/        现行架构（canonical）
│       └── trajectories/   150 个训练轨迹对 (noisy / stable)
├── tools/              12 个通用小工具（视频↔图片、缩放、拼接、指标、绘图……）
├── common/             ★ 共享模块（本仓库新增）
├── requirements.txt
└── pyproject.toml
```

`offline/basic/` 下 `src/` 与 `u16/` **刻意不展平**：两者各有一个 `stable.py`，
且都靠"同级目录直接 import"工作（`u16/stable.py` → `keii_data_load` → `ir_color`），
展平会同时造成重名冲突和 import 断裂。

---

## 2. 环境

```bash
conda activate py39            # Python 3.9.23，见下
pip install -r requirements.txt
pip install -e . --no-deps     # 让各项目能 import common
```

⚠️ **`pip install -e .` 必须带 `--no-deps`** —— `pyproject.toml` 里故意没有写
`[project.dependencies]`，依赖统一由 `requirements.txt` 管理。原因是本机 py39 环境里
cv2 4.10 / torch 2.8+cu129 / numpy 1.26.4 的 ABI 是手工对齐好的，一旦让 pip 去索引
解析依赖，numpy 很容易被升到 2.x，连带弄坏 cv2 与 torch。

⚠️ **默认 `python` 不能用来跑这些脚本**（它是 Anaconda base 3.13.5，没有 cv2 也没有
torch）。请显式使用 `D:\program\Anaconda\envs\py39\python.exe`。

⚠️ `opencv-contrib-python` 是硬性要求：`offline/basic/u16/img_trans.py` 用到
`cv2.ximgproc.guidedFilter`，普通 `opencv-python` 没有 `ximgproc` 子模块。

---

## 3. 大文件（**有意**随仓库版本管理，不要 gitignore）

| 文件 | 体积 | 说明 |
|---|---|---|
| `offline/basic/u16/20230831171237_00.IRV` | 88 MiB | 唯一的红外原始样本，KEII-IR 格式 |
| `offline/enhanced/models/weights/*.pth` | 97 MiB | SuperGlue indoor/outdoor + SuperPoint 权重 |
| `online/enhanced/trajectories/*.npy` | 150 个 | 训练用的 (N,3) 轨迹对 |
| `online/basic/*.mp4` | 5 MiB | 更早一次运行的演示输出 |

`.gitignore` 里**故意没有**写 `*.pth` / `*.npy` / `*.IRV` / `*.mp4` / `models/`。
这几个是最常见的模板写法，但加进来会让上面这些文件静默消失。改完 `.gitignore` 请自检：

```bash
git check-ignore -v offline/basic/u16/20230831171237_00.IRV \
    offline/enhanced/models/weights/superglue_indoor.pth \
    online/enhanced/trajectories/video_00_noisy.npy \
    online/basic/original_output.mp4      # 应当没有任何输出
```

---

## 4. 怎么运行

**规则：先 `cd` 到脚本自己所在的目录再运行。** 这些脚本里的数据路径是相对
**当前工作目录**解析的，不是相对 `__file__`：

```bash
cd offline/basic/src       && python stable.py
cd offline/enhanced        && python main.py
cd online/basic            && python stable.py     # 需要摄像头
cd online/enhanced/method1 && python train.py      # 读 ../trajectories

# offline/basic/u16 —— 仓库里唯一自带数据、能真正端到端跑的一段（已实测 596 帧通过）
cd offline/basic/u16 && python stable.py --input 20230831171237_00.IRV --output out.mp4
```

⚠️ `u16/stable.py` 的默认 `--input` 是 `50号阀定位销漏气.IRV`，而仓库里实际带的文件叫
`20230831171237_00.IRV` —— 所以**必须显式传 `--input`**，直接 `python stable.py` 会因找不到文件而失败。
（这是合并前就存在的问题，见"已知问题"第 8 条。）

合并后有两处相对路径的**深度变了**（都已失效，仅作记录）：

| 位置 | 原写法 | 原本解析到 | 现在解析到 |
|---|---|---|---|
| `offline/enhanced/superglue.py` | `../Data/11.mp4` | `stable/Data/` | `stable/offline/Data/` |
| `online/enhanced/method/data.py` | `../../data/Zooming/` | `stable/data/` | `stable/online/data/` |

---

## 5. 共享模块 `common/`

| 模块 | 内容 | 来源（原重复位置） |
|---|---|---|
| `trajectory.py` | `TrajectorySmoother` / `moving_average` | `offline/basic/src/stable.py`、`src/stable1.py`、`u16/stable.py`（2 遍）+ `offline/enhanced/smooth.py`（3 遍） |
| `borders.py` | `build_transformation_matrix` / `extreme_corners` / `min_auto_border_size` / `fix_border` | `src/stable.py`、`u16/stable.py`、`offline/enhanced/fixborder.py` |
| `features.py` | `GridCalculator` / `detect_features` / `fb_check` / `estimate_transform_ecc` / `detect_bucketing` | `src/stable.py`、`src/stable1.py`、`src/huatu2.py`、`u16/stable.py` |
| `shake.py` | 光流 + 峰值检测判抖 | `offline/basic/src/detection.py` ≡ `offline/enhanced/detection.py`（**字节完全相同**） |
| `stability_metrics.py` | CR / DV / SS + 帧间 PSNR / SSIM | `src/evaluate.py` ≡ `offline/enhanced/evaluate.py`（字节相同）+ 3 份 `ssimAndPsnr.py` |
| `image_io.py` | `natural_sort_key` / `extract_frames` / `frame2vid` / `video_to_images` | `tools/图片合成视频.py`、`tools/图片排序.py`、`offline/enhanced/superglue.py`、`tools/视频转图片.py` |
| `plotting.py` | 中文字体配置 | 约 15 处（`SimHei` 与 `msyh.ttc` 两种写法） |

### 重构时最容易踩的坑（已参数化，别"统一"掉）

| 差异点 | 各处取值 | 后果 |
|---|---|---|
| `TrajectorySmoother.passes` | 2（basic 三处） / 3（enhanced） | 平滑强度不同，**数值会变** |
| `fix_border(border_mode=)` | `BORDER_REPLICATE`（basic） / `BORDER_CONSTANT`（enhanced 的 fixborder.py） | 画面出界时补边效果不同 |
| `extract_grid_points(shape=)` | `'3d'`→(N,1,2)+None（src/stable.py） / `'2d'`→(N,2)（u16/stable.py） | 形状不同，cv2 光流只吃 `'3d'` |
| `detect_bucketing(min_distance=)` | 5（stable1.py） / 10（huatu2.py） | 角点密度不同 |
| `natural_sort_key(basename=, lower=)` | 三种变体 | 排序结果不同 |

### 哪些地方**没有**改，以及为什么

去重只做了"确定等价"的部分。以下一律保持原样：

- **三份 `ssimAndPsnr.py` 没有合并成薄壳** —— 它们唯一的差异就是各自的 CLI 默认路径
  （`./9.mp4` / `./18.mp4` / `./11.mp4`），薄壳化买不到任何去重收益，反而给一个
  本来能跑的 CLI 引入新失败点。它们的核心函数已收录进 `common/stability_metrics.py`。
- **`tools/metrics.py`（490 行 `VideoStabilityEvaluator`）没有并入** —— 它算的是同一套
  CR/DV/SS，但接口是"视频路径进、指标出"，与 `evaluate.py` 的"目录进、目录出"不同，
  且实现更健壮。两个入口都保留。
- **实验脚本一律没动**：`src/huatu1.py`、`huatu2.py`、`huatu3.py`、`hautu.py`、
  `src/amplitude_limit.py`、`src/area_limit.py`、`src/chafen.py`。
  它们是平行对比实验（例如 `online/basic/` 的 5 个文件是 5 种平滑滤波的对照，
  不是版本迭代），重构会破坏其"独立可跑"的价值。
- **`offline/enhanced/models/`（SuperGlue）没有上提到 `common/`** —— 它目前只被
  `offline/enhanced` 用，不属于重复代码；而且 97 MiB 权重靠相对路径
  `weights/*.pth` 加载，搬动要改路径、风险大于收益。

---

## 6. 已知问题（本次整合**未**修改，仅记录）

1. **大量脚本指向不存在的 `../Data/`**：`offline/basic/src/detection.py`、
   `evaluate.py`、`offline/enhanced/superglue.py`、`online/enhanced/method/data.py`
   （还指向 `../../data/Zooming/28stb.avi`）。数据集不在仓库里，这些脚本目前跑不通。
2. **两个失效 import**（按决定**只标注、不改代码**）：
   - `offline/basic/u16/img_trans.py:3` — `from keii_sdk.ir_color import ppbyIron`，
     `keii_sdk` 包不在仓库中。同级的 `u16/ir_color.py` 里有同名调色板表，
     且 `keii_data_load.py` 引的就是它。
   - `online/enhanced/method/train.py:9` — `from Graph.enhancedEdition.model import ...`，
     `Graph` 包不在仓库中。同目录 `method/model.py` 定义了同名类，
     `method/test.py` 引的就是它。
3. **`online/enhanced/method/train.py` 权重保存路径错位**：写 `../multi_model_ensemble.pth`，
   实际解析到仓库根而非 `method/`；而权重其实存放在 `method/` 下。
   `method1/train.py` 是修正后的版本。
4. **潜在 bug**：`offline/basic/u16/img_trans.py` 的 `keii_read_jpg` 调用
   `self.Hist16to8(fuse_u16, Lowratio=8)`，但 `Hist16to8(self, img)` 不接受该参数。
5. **`online/basic/stable*.py` 都不写输出**：`stable.py` 的 `VideoWriter` 被注释，
   `stable2/3/4.py` 的 `out.write()` 被注释，只做 `imshow` 实时显示。
   仓库里的 `original_output.mp4` / `stabilized_output.mp4` 是更早一次运行的遗留产物。
6. **"在线"名不副实的两处**：`online/basic/stable1.py` 的滑动平均在 250 帧缓冲内
   **非因果**（用到了窗口内的"未来"帧）；`stable4.py` 每帧对**全部历史**重解优化，
   代价随帧数无限增长。两者与"在线/低延迟"的定位有出入。
7. **`common/stability_metrics.py` 继承的两个隐患**（未改）：
   `compute_distortion_value` 在序列为空时抛 `ValueError`；
   `compute_stability_score` 在 FFT 全零时除以零。
8. **`offline/basic/u16/stable.py` 的默认输入文件名对不上**：默认 `--input` 是
   `50号阀定位销漏气.IRV`，但仓库里实际带的样本叫 `20230831171237_00.IRV`，
   所以直接 `python stable.py` 会失败，必须显式传 `--input`（见第 4 节的命令）。
   这是合并前就有的问题。

---

## 7. 来源与历史

由 5 个仓库用 `git subtree` 合并而来，**全部原始提交都是本仓库的祖先**，
可用 `git show <sha>:<path>` 直接查看合并前的任意版本。

| 原仓库 | 目录 | 提交数 | 原 master |
|---|---|---|---|
| `yiming727/offlineBasic` | `offline/basic/` | 16 | `2538067` |
| `yiming727/offlineEnhanced` | `offline/enhanced/` | 3 | `492fa2c` |
| `yiming727/offlineTool` | `tools/` | 2 | `ca7ffdc` |
| `yiming727/onlineBasic` | `online/basic/` | 2 | `9f58eaa` |
| `yiming727/onlineEnhanced` | `online/enhanced/` | 13 | `af60a27` |

合计 36 个原始提交。例：

```bash
git show 2538067:src/evaluate.py      # 看合并前 offlineBasic 里的原始文件
```

---

## 8. 许可提示

`offline/enhanced/models/` 下的 `matching.py`、`superglue.py`、`superpoint.py`、`utils.py`
来自 SuperGlue 参考实现（作者 Paul-Edouard Sarlin、Daniel DeTone、Tomasz Malisiewicz 等），
文件头带有 Magic Leap 的版权声明。沿用/分发前请遵守其原始许可。
