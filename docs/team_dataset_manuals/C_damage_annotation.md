# C 岗位手册：损坏标注、Mask 与合成退化

## 1. 你的任务

你有两类工作：

1. 对真实损坏照片标注损坏类型和严重程度，并为代表样本绘制局部 mask。
2. 对干净目标图生成可复现的轻、中、重合成退化训练对。

你的状态流转是：

```text
CLEANED -> ANNOTATED
```

只接收经过 B 清洗和 D 第一次抽查的图片。

## 2. 真实照片标注准备

生成当前首批任务表：

```powershell
cd C:\Users\24211\PycharmProjects\PhotoRevival
.\.venv\Scripts\python.exe -m scripts.data.team_dataset_workflow prepare-annotations `
  --batch-id 20260727_loc_pilot_001 `
  --annotator "你的真实姓名"
```

任务文件：

```text
data/work/batches/20260727_loc_pilot_001/03_annotation/damage_annotations.csv
```

对应图片：

```text
data/work/batches/20260727_loc_pilot_001/02_cleaned/accepted/
```

## 3. 损坏等级统一标准

所有字段只能填写 `0`、`1`、`2`、`3`：

| 等级 | 判断 |
|---:|---|
| 0 | 没有发现这种损坏 |
| 1 | 轻度，局部可见，不影响主体识别 |
| 2 | 中度，明显影响观感或局部内容 |
| 3 | 重度，大面积存在或主体信息缺失 |

不要用自己的审美判断“照片好不好看”，只判断对应损坏是否存在以及严重程度。

## 4. 每个字段怎么判断

| 字段 | 定义 | 容易误判的情况 |
|---|---|---|
| `scratch` | 细长表面划痕 | 头发、衣服纹理不是划痕 |
| `crack` | 相纸或乳剂裂纹，常有分叉 | 建筑线条不是裂纹 |
| `fold` | 折叠产生的直线或带状痕迹 | 相框边缘不是折痕 |
| `dust` | 分散的小亮点或黑点 | 胶片颗粒不是单独灰尘点 |
| `stain` | 液体、油污等不规则色斑 | 原始背景阴影不是污渍 |
| `mold` | 霉菌造成的斑点、丝状或团状区域 | 普通纸张纹理不是霉斑 |
| `tear` | 边缘或内部撕裂 | 正常相纸边框不是撕裂 |
| `missing_region` | 图像内容或乳剂完全缺失 | 纯背景区域不算缺失 |
| `fading` | 对比度下降、整体变淡 | 高调摄影风格不一定是褪色 |
| `color_cast` | 非原始意图的泛黄、偏红、偏绿 | 褐色调老照片要结合整体判断 |
| `blur` | 失焦或运动模糊 | 低对比不等于模糊 |
| `noise` | 胶片颗粒、扫描噪声 | 灰尘和划痕单独标注 |
| `jpeg_artifact` | 块状压缩、振铃 | 自然颗粒不是 JPEG 块 |
| `uneven_exposure` | 局部过亮、过暗或扫描光照不均 | 原始布光阴影不一定是损坏 |

## 5. 逐张填写流程

1. 根据 `source_id` 打开图片。
2. 先看全图，判断全局褪色、偏色、模糊和曝光。
3. 放大查看划痕、裂纹、灰尘、霉斑和缺失。
4. 按 0-3 填完全部 14 个损坏字段，不能留空。
5. 看不清时填 `uncertain=1`，在 `notes` 写明争议点。
6. 填写自己的真实姓名到 `annotator`。
7. 将该行状态改为 `ANNOTATED`。
8. 每完成 10 张保存一次，并备份 CSV。

禁止直接复制上一行标签。相似照片也必须独立查看。

## 6. 哪些照片需要 Mask

优先给 200-300 张代表性照片画 mask，不要求所有真实照片都有 mask。

需要局部 mask：

- 划痕。
- 裂纹。
- 折痕造成的内容破坏。
- 撕裂。
- 乳剂脱落和内容缺失。
- 边界清楚的污渍或霉斑。

不画局部 mask：

- 全局褪色。
- 全图偏色。
- 整体模糊。
- 全局噪声。
- 普遍 JPEG 压缩。
- 全局曝光不均。

## 7. Mask 制作规范

1. 使用原图同尺寸画布。
2. 保存为单通道 PNG。
3. 黑色 `0` 表示无需局部修复。
4. 白色 `255` 表示损坏区域。
5. 不使用灰色软边作为最终真值，除非项目另外定义权重 mask。
6. 不要覆盖人物轮廓、发丝、衣服花纹等正常内容。
7. 文件名与 `source_id` 对应。
8. 把相对路径填入 `mask_path`。

建议目录：

```text
data/work/batches/<batch_id>/03_annotation/masks/<source_id>.png
```

完成后检查：原图和 mask 宽高完全一致，mask 不是全黑或全白，覆盖区域确实是局部损坏。

## 8. 双人复核与争议

D 复核时：

1. D 不先看你的结论，独立判断损坏等级。
2. 两人比较差异达到 2 级以上的字段。
3. 共同查看原图并记录最终结论。
4. 无法达成一致时，由 A 或 B 第三人裁决。
5. 复核人填写 `reviewer`，状态改为 `REVIEWED`。

不要默默覆盖原值。争议原因和最终决定写入 `notes`。

## 9. 合成退化任务

只有 `quality_label=clean_target_ok` 的图片可作为目标图。每张生成轻、中、重三个版本：

```text
target/source_000001.png
degraded/source_000001_light.png
degraded/source_000001_medium.png
degraded/source_000001_heavy.png
masks/source_000001_light.png
masks/source_000001_medium.png
masks/source_000001_heavy.png
metadata/source_000001.json
```

退化应覆盖：

- 全局：褪色、偏色、对比度、gamma、暗角、曝光不均。
- 局部：划痕、裂纹、折痕、污渍、霉斑、灰尘、撕裂、缺失。
- 成像：模糊、胶片颗粒、Gaussian/Poisson 噪声、JPEG、下采样、扫描噪声。

## 9.1 运行合成退化生成脚本

从 B 的 `02_cleaned` 批次生成轻/中/重退化对和 mask：

```powershell
cd C:\Users\24211\PycharmProjects\PhotoRevival
.\.venv\Scripts\python.exe -m scripts.synthetic.generate `
  --batch-id 20260727_loc_pilot_001 `
  --sample 20
```

脚本只取 `manifest.csv` 中 `quality_label=clean_target_ok` 的图片作为目标图，输出到：

```text
data/work/batches/<batch_id>/05_synthetic_pairs/
├── target/<source_id>.png
├── degraded/<source_id>_<severity>.png
├── masks/<source_id>_<severity>.png        # 0=干净，255=损坏
├── metadata/<source_id>.json               # 单文件，内含 light/medium/heavy 三档各自的种子 + 每步退化参数
├── collages/<source_id>_comparison.png
├── manifest.csv
├── summary.json
└── contact_sheet.jpg
```

- `--sample 20`：处理 20 张目标图（60 组样本），并额外生成 `sample_manifest.csv` 和 `contact_sheet_sample.jpg`，作为提交 D 审核的首批试样。
- `--verify-only`：对已生成目录做自动验收（尺寸一致、mask 二值且非空、覆盖率在界内、种子复现），失败退出码为 2。
- **一条命令跑全部可用性验证**（fixture 生成、端到端生成、自检、确定性复现、全局/样例/幂等模式、参数校验，无需真实数据）：

  ```powershell
  python -m scripts.synthetic.verify
  ```

  全部通过退出码为 0，任一失败为 1，并逐项打印 PASS/FAIL 清单。
- 局部 mask 来自损坏图层 alpha，不是检测器预测；全局褪色/噪声/模糊不产生局部 mask（用 `--global-only` 生成 `global_restoration` 样本）。
- 每个变体独立 seed，记录在单文件 `metadata/<source_id>.json` 的对应档位里，同参数重跑结果字节级一致。
- 局部损坏来自 `data/textures/Resource-Boy-Grunge-Textures/Resource Boy - Grunge Textures/` 的真实 grunge 素材；程序只画折痕/撕裂/缺失等结构性损坏，不自行生成纹理。
- 本地无数据时可先 `--make-fixture` 生成确定性小样本演练。

## 10. 合成数据硬性要求

1. 每个版本保存随机种子。
2. 保存每个退化步骤和参数。
3. mask 直接来自局部损坏图层的 alpha，不用检测模型预测结果冒充真值。
4. 输入、目标和 mask 尺寸完全一致。
5. 空 mask 不进入局部修复训练集。
6. 局部 mask 覆盖率主要在 1%-35%。
7. 只有全局损坏时标记 `global_restoration`，不伪造小 mask。
8. 轻、中、重必须肉眼可区分，但重度不能破坏到毫无真实感。

开始批量生成前，先交付 20 张目标图的 60 组试样和对比拼图，由 D 审核通过后再扩大。

## 11. 交付前检查

- [ ] 真实照片所有损坏字段均为 0-3，没有空白。
- [ ] `annotator` 和状态已填写。
- [ ] 不确定样本有 notes。
- [ ] mask 与原图尺寸一致。
- [ ] mask 不误伤正常纹理。
- [ ] 合成样本保存随机种子和参数。
- [ ] 输入、目标、mask 一一对应。
- [ ] 空 mask 为 0。
- [ ] 已生成抽样对比拼图。

## 12. 群内交接模板

```text
姓名/角色：王五 / C
批次：20260727_loc_pilot_001
真实照片输入：57
已标注：57
已画 mask：20
不确定样本：4
主要争议：褐色调是否属于 color_cast
标注表：data/work/batches/20260727_loc_pilot_001/03_annotation/damage_annotations.csv
需要 D：逐图复核并运行验收
```
