# 合成退化流水线（数据集 C 岗）

本目录是数据集建设中 **C 岗：合成退化 + mask 生成** 的完整实现，从干净目标图生成可复现的轻/中/重退化训练对。真实照片的人工 mask 标注不在此目录。

## 文件拆分

| 文件 | 职责 |
|---|---|
| `analyze_textures.py` | 对 250 张无标签纹理使用人工视觉标签，计算素材属性并生成分类目录和总览 |
| `texture_assets.py` | 本目录内的素材路径和素材文件枚举辅助逻辑 |
| `texture_catalog.csv` | 250 张素材的主标签、子标签、辅助标签、混合模式和启用状态 |
| `engine.py` | **严格全素材退化引擎**（numpy + Pillow）。按类别选材、组合、生成 mask、控制覆盖率并记录完整素材参数 |
| `generate.py` | **生成入口**（CLI）。从 `02_cleaned` 取 `clean_target_ok` 图片，输出到 `<batch>/05_synthetic_pairs` |
| `verify.py` | **自动验证**（16 项检查，含素材目录、评分、确定性复现和遮罩外零改动） |
| `REQUIREMENTS_AUDIT.md` | 对照 C 岗需求的逐项实现审查与剩余工作 |
| `README.md` | 本说明 |

Python 依赖：numpy、Pillow。纹理分析脚本另需 OpenCV。生成器会只读复用现有 `scripts/data/team_dataset_workflow.py` 和 `src/photo_revival/paths.py`，不修改这些目录。

**严格素材要求**：所有可见做旧都来自 `data/textures/Resource-Boy-Grunge-Textures/Resource Boy - Grunge Textures/` 的已分类素材。程序只允许裁剪、缩放、90 度旋转、翻转、阈值选择、透明度调整和 `screen`/`multiply` 合成；不生成偏色、褪色、暗角、噪声、模糊、压缩伪影、折痕形状或缺失区域。

## 素材分类与组合

| 组别 | 类别 |
|---|---|
| `detail`（细节类） | `scratch`、`crack`、`dust` |
| `area`（面积类） | `stain`、`mold`、`surface_wear` |
| `structure`（结构类） | `edge_wear`、`fold`、`tear_missing` |

当前目录共 250 张素材，其中 247 张启用，3 张标记为 `reject`。语义分类来自逐张人工目视复核，不使用 OCR，也不让图像特征自动决定类别。

分类字段采用明确的主标签--子标签格式：

- `main_label`：`detail`、`area`、`structure` 或 `reject`。
- `sub_label`：具体损坏形态，例如 `scratch`、`stain`、`fold`。
- `label_path`：两级路径，例如 `area--stain`。
- `tags`：辅助多标签；一张素材同时包含污渍和表面磨损时可写为 `stain;surface_wear`。
- `primary_category`：为现有运行时代码保留的兼容字段，值必须与 `sub_label` 相同。
- `label_source`：固定为 `manual_visual`，表明语义标签来自人工视觉复核。

`mixed` 不再是主类或子类。多种损坏共存时，以主导用途确定 `sub_label`，其余形态写入 `tags`。素材尺度使用 `fine`、`linear`、`coarse`、`multi_scale`，避免再次混淆。

- 必选组合只保证每档的类别覆盖；最终层数由目标破损分数决定。
- 轻度：目标分数 15-34，至少覆盖细节或表面类。
- 中度：目标分数 35-64，至少覆盖细节类 + 面积类。
- 重度：目标分数 65-90，至少覆盖细节类 + 面积类 + 结构类。
- 生成器会持续加入未重复素材，直到组合达到目标分数；不使用固定的 1/2/3 张纹理上限。

重新分析并在 `scripts/synthetic/texture_reviews/` 输出分类总览：

```powershell
python -m scripts.synthetic.analyze_textures
```

## 用法（从仓库根目录）

```powershell
# 一条命令跑全部 16 项可用性验证（约 6 秒，无需真实数据）
python -m scripts.synthetic.verify

# 生成演示用确定性小样本
python -m scripts.synthetic.generate --make-fixture

# 对真实批次生成退化对（只取 quality_label=clean_target_ok 的图）
python -m scripts.synthetic.generate --batch-id 20260727_loc_pilot_001 --sample 20
python -m scripts.synthetic.generate --batch-id 20260727_loc_pilot_001 --limit 100

# 指定其他纹理目录和分类目录
python -m scripts.synthetic.generate --clean-dir <clean_dir> --output-dir <output_root> `
  --texture-dir <texture_dir> --texture-catalog <catalog.csv>

# 对已生成目录做自动验收（失败退出码 2）
python -m scripts.synthetic.generate --verify-only --output-dir <output_root>
```

生成命令默认在 `<output_root>/MASK_HANDOFF.md` 写入按标签组合整理的 mask
交接表；也可以显式指定 `--handoff-md <path>`。交接表中的图片使用相对路径，
可随生成目录一起提交或交接。

## 产出格式

```text
<output_root>/
├── target/<source_id>.png                      # 干净目标图
├── degraded/<source_id>_{light,medium,heavy}.png
├── masks/<source_id>_{light,medium,heavy}.png  # 0=干净，255=损坏（来自损坏层 alpha）
├── metadata/<source_id>.json                   # 单文件，内含 light/medium/heavy 三档各自的 seed + 每步退化参数
├── collages/<source_id>_comparison.png         # 供人工审核的对比拼图
├── manifest.csv / summary.json / contact_sheet.jpg
└── MASK_HANDOFF.md                             # 按主标签--子标签组合的全部 mask 表格
```

## 关键约定

- mask 直接来自所选素材的有效像素，不是检测器预测或程序绘制。
- mask 严格二值 `{0,255}`，与 target/degraded 尺寸一致；轻度 1%-8%，中度 5%-20%，重度 15%-32%。
- 每个变体独立 seed，记录在单文件 metadata 的对应档位里；同参数重跑字节级一致（确定性见 engine.py）。
- 所有当前合成样本均为 `material_only` + `local_repair`，mask 必须非空。
- 自动验收要求遮罩外 target 与 degraded 逐像素完全一致，以证明没有全局或程序生成做旧。
- `evaluate_damage_score()` 输出 0-100 分，并记录覆盖率、素材强度、层数存在度和标签多样性四个分量。
- manifest 和 metadata 记录 `damage_score`、`layer_count`、`label_combination`，以及每个素材层的评分明细。
- metadata 记录目录 SHA-256、主标签--子标签、辅助标签、素材文件、变换、阈值、透明度和混合模式。
- `MASK_HANDOFF.md` 按标签组合列出每一张实际生成的 mask、来源、等级、分数、覆盖率、层数和素材链接。
- 训练侧适配（`find_training_pairs` 后缀匹配、`prepare_masks` precomputed 模式）为后续工作，当前未实现。
