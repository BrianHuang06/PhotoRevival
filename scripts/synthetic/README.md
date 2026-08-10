# 合成退化流水线（数据集 C 岗）

本目录是数据集建设中 **C 岗：合成退化 + mask 生成** 的完整实现，从干净目标图生成可复现的轻/中/重退化训练对。真实照片的人工 mask 标注不在此目录。

## 文件拆分

| 文件 | 职责 |
|---|---|
| `analyze_textures.py` | 分析无标签纹理，结合人工复核与图像特征生成多标签目录和分类总览 |
| `texture_assets.py` | 本目录内的素材路径和素材文件枚举辅助逻辑 |
| `texture_catalog.csv` | 250 张素材的分类目录、主类别、辅助标签、混合模式和启用状态 |
| `engine.py` | **严格全素材退化引擎**（numpy + Pillow）。按类别选材、组合、生成 mask、控制覆盖率并记录完整素材参数 |
| `generate.py` | **生成入口**（CLI）。从 `02_cleaned` 取 `clean_target_ok` 图片，输出到 `<batch>/05_synthetic_pairs` |
| `verify.py` | **自动验证**（16 项检查，含素材目录、确定性复现和遮罩外零改动） |
| `REQUIREMENTS_AUDIT.md` | 对照 C 岗需求的逐项实现审查与剩余工作 |
| `README.md` | 本说明 |

Python 依赖：numpy、Pillow。纹理分析脚本另需 OpenCV。生成器会只读复用现有 `scripts/data/team_dataset_workflow.py` 和 `src/photo_revival/paths.py`，不修改这些目录。

**严格素材要求**：所有可见做旧都来自 `data/textures/Resource-Boy-Grunge-Textures/Resource Boy - Grunge Textures/` 的已分类素材。程序只允许裁剪、缩放、90 度旋转、翻转、阈值选择、透明度调整和 `screen`/`multiply` 合成；不生成偏色、褪色、暗角、噪声、模糊、压缩伪影、折痕形状或缺失区域。

## 素材分类与组合

| 组别 | 类别 |
|---|---|
| 细节类 | `scratch`、`crack`、`dust` |
| 面积类 | `stain`、`mold`、`surface_wear`、`mixed` |
| 结构类 | `edge_wear`、`fold`、`tear_missing` |

当前目录共 250 张素材，其中 247 张启用，3 张标记为 `reject`。每张素材可以有多个标签，但运行时按 `primary_category` 进入对应素材池。

- 轻度：1 层，细节类或轻表面磨损。
- 中度：2 层，细节类 + 面积类。
- 重度：3 层，细节类 + 面积类 + 结构类。

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

## 产出格式

```text
<output_root>/
├── target/<source_id>.png                      # 干净目标图
├── degraded/<source_id>_{light,medium,heavy}.png
├── masks/<source_id>_{light,medium,heavy}.png  # 0=干净，255=损坏（来自损坏层 alpha）
├── metadata/<source_id>.json                   # 单文件，内含 light/medium/heavy 三档各自的 seed + 每步退化参数
├── collages/<source_id>_comparison.png         # 供人工审核的对比拼图
├── manifest.csv / summary.json / contact_sheet.jpg
```

## 关键约定

- mask 直接来自所选素材的有效像素，不是检测器预测或程序绘制。
- mask 严格二值 `{0,255}`，与 target/degraded 尺寸一致；轻度 1%-8%，中度 5%-20%，重度 15%-32%。
- 每个变体独立 seed，记录在单文件 metadata 的对应档位里；同参数重跑字节级一致（确定性见 engine.py）。
- 所有当前合成样本均为 `material_only` + `local_repair`，mask 必须非空。
- 自动验收要求遮罩外 target 与 degraded 逐像素完全一致，以证明没有全局或程序生成做旧。
- metadata 记录目录 SHA-256、类别配方、素材文件、变换、阈值、透明度和混合模式。
- 训练侧适配（`find_training_pairs` 后缀匹配、`prepare_masks` precomputed 模式）为后续工作，当前未实现。
