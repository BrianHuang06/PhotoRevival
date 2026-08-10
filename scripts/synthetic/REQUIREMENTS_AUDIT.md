# C 岗全素材合成需求对照审查

审查日期：2026-08-10

对照来源：`docs/team_dataset_manuals/C_damage_annotation.md`（只读）

审查范围：`scripts/synthetic` 内的合成退化、素材分类、mask、元数据、生成 CLI 和自动验收。

| 序号 | C 岗要求 | 当前实现 | 状态 |
|---:|---|---|---|
| 1 | 只使用 `quality_label=clean_target_ok` 的目标图 | `generate._select_sources` 按 manifest 过滤并保留 `source_id` | 已实现 |
| 2 | 每张目标图生成轻、中、重三个版本 | `SEVERITY_PROFILE` 定义三档，CLI 默认生成三档 | 已实现 |
| 3 | 所有做旧来自素材 | 引擎只允许 `texture_composite`，不生成偏色、褪色、暗角、噪声、模糊、JPEG 或损坏形状 | 已实现 |
| 4 | 无标签纹理需要分类 | `texture_catalog.csv` 覆盖 250 张素材，包含主类别、多标签和审计字段 | 已实现 |
| 5 | 按类别组合素材 | 细节、面积、结构三组共 10 个可用类别 | 已实现 |
| 6 | 轻、中、重组合有明确差异 | 轻度 1 层；中度 2 层；重度 3 层 | 已实现 |
| 7 | 程序只变换和组合素材 | 只允许裁剪、缩放、90 度旋转、翻转、阈值、透明度及两种混合 | 已实现 |
| 8 | mask 来自素材真值 | 每层从纹理有效像素产生二值选择，最终 mask 为各层并集 | 已实现 |
| 9 | mask 同尺寸、二值且非空 | 保存为 `{0,255}` 单通道 PNG，并由验收强制检查 | 已实现 |
| 10 | 覆盖率分档 | 轻度 1%-8%，中度 5%-20%，重度 15%-32% | 已实现 |
| 11 | 遮罩外不得改变 | 输出前回填原图遮罩外像素，验收逐像素比较 | 已实现 |
| 12 | 保存种子和全部参数 | 每个 source 的 metadata 记录三档 seed、素材、类别、变换、阈值、透明度与混合模式 | 已实现 |
| 13 | 素材来源可审计 | metadata 和 summary 保存分类目录 SHA-256 及素材字段 | 已实现 |
| 14 | 同参数可复现 | 每个变体使用独立 NumPy RNG，自动验收重建图像和 mask | 已实现 |
| 15 | 输出结构完整 | 生成 target、degraded、masks、metadata、collages、manifest、summary 和 contact sheet | 已实现 |
| 16 | 自动验收 | `python -m scripts.synthetic.verify` 执行 16 项端到端检查 | 已实现 |
| 17 | 先交 20 张、60 组试样给 D 审核 | 生成能力具备 | 尚未执行人工审核流程 |
| 18 | 扩大正式生产规模 | 支持批次、limit、sample、幂等重跑和覆盖写入 | 尚未执行正式批量生产 |

## 分类摘要

250 张素材中 247 张启用，3 张标记为 `reject`。

| 类别 | 启用素材数 |
|---|---:|
| `scratch` | 12 |
| `crack` | 8 |
| `dust` | 26 |
| `stain` | 20 |
| `mold` | 69 |
| `surface_wear` | 46 |
| `mixed` | 10 |
| `edge_wear` | 21 |
| `fold` | 29 |
| `tear_missing` | 6 |

分类目录 SHA-256：

```text
36ae3c8f3cb56d2cde0afcb8bad73fba0543f9353fa0e4c701c32895b833a90b
```

## 结论

`scripts/synthetic` 的正式执行路径符合“全部做旧来自素材，并按类别组合使用”的代码要求。剩余工作是补足 20 张/60 组首批试样并完成人工质检，通过后再进行正式批量生产。
