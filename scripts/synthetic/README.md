# 合成退化流水线（数据集 C 岗）

本目录是数据集建设中 **C 岗：合成退化 + mask 生成** 的完整实现，从干净目标图生成可复现的轻/中/重退化训练对。真实照片的人工 mask 标注不在此目录。

## 文件拆分

| 文件 | 职责 |
|---|---|
| `engine.py` | **退化引擎**（纯 numpy + Pillow，无 torch）。三类退化（全局/局部/成像）、RGBA 损坏层 → mask、覆盖率控制、确定性 RNG、保存与拼图 |
| `generate.py` | **生成入口**（CLI）。从 `02_cleaned` 取 `clean_target_ok` 图片，输出到 `<batch>/05_synthetic_pairs` |
| `verify.py` | **自动验证**（16 项可用性检查，含 fixture 演练与确定性复现） |
| `README.md` | 本说明 |

依赖：仅 numpy、Pillow。`photo_revival` 包仅用于路径常量（`generate.py`），未安装时脚本会自动把 `src/` 加入 `sys.path`。

## 用法（从仓库根目录）

```powershell
# 一条命令跑全部 16 项可用性验证（约 6 秒，无需真实数据）
python -m scripts.synthetic.verify

# 生成演示用确定性小样本
python -m scripts.synthetic.generate --make-fixture

# 对真实批次生成退化对（只取 quality_label=clean_target_ok 的图）
python -m scripts.synthetic.generate --batch-id 20260727_loc_pilot_001 --sample 20
python -m scripts.synthetic.generate --batch-id 20260727_loc_pilot_001 --limit 100

# 对已生成目录做自动验收（失败退出码 2）
python -m scripts.synthetic.generate --verify-only --output-dir <output_root>
```

## 产出格式

```text
<output_root>/
├── target/<source_id>.png                      # 干净目标图
├── degraded/<source_id>_{light,medium,heavy}.png
├── masks/<source_id>_{light,medium,heavy}.png  # 0=干净，255=损坏（来自损坏层 alpha）
├── metadata/<source_id>_{light,medium,heavy}.json  # seed + 每步退化参数
├── collages/<source_id>_comparison.png         # 供人工审核的对比拼图
├── manifest.csv / summary.json / contact_sheet.jpg
```

## 关键约定

- mask 直接来自局部损坏图层的 alpha，不是检测器预测。
- mask 严格二值 {0,255}，与 target/degraded 尺寸一致，覆盖率目标 1%-35%。
- 每个变体独立 seed，同参数重跑字节级一致（确定性见 engine.py）。
- `--global-only` 生成全图修复样本（`global_restoration`，mask 全零）。
- 训练侧适配（`find_training_pairs` 后缀匹配、`prepare_masks` precomputed 模式）为后续工作，当前未实现。
