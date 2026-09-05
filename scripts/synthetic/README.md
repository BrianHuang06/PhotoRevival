# PhotoRevival synthetic tools

本目录保存 C 岗来源整理、原图破损标注、纹理分类和可复现合成代码。
最终图片文件不放在仓库内，而在网盘资料包：

```text
D:\My_Docs\Programmes\NCIEP\My_Materials\MarkedImages
```

大文件图片位于网盘的 `MarkedImages/dataset_images/`，仓库只保存代码和非图片数据资料。
当前资料索引同步在本目录的 `dataset_info/`。

## 主要文件

| 文件 | 职责 |
|---|---|
| `source_pipeline.py` | 生成来源清单 |
| `recheck_damage_labels.py` | 按现有纹理目录比对 8 类物理破损，并检查 6 类图像属性破损 |
| `export_dataset.py` | 导出 `damaged_obvious` 和 `clean_near` 两类结果 |
| `crop_photo_frames.py` | 检测相框内框并保守裁剪，保留安全边缘；无可靠相框时原样复制 |
| `sync_dataset_metadata.py` | 以最终图片文件为准同步正式索引的尺寸、哈希、大小和格式 |
| `analyze_textures.py` | 对纹理素材做视觉属性和主标签--子标签分类 |
| `texture_assets.py` | 纹理素材路径和枚举辅助 |
| `engine.py` | 基于素材的退化、mask、损坏分数和组合逻辑 |
| `generate.py` | 可复现合成样本生成入口 |
| `verify.py` | 自动验证 |
| `DATASET_INFO.md` | 当前数据集信息 |
| `CSV_GUIDE.md` | CSV 字段和用途说明 |
| `dataset_info/` | 当前索引、标签、来源和汇总信息 |

## 标签约定

C 原图破损使用多标签 `0/1`：

```text
scratch, crack, fold, dust, stain, mold, tear,
missing_region, fading, color_cast, blur, noise,
jpeg_artifact, uneven_exposure
```

黑白图片只记录 `color_mode=grayscale`，不因为黑白自动标记为 `fading` 或
`color_cast`。

纹理素材使用主标签--子标签格式，例如：

```text
detail--scratch
area--stain
structure--fold
```

`mixed` 不作为正式主标签；混合纹理以主导用途作为主标签，其余形态写入辅助标签。
纹理语义分类采用视觉属性、规则和人工复核，不使用 OCR。

## 数据分类

导出脚本执行以下规则：

```text
排除 quality_label=clean_target_ok
damage_* 数量 > 0  -> damaged_obvious
damage_* 数量 = 0  -> clean_near
```

当前资料包含 1239 张 `damaged_obvious` 和 862 张 `clean_near`，合计 2101 张。
各类图片在网盘 `MarkedImages/dataset_images/` 下按类别编号，索引见
`dataset_info/indexes/` 和 `dataset_info/labels/`。

相框检测结果已作为正式结果使用：329 张可靠内框图片完成裁剪，1772 张保持原样。
检测到内框时向外扩展照片短边的 3.5% 作为安全边距，避免裁掉照片边缘纹理。
最终图片目录是 `MarkedImages/dataset_images/`，正式索引中的 `width`、`height`、
`export_sha256`、`file_size_bytes` 和 `actual_format` 已从磁盘实际文件同步。

## 常用命令

从项目根目录运行：

```powershell
# 生成来源资料
python -m scripts.synthetic.source_pipeline `
  --images D:\My_Docs\Programmes\NCIEP\images `
  --manifest D:\My_Docs\Programmes\NCIEP\manifest.csv `
  --output-dir scripts\synthetic

# 按现有纹理素材重新检查 C 标签
python -m scripts.synthetic.recheck_damage_labels `
  --images D:\My_Docs\Programmes\NCIEP\images `
  --source-manifest scripts\synthetic\source_manifest.csv `
  --texture-dir data\textures\Resource-Boy-Grunge-Textures\Resource Boy - Grunge Textures `
  --texture-catalog scripts\synthetic\texture_catalog.csv `
  --output scripts\synthetic\damage_labels.csv

# 导出两类结果到网盘资料包
python -m scripts.synthetic.export_dataset `
  --images D:\My_Docs\Programmes\NCIEP\images `
  --source-manifest scripts\synthetic\source_manifest.csv `
  --damage-labels scripts\synthetic\damage_labels.csv `
  --output-dir D:\My_Docs\Programmes\NCIEP\My_Materials\MarkedImages

# 检测相框并生成裁剪结果
python -m scripts.synthetic.crop_photo_frames `
  --dataset-root D:\My_Docs\Programmes\NCIEP\My_Materials\MarkedImages `
  --dataset-index D:\My_Docs\Programmes\NCIEP\My_Materials\MarkedImages\indexes\dataset_index.csv `
  --output-dir D:\My_Docs\Programmes\NCIEP\My_Materials\MarkedImages

# 图片被人工或程序裁剪后，重新同步正式索引
python -m scripts.synthetic.sync_dataset_metadata `
  --package-root D:\My_Docs\Programmes\NCIEP\My_Materials\MarkedImages

# 验证代码和合成引擎
python -m scripts.synthetic.verify
```

`generate.py` 仍用于需要时重新生成合成训练样本，但生成图片不作为当前
`MarkedImages` 的最终结果，也不应与两类来源结果混在一起。

当前 C 标签方法为 `texture_catalog_match_v1+source_attribute_v1`：
8 类物理破损与现有纹理目录进行局部视觉描述子比对；6 类没有对应纹理素材的
图像属性破损使用图像属性证据。黑白属性只记录在 `color_mode`。

## 素材规则

所有可见做旧来自 `data/textures/` 中已分类素材。程序只做裁剪、缩放、旋转、
翻转、阈值选择、透明度调整和 `screen`/`multiply` 合成；不额外生成噪声、模糊、
偏色、褪色、压缩伪影、折痕形状或缺失区域。
