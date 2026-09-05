# CSV 文件说明

## 文件用途

| 文件 | 用途 |
|---|---|
| `indexes/dataset_index.csv` | 全部 2101 张图片的正式索引 |
| `indexes/damaged_index.csv` | 1239 张明显破损图片索引 |
| `indexes/clean_index.csv` | 862 张近似干净图片索引 |
| `labels/dataset_labels.csv` | 结果图片的类别和 C 标签 |
| `labels/damage_labels.csv` | 来源图片的破损标签与识别证据 |
| `labels/texture_catalog.csv` | 纹理素材分类和视觉属性 |
| `provenance/source_manifest.csv` | 当前结果图片的来源和许可信息 |
| `metadata/dataset_summary.json` | 当前数据集统计 |
| `integrity/PACKAGE_CONTENTS.csv` | 包内文件完整性清单 |

## 索引字段

正式图片索引包含：

- `dataset_id`、`dataset_class`、`class_seq`：图片编号和类别；
- `source_id`、`original_filename`：来源标识；
- `export_filename`、`export_path`：包内图片文件名和路径；
- `source_sha256`：来源文件哈希；
- `width`、`height`：最终图片像素尺寸；
- `export_sha256`、`file_size_bytes`、`actual_format`：正式图片文件信息；
- `content_type`、`color_mode`、`quality_label`：图片类型和图像属性；
- `damage_label_list`、`damage_label_count`：破损标签汇总；
- `texture_match_label_list`、`texture_match_scores`、`texture_match_files`：纹理匹配结果；
- 14 个 `damage_*` 字段：C 破损多标签结果。

## C 破损标签

```text
scratch, crack, fold, dust, stain, mold, tear,
missing_region, fading, color_cast, blur, noise,
jpeg_artifact, uneven_exposure
```

字段值为 `0/1`。`jpeg_artifact` 表示 JPEG 压缩伪影，不表示实体破损。
