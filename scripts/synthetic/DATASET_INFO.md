# 数据集信息

## 数据内容

当前数据集包含 2101 张照片：

| 类别 | 数量 |
|---|---:|
| `damaged_obvious` | 1239 |
| `clean_near` | 862 |

图片位于：

```text
dataset_images/damaged_obvious/
dataset_images/clean_near/
```

## 读取方式

1. 使用 `indexes/dataset_index.csv` 获取全部图片路径和文件信息；
2. 使用 `labels/dataset_labels.csv` 获取图片类别和破损标签；
3. 使用 `indexes/damaged_index.csv` 或 `indexes/clean_index.csv` 按类别筛选；
4. 使用 `integrity/PACKAGE_CONTENTS.csv` 校验传输后的文件。

## 标签规则

C 标签为以下 14 个破损维度的多标签 `0/1`：

```text
scratch, crack, fold, dust, stain, mold, tear,
missing_region, fading, color_cast, blur, noise,
jpeg_artifact, uneven_exposure
```

黑白属性只记录为 `color_mode=grayscale`。纹理素材使用主标签--子标签格式，
例如 `detail--scratch`、`area--stain`、`structure--fold`。

## 文件位置

- 图片：`dataset_images/`
- 图片索引：`indexes/`
- 标签：`labels/`
- 来源和许可：`provenance/source_manifest.csv`
- 统计：`metadata/dataset_summary.json`
- 完整性清单：`integrity/PACKAGE_CONTENTS.csv`
