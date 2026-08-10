# Mask 素材标签与破损评分交接表

## 交付概况

- 生成时间：2026-08-10T16:11:33.780703+00:00
- 样例目录：`generated_preview`
- Mask 数量：12
- 素材目录哈希：`910e45249413a129ba6d1b2d850b6acbbaaf63dadb28abbea9f167b1de0a0c16`
- 生成模式：仅组合已分类素材，不使用程序生成的做旧纹理。
- 标签来源：人工视觉分类（`manual_visual`），不是 OCR。
- 标签格式：`主标签--子标签`；多标签组合用 ` + ` 连接。

## 破损评分

| 分量 | 最高分 | 含义 |
|---|---:|---|
| 覆盖率 | 50 | 最终二值 mask 的受损像素比例，32% 覆盖时达到该分量上限 |
| 素材强度 | 32 | 各层覆盖率、透明度、破损类别、尺度、密度和分布加权累计 |
| 层数存在度 | 8 | 随实际组合层数对数增长，不用固定素材数量限制等级 |
| 标签多样性 | 10 | 组合中不同主标签和子标签的丰富度 |

| 等级 | 分数范围 | 覆盖率约束 |
|---|---:|---:|
| light | 15-34 | 1%-8% |
| medium | 35-64 | 5%-20% |
| heavy | 65-90 | 15%-32% |

## 标签使用统计

| 主标签--子标签 | 使用该标签的 Mask 数 |
|---|---:|
| `area--mold` | 5 |
| `area--stain` | 5 |
| `area--surface_wear` | 7 |
| `detail--crack` | 5 |
| `detail--dust` | 4 |
| `detail--scratch` | 8 |
| `structure--edge_wear` | 2 |
| `structure--fold` | 2 |
| `structure--tear_missing` | 2 |

## 素材使用统计

| 素材文件 | 使用次数 |
|---|---:|
| `001.jpg` | 2 |
| `012.jpg` | 1 |
| `019.jpg` | 1 |
| `031.jpg` | 2 |
| `035.jpg` | 1 |
| `048.jpg` | 1 |
| `057.jpg` | 1 |
| `058.jpg` | 1 |
| `077.jpg` | 1 |
| `085.jpg` | 1 |
| `086.jpg` | 1 |
| `093.jpg` | 1 |
| `102.jpg` | 1 |
| `104.jpg` | 1 |
| `106.jpg` | 2 |
| `109.jpg` | 1 |
| `113.jpg` | 1 |
| `115.jpg` | 1 |
| `116.jpg` | 2 |
| `118.jpg` | 1 |
| `119.jpg` | 1 |
| `134.jpg` | 1 |
| `135.jpg` | 2 |
| `141.jpg` | 2 |
| `142.jpg` | 2 |
| `151.jpg` | 1 |
| `153.jpg` | 2 |
| `154.jpg` | 1 |
| `176.jpg` | 1 |
| `184.jpg` | 1 |
| `185.jpg` | 2 |
| `187.jpg` | 1 |
| `189.jpg` | 1 |
| `196.jpg` | 1 |
| `198.jpg` | 2 |
| `202.jpg` | 1 |
| `206.jpg` | 1 |
| `209.jpg` | 1 |
| `215.jpg` | 1 |
| `238.jpg` | 1 |
| `240.jpg` | 3 |
| `241.jpg` | 1 |
| `243.jpg` | 2 |
| `244.jpg` | 1 |
| `246.jpg` | 2 |
| `249.jpg` | 1 |

## 全部 Mask

### `detail--crack + area--stain + area--mold`

| Mask | 来源 / 等级 | 分数 | 覆盖率 | 层数 | 素材 | 文件 |
|---|---|---:|---:|---:|---|---|
| ![demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106) medium](<generated_preview/masks/demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106)_medium.png>) | `demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106)` / `medium` | 60.22 | 19.66% | 7 | `135.jpg`<br>`113.jpg`<br>`249.jpg`<br>`109.jpg`<br>`142.jpg`<br>`240.jpg`<br>`151.jpg` | [degraded](<generated_preview/degraded/demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106)_medium.png>)<br>[metadata](<generated_preview/metadata/demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106).json>) |

### `detail--crack + area--surface_wear + area--stain`

| Mask | 来源 / 等级 | 分数 | 覆盖率 | 层数 | 素材 | 文件 |
|---|---|---:|---:|---:|---|---|
| ![demo_1_Portrait_of_a_woman,_1850_(13999672088) medium](<generated_preview/masks/demo_1_Portrait_of_a_woman,_1850_(13999672088)_medium.png>) | `demo_1_Portrait_of_a_woman,_1850_(13999672088)` / `medium` | 55.90 | 16.24% | 6 | `135.jpg`<br>`189.jpg`<br>`154.jpg`<br>`116.jpg`<br>`185.jpg`<br>`240.jpg` | [degraded](<generated_preview/degraded/demo_1_Portrait_of_a_woman,_1850_(13999672088)_medium.png>)<br>[metadata](<generated_preview/metadata/demo_1_Portrait_of_a_woman,_1850_(13999672088).json>) |

### `detail--dust + area--mold + structure--edge_wear + area--surface_wear + structure--fold`

| Mask | 来源 / 等级 | 分数 | 覆盖率 | 层数 | 素材 | 文件 |
|---|---|---:|---:|---:|---|---|
| ![demo_1_Portrait_of_a_woman,_1850_(13999672088) heavy](<generated_preview/masks/demo_1_Portrait_of_a_woman,_1850_(13999672088)_heavy.png>) | `demo_1_Portrait_of_a_woman,_1850_(13999672088)` / `heavy` | 74.01 | 21.27% | 5 | `244.jpg`<br>`198.jpg`<br>`184.jpg`<br>`104.jpg`<br>`153.jpg` | [degraded](<generated_preview/degraded/demo_1_Portrait_of_a_woman,_1850_(13999672088)_heavy.png>)<br>[metadata](<generated_preview/metadata/demo_1_Portrait_of_a_woman,_1850_(13999672088).json>) |

### `detail--dust + area--stain + structure--fold + detail--crack + structure--tear_missing`

| Mask | 来源 / 等级 | 分数 | 覆盖率 | 层数 | 素材 | 文件 |
|---|---|---:|---:|---:|---|---|
| ![demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform heavy](<generated_preview/masks/demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform_heavy.png>) | `demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform` / `heavy` | 78.40 | 22.56% | 6 | `102.jpg`<br>`119.jpg`<br>`153.jpg`<br>`116.jpg`<br>`141.jpg`<br>`176.jpg` | [degraded](<generated_preview/degraded/demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform_heavy.png>)<br>[metadata](<generated_preview/metadata/demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform.json>) |

### `detail--scratch + area--mold + structure--tear_missing`

| Mask | 来源 / 等级 | 分数 | 覆盖率 | 层数 | 素材 | 文件 |
|---|---|---:|---:|---:|---|---|
| ![demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106) heavy](<generated_preview/masks/demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106)_heavy.png>) | `demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106)` / `heavy` | 69.86 | 17.79% | 4 | `246.jpg`<br>`198.jpg`<br>`115.jpg`<br>`141.jpg` | [degraded](<generated_preview/degraded/demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106)_heavy.png>)<br>[metadata](<generated_preview/metadata/demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106).json>) |

### `detail--scratch + area--stain + structure--edge_wear + area--mold + detail--crack`

| Mask | 来源 / 等级 | 分数 | 覆盖率 | 层数 | 素材 | 文件 |
|---|---|---:|---:|---:|---|---|
| ![demo_2_Dewey_Arch,_New_York heavy](<generated_preview/masks/demo_2_Dewey_Arch,_New_York_heavy.png>) | `demo_2_Dewey_Arch,_New_York` / `heavy` | 70.74 | 19.22% | 5 | `209.jpg`<br>`012.jpg`<br>`057.jpg`<br>`142.jpg`<br>`185.jpg` | [degraded](<generated_preview/degraded/demo_2_Dewey_Arch,_New_York_heavy.png>)<br>[metadata](<generated_preview/metadata/demo_2_Dewey_Arch,_New_York.json>) |

### `detail--scratch + area--surface_wear`

| Mask | 来源 / 等级 | 分数 | 覆盖率 | 层数 | 素材 | 文件 |
|---|---|---:|---:|---:|---|---|
| ![demo_2_Dewey_Arch,_New_York light](<generated_preview/masks/demo_2_Dewey_Arch,_New_York_light.png>) | `demo_2_Dewey_Arch,_New_York` / `light` | 23.38 | 4.60% | 3 | `106.jpg`<br>`001.jpg`<br>`077.jpg` | [degraded](<generated_preview/degraded/demo_2_Dewey_Arch,_New_York_light.png>)<br>[metadata](<generated_preview/metadata/demo_2_Dewey_Arch,_New_York.json>) |
| ![demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106) light](<generated_preview/masks/demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106)_light.png>) | `demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106)` / `light` | 25.79 | 5.70% | 4 | `106.jpg`<br>`243.jpg`<br>`093.jpg`<br>`196.jpg` | [degraded](<generated_preview/degraded/demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106)_light.png>)<br>[metadata](<generated_preview/metadata/demo_4_Brunner_bridge_and_coal_mines,_ca_1900_(3057612106).json>) |

### `detail--scratch + area--surface_wear + area--mold`

| Mask | 来源 / 等级 | 分数 | 覆盖率 | 层数 | 素材 | 文件 |
|---|---|---:|---:|---:|---|---|
| ![demo_2_Dewey_Arch,_New_York medium](<generated_preview/masks/demo_2_Dewey_Arch,_New_York_medium.png>) | `demo_2_Dewey_Arch,_New_York` / `medium` | 42.09 | 12.16% | 5 | `035.jpg`<br>`001.jpg`<br>`241.jpg`<br>`085.jpg`<br>`238.jpg` | [degraded](<generated_preview/degraded/demo_2_Dewey_Arch,_New_York_medium.png>)<br>[metadata](<generated_preview/metadata/demo_2_Dewey_Arch,_New_York.json>) |

### `detail--scratch + area--surface_wear + area--stain + detail--crack`

| Mask | 来源 / 等级 | 分数 | 覆盖率 | 层数 | 素材 | 文件 |
|---|---|---:|---:|---:|---|---|
| ![demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform medium](<generated_preview/masks/demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform_medium.png>) | `demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform` / `medium` | 39.02 | 9.51% | 4 | `243.jpg`<br>`206.jpg`<br>`118.jpg`<br>`240.jpg` | [degraded](<generated_preview/degraded/demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform_medium.png>)<br>[metadata](<generated_preview/metadata/demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform.json>) |

### `detail--scratch + detail--dust`

| Mask | 来源 / 等级 | 分数 | 覆盖率 | 层数 | 素材 | 文件 |
|---|---|---:|---:|---:|---|---|
| ![demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform light](<generated_preview/masks/demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform_light.png>) | `demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform` / `light` | 21.65 | 4.78% | 4 | `048.jpg`<br>`031.jpg`<br>`246.jpg`<br>`215.jpg` | [degraded](<generated_preview/degraded/demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform_light.png>)<br>[metadata](<generated_preview/metadata/demo_3_Sgt._Samuel_Smith,_African_American_soldier_in_Union_uniform.json>) |

### `detail--scratch + detail--dust + area--surface_wear`

| Mask | 来源 / 等级 | 分数 | 覆盖率 | 层数 | 素材 | 文件 |
|---|---|---:|---:|---:|---|---|
| ![demo_1_Portrait_of_a_woman,_1850_(13999672088) light](<generated_preview/masks/demo_1_Portrait_of_a_woman,_1850_(13999672088)_light.png>) | `demo_1_Portrait_of_a_woman,_1850_(13999672088)` / `light` | 31.05 | 7.71% | 7 | `031.jpg`<br>`019.jpg`<br>`202.jpg`<br>`187.jpg`<br>`134.jpg`<br>`086.jpg`<br>`058.jpg` | [degraded](<generated_preview/degraded/demo_1_Portrait_of_a_woman,_1850_(13999672088)_light.png>)<br>[metadata](<generated_preview/metadata/demo_1_Portrait_of_a_woman,_1850_(13999672088).json>) |

## 交接字段

- `damage_score`：0-100 的综合破损分数。
- `layer_count`：该图片实际组合的素材层数。
- `label_combination`：按首次出现顺序去重后的标签组合。
- `damage_assessment.layer_scores`：每一层素材的评分明细。
- `category_recipe`：组合顺序、标签、类别和素材文件。
- `mask_coverage`：最终联合二值 mask 的覆盖率。
