# B 岗位手册：清洗与基础标注

## 1. 你的任务

你负责判断图片能否成为可信的数据样本，包括能否打开、分辨率是否足够、是否重复、是否有水印，以及图片内容属于什么类别。

你的状态流转是：

```text
RIGHTS_OK -> CLEANED
RIGHTS_OK -> REJECTED
```

没有 `RIGHTS_OK` 的图片一律不处理。

## 2. 接收检查

从 A 接收完整批次，必须包括：

```text
images/
manifest.csv 或 manifest.jsonl
license 字段
```

先检查：

1. manifest 行数和图片数是否一致。
2. 所有待处理行状态是否为 `RIGHTS_OK`。
3. `source_id`、来源和 `license` 是否非空。
4. 批次 ID 是否与目录名一致。

不满足时整批退回 A，不要自己猜测或修改 License。

## 3. 运行自动清洗

当前试运行批次：

```powershell
cd C:\Users\24211\PycharmProjects\PythonProject
.\.venv\Scripts\python.exe -m scripts.data.team_dataset_workflow clean `
  --batch-id 20260727_loc_pilot_001 `
  --operator "你的真实姓名" `
  --min-short-side 768
```

程序会完成：

1. 解码检查。
2. EXIF 方向修正。
3. 转换为 RGB 工作副本。
4. 分辨率检查。
5. SHA-256 完全重复检查。
6. accepted/rejected 分流。
7. 生成清洗报告和抽样拼图。

输出：

```text
data/work/batches/<batch_id>/02_cleaned/
├── accepted/
├── rejected/
├── manifest.csv
├── contact_sheet.jpg
└── cleaning_report.json
```

## 4. 人工逐图检查

程序不能可靠发现所有视觉问题。先看 `contact_sheet.jpg`，再逐张打开 `accepted`。

### 4.1 必须拒绝

- 商业图库或明显版权水印。
- 无法辨认主体的严重模糊。
- 短边低于 512 像素。
- 明显拼图、截图、网页边框或聊天界面。
- 大面积字幕、日期戳或与任务无关的文字覆盖。
- 完全重复或仅改文件名的重复图。
- 严重 JPEG 块导致细节不可用。
- 错误裁剪，主体重要部位被意外截断。

### 4.2 可以保留但必须标记

- 自然划痕、裂纹、霉斑和褪色：`existing_damage`。
- 黑白或褐色照片：保留原色，不强制上色。
- 合理的原始相纸边缘：写入 `notes`。
- 历史扫描产生的轻微颗粒：标记 `noise`。
- 真实老照片轻度失焦：标记 `blurred`，交 D 判断用途。

## 5. 基础内容标签

在 `content_type` 中选择一个主标签：

```text
single_portrait
group_portrait
indoor_scene
outdoor_scene
architecture
document
other
```

在 `quality_label` 中填写主要质量结论：

```text
clean_target_ok
existing_damage
blurred
watermark
compression_artifact
low_resolution
bad_crop
duplicate
```

一张图片存在多个问题时，把最主要问题写在 `quality_label`，其他问题写入 `notes`。

## 6. 区分干净目标图和真实损坏图

### 干净目标图

要求主体清楚、无明显损坏、无水印、无严重噪声，短边最低 512，优先达到 768。它可以用于生成合成退化训练对。

### 真实损坏图

存在真实划痕、缺损、褪色等问题。它主要进入真实验证集、真实测试集或案例展示，不能直接作为干净目标图。

当前 57 张 LOC 达盖尔银版照片属于真实损坏图，即使自动清洗通过，也不能标成 `clean_target_ok`。

## 7. 拒绝操作

拒绝时：

1. 将对应行 `status` 改为 `REJECTED`。
2. 在 `quality_label` 填写固定拒绝原因。
3. 在 `notes` 写清肉眼判断。
4. 把工作副本放入 `rejected`。
5. 不删除 A 阶段或 `data/raw` 中的原始文件。

示例：

```text
quality_label=watermark
notes=右下角存在商业图库半透明水印，无法安全裁除
```

## 8. 近似重复处理

发现同一人物、同一姿势或同一照片的不同裁剪时：

1. 不要立即删除。
2. 在两行 `notes` 写 `possible_duplicate` 和对方 `source_id`。
3. 保留清晰度更高、来源更完整的一张作为候选。
4. 交 D 复核感知哈希和来源。
5. D 决定保留、分组或拒绝。

## 9. 交给 D 的第一次质检

D 会抽查：

- 20 张 accepted。
- 最多 10 张 rejected。
- 水印、方向、裁剪和重复情况。
- 干净目标图与真实损坏图是否混淆。

收到退回列表后，只修改指定 `source_id`，并在 `notes` 记录修改原因和日期。修正完成后重新通知 D。

## 10. 交付前检查

- [ ] 所有图片都能正常打开。
- [ ] accepted 中没有短边低于门槛的图片。
- [ ] 完全重复为 0。
- [ ] 水印漏检为 0。
- [ ] 每张图片都有内容标签。
- [ ] 每张图片都有质量标签。
- [ ] 所有拒绝图都有明确原因。
- [ ] 已生成并人工查看 contact sheet。
- [ ] 没有修改或删除原始文件。

## 11. 群内交接模板

```text
姓名/角色：李四 / B
批次：20260802_gzlib_001
输入：92
CLEANED：80
REJECTED：12
拒绝原因：水印 3、低分辨率 4、重复 2、严重模糊 3
真实损坏图：35
干净目标候选：45
交付目录：data/work/batches/20260802_gzlib_001/02_cleaned
需要 D：第一次抽查
```
