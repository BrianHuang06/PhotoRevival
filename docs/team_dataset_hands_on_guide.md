# 四人数据集实操手册

> 已按岗位拆分为四份独立手册，统一入口见 `docs/team_dataset_manuals/README.md`。

这份手册用于带四名成员从原始图片开始，完成授权、清洗、标注、质检和版本冻结。第一次先用现有 57 张 Library of Congress 照片跑通流程，批次名固定为：

当前阶段只做线上公开数据采集和官方 API 爬虫，暂不安排线下走访、现场扫描或家庭相册征集。

```text
20260727_loc_pilot_001
```

这 57 张属于真实老照片候选数据，只能用于真实损坏验证、测试和案例分析，不能当作“无损干净目标图”生成训练配对。

## 一、开始前由组长完成

### 1. 确定四人身份

| 角色 | 建议负责人 | 最终交付 |
|---|---|---|
| A 数据采集、爬虫与授权 | 组员 1 | 爬虫、原始图片、来源、许可证和哈希完整的批次 |
| B 清洗与基础标注 | 组员 2 | 合格图、拒绝图、清洗报告和抽样拼图 |
| C 损坏标注与掩码 | 组员 3 | 每张照片的损坏等级，代表样本的像素 mask |
| D 质检与版本 | 组长/组员 4 | 抽查记录、划分结果、最终审计和数据集版本 |

组长把真实姓名填入命令中的 `姓名`。不要四个人共用同一个名字，否则无法追责和复核。

### 2. 所有人统一环境

在项目根目录打开 PowerShell：

```powershell
cd C:\Users\24211\PycharmProjects\PythonProject
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

看到命令行前出现 `(.venv)` 即可。图片放共享网盘，代码、CSV 清单和文档放 Git。不要把整个 `data` 目录提交到普通 Git。

### 3. 每批只处理 50-100 张

批次名格式：

```text
YYYYMMDD_来源_序号
20260727_loc_pilot_001
20260728_gzlib_001
```

每批必须整批交接。任何队员都不能只在群里发几张图片，而不发 manifest 和报告。

## 二、第 1 步：A 做来源和授权

### A-1 运行或维护采集爬虫

A 负责运行 `scripts/data/download_loc_dataset.py`，并维护后续新增来源的爬虫。正式下载前先用 `--limit 5` 试跑，检查 manifest 和授权字段；通过后再按 50-100 张一个批次扩大。完整参数和故障处理见 `docs/team_dataset_manuals/A_acquisition_crawler_rights.md`。

### A-2 检查原始数据

打开：

```text
data/raw/library_of_congress/daguerreotypes/
```

必须看到：

```text
images/
manifest.jsonl
summary.json
contact_sheet.jpg
README.md
```

逐项检查：

1. `manifest.jsonl` 中每张图有原始项目页 `item_url`。
2. `license` 为项目白名单中的值，当前 LOC 应为 `Public Domain`。
3. 文件名能在 `images` 中找到。
4. 不修改、不压缩、不覆盖原始图片。

### A-3 建立首个工作批次

```powershell
python -m scripts.data.team_dataset_workflow import-loc `
  --batch-id 20260727_loc_pilot_001 `
  --operator "A的姓名"
```

成功标志：终端显示 `57 RIGHTS_OK`。交付目录：

```text
data/work/batches/20260727_loc_pilot_001/01_rights_ok/
├── images/
├── manifest.csv
└── intake_report.json
```

### A-4 A 的人工抽查

随机打开 10 张图片，并在 `manifest.csv` 中核对对应网页、License 和哈希。License 为空或不在白名单时，将该行状态改为 `NEW`，在 `notes` 写明原因并通知 D。`rights_url` 和网页版权描述只作存档，不参与版权结论。

### A-5 发给群里的交接信息

```text
角色：A
批次：20260727_loc_pilot_001
来源：Library of Congress Daguerreotypes
导入：57 张
授权：Public Domain，已抽查 10 张
交付：data/work/batches/20260727_loc_pilot_001/01_rights_ok
下一位：B
```

## 三、第 2 步：B 做清洗和基础标注

### B-1 运行自动清洗

```powershell
python -m scripts.data.team_dataset_workflow clean `
  --batch-id 20260727_loc_pilot_001 `
  --operator "B的姓名" `
  --min-short-side 768
```

自动完成：

1. 图片解码检查。
2. EXIF 方向修正。
3. 转换为 RGB 工作副本。
4. 短边 768 像素门槛检查。
5. SHA-256 完全重复检查。
6. accepted/rejected 分流。
7. 生成 `contact_sheet.jpg` 和 `cleaning_report.json`。

自动检查不能识别所有水印、错误裁剪和内容质量，因此下一步不能省略。

### B-2 人工逐图检查

打开：

```text
data/work/batches/20260727_loc_pilot_001/02_cleaned/contact_sheet.jpg
data/work/batches/20260727_loc_pilot_001/02_cleaned/accepted/
```

每张图依次回答：

1. 是否有图库水印、字幕或后期拼接？有则拒绝。
2. 人脸或主体是否能辨认？不能辨认则在 `notes` 说明。
3. 是否严重失焦或压缩？记录到 `quality_label`。
4. 内容属于单人、多人、室内、室外、建筑、文档还是其他？填入 `content_type`。
5. 是否与前面照片高度相似？疑似重复先写 `possible_duplicate`，交 D 复核。

拒绝图片时：在 `manifest.csv` 把 `status` 改为 `REJECTED`，填写 `quality_label` 和 `notes`。不要删除 `01_rights_ok/images` 中的原始文件。

### B-3 D 第一次抽查

D 随机检查：

- 10 条授权记录。
- 20 张 accepted。
- 最多 10 张 rejected。
- 重点检查水印、错误方向、重复和不合理拒绝。

通过后在群里回复 `B阶段通过，可以进入C阶段`。不通过时列出具体 `source_id`，B 修正 CSV 后重新交付。

## 四、第 3 步：C 做真实损坏标注

### C-1 生成标注任务表

```powershell
python -m scripts.data.team_dataset_workflow prepare-annotations `
  --batch-id 20260727_loc_pilot_001 `
  --annotator "C的姓名"
```

打开：

```text
data/work/batches/20260727_loc_pilot_001/03_annotation/damage_annotations.csv
```

每种损坏只能填整数：

```text
0 = 没有
1 = 轻度，局部可见，不影响主体
2 = 中度，明显影响观感或局部内容
3 = 重度，大面积损坏或主体信息缺失
```

### C-2 逐张标注

按 `source_id` 打开对应照片，依次填写：

```text
scratch       划痕
crack         裂纹
fold          折痕
dust          灰尘点
stain         污渍
mold          霉斑
tear          撕裂
missing_region 内容缺失
fading        褪色
color_cast    偏色、泛黄
blur          模糊
noise         颗粒或扫描噪声
jpeg_artifact 压缩块
uneven_exposure 曝光不均
```

规则：

1. 看不清时填 `uncertain=1`，在 `notes` 写原因。
2. 每行完成后填 `annotator`，状态改为 `ANNOTATED`。
3. 模糊、褪色、噪声不需要画局部 mask。
4. 只有划痕、裂纹、撕裂、缺失等局部重建区域才画 mask。
5. mask 为单通道 PNG，黑色 0 表示完好，白色 255 表示损坏，尺寸必须与原图一致。

### C-3 双人复核

D 至少复核全部 57 张的损坏等级。发现分歧时，两人先独立记录，再共同讨论；无法确定时由 A 或 B 作为第三人裁决。复核完成后填 `reviewer`，把状态改为 `REVIEWED`。

首批用于统一标准，四人一起选出：

- 5 张轻度损坏示例。
- 5 张中度损坏示例。
- 5 张重度损坏示例。
- 5 张容易误标的反例。

这些示例以后作为新批次的标注基准。

## 五、第 4 步：D 做机器验收

C 和 D 都完成后运行：

```powershell
python -m scripts.data.team_dataset_workflow validate-annotations `
  --batch-id 20260727_loc_pilot_001
```

验收会检查：

1. 所有状态是否为 `ANNOTATED` 或 `REVIEWED`。
2. 每种损坏是否只填 0、1、2、3。
3. 标注人是否非空。
4. 是否至少记录一种可见损坏。

失败时打开：

```text
data/work/batches/20260727_loc_pilot_001/04_qa/annotation_validation.json
```

根据具体行号退回 C 修正，直到显示 `validation passed`。机器通过后，D 仍需做 10% 图像与标签人工抽查。

随时查看批次进度：

```powershell
python -m scripts.data.team_dataset_workflow status `
  --batch-id 20260727_loc_pilot_001
```

## 六、第 5 步：D 划分与冻结

首批只有 57 张，主要用于跑通流程，不立即冻结为 `dataset_v1`。累计达到目标后，D 才执行正式划分：

1. 按 `source_id` 分组，先排除近似重复和同一相册近邻。
2. 真实验证集至少 100 张，真实测试集至少 150 张。
3. 测试集不能参与调参、训练或早停选择。
4. 合成数据按原始干净图分组做 80/10/10 划分。
5. 同一原图的裁剪和轻中重版本必须在同一集合。
6. 运行 `python -m scripts.diagnostics.audit_dataset` 生成最终报告。
7. 创建 `DATASET_CARD.md`，写明来源、许可、数量、分布、已知偏差和禁止用途。
8. 保存最终 manifest 的 SHA-256，版本命名为 `photo_revival_dataset_v1_YYYYMMDD`。

## 七、干净目标图与合成训练数据的并行流程

现有 LOC 数据不是干净目标图。A 还要并行寻找授权明确、分辨率足够、无明显损坏的照片。每批依然先走 A、B、D 第一次抽查，再交 C。

C 对每张合格干净图生成轻、中、重三组退化：

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

每个 metadata 必须保存随机种子和全部退化参数。mask 必须直接来自局部损坏图层，不能用检测模型预测结果冒充真值。全局褪色、噪声和模糊标记为全图修复任务，不伪造极小 mask。

正式开始大规模合成前，C 先提交 20 张、共 60 组样本，四人共同看对比拼图。只有损坏自然、轻中重可区分、mask 对齐且覆盖合理，才能扩到 1,000 张目标图和 3,000 组训练对。

## 八、每天怎么开工和收工

每天开始：

1. D 在群里公布每人当天批次和数量。
2. 每人确认自己接到的输入 manifest 版本。
3. 先处理 5 张测试标准是否一致，再批量处理。
4. 遇到争议立即记录 `source_id`，不要私自猜测。

每天收工统一汇报：

```text
姓名/角色：
日期：
今日批次：
输入数量：
完成数量：
拒绝或退回数量：
主要问题：
输出目录：
需要谁复核：
明日计划：
```

组长只看三件事：今天状态前进了多少、哪些样本被退回、是否有人绕过 manifest 直接传图片。

## 九、首批完成标准

- [ ] A：57 张均可追溯，10 张人工授权抽查完成。
- [ ] B：57 张均完成自动清洗和人工内容检查。
- [ ] D：第一次抽查完成并留下问题清单。
- [ ] C：所有损坏字段均填写 0-3，标注人非空。
- [ ] D：所有行完成复核，争议有裁决记录。
- [ ] D：自动验收通过，10% 人工抽查通过。
- [ ] 全员：形成 20 张统一标注示例。
- [ ] 组长：记录首批耗时，调整下一批每人日工作量。

首批不是为了训练，而是为了证明团队可以稳定地产出可信数据。首批标准统一后，再扩充数量会快很多。
