# 数据集四人协作手册入口

本目录将数据集工作拆成四个独立岗位。每位成员先读自己的手册，组长同时阅读 D 手册和总流程。

> 当前阶段范围：只做线上公开数据源、官方 API 和合规网页采集，暂不安排档案馆走访、家庭相册征集、现场扫描或线下授权联系。

| 角色 | 手册 | 核心职责 |
|---|---|---|
| A | `A_acquisition_crawler_rights.md` | 爬虫开发与运行、数据采集、来源追溯、授权核查、原始数据入库 |
| B | `B_cleaning_and_labels.md` | 解码、分辨率、去重、质量筛选、基础内容标签 |
| C | `C_damage_annotation.md` | 损坏类型与等级、局部 mask、合成退化和参数记录 |
| D | `D_qa_and_release.md` | 双阶段质检、数据划分、防泄漏、版本冻结和验收 |

## 固定流转顺序

```text
NEW
  -> RIGHTS_OK
  -> CLEANED
  -> ANNOTATED
  -> QA_PASS
  -> TRAIN / VALIDATION / TEST
```

任何成员不得跳过状态，也不得只传图片而不传 manifest。发现问题时退回上一负责人，并使用 `source_id` 指明具体样本。

## 当前试运行批次

```text
批次 ID：20260727_loc_pilot_001
数据类型：真实损坏老照片
数量：57 张
当前状态：等待 C 人工标注
```

查看当前状态：

```powershell
cd C:\Users\24211\PycharmProjects\PythonProject
.\.venv\Scripts\python.exe -m scripts.data.team_dataset_workflow status `
  --batch-id 20260727_loc_pilot_001
```

## 公共规则

1. `data/raw` 中的原始文件只读保存，任何人不得覆盖或删除。
2. 文件名只用英文、数字、下划线和短横线。
3. 每批建议 50-100 张，必须整批交接。
4. 图片不提交普通 Git；脚本、CSV 模板、规则和小型报告进入 Git。
5. 有疑问时写入 `notes`，不要凭感觉补全授权或损坏标签。
6. 训练集、验证集和测试集必须按 `source_id` 分组划分。

完整团队总流程参见 `docs/team_dataset_hands_on_guide.md`。
