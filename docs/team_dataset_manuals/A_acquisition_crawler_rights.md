# A 岗位手册：数据采集、爬虫与授权

## 1. 你的任务

你负责完成数据进入项目的全过程：寻找可靠来源、开发和维护爬虫、控制下载速度、保存原始文件和元数据、自动去重、检查 License，并保证以后能够追溯到原始页面。你的工作完成后，B 才能开始清洗。

你的职责包括：

1. 调研官方数据源、API 和使用条款。
2. 编写、测试和维护下载脚本。
3. 小规模试跑，确认字段、质量和 License 白名单过滤正确。
4. 批量下载并处理网络重试、断点续传和失败记录。
5. 保存原图、来源 URL、许可证、哈希和下载时间。
6. 生成统计报告和抽样拼图，检查 License 字段完整性。
7. 将合格批次交给 B。

你的最终状态是：

```text
NEW -> RIGHTS_OK
```

你不负责判断划痕等级，也不要修改图片内容。版权结论只依据标准化后的 `license` 字段，不再要求队员解释网页版权文字。

当前阶段只处理线上公开来源，暂不负责线下走访、家庭相册征集、现场扫描或联系馆藏机构获取非公开资料。

## 2. A 的完整工作流程

```text
调研数据源
  -> 阅读 API 与 robots/使用条款
  -> 确定允许用途
  -> 编写或配置爬虫
  -> 下载 5-10 张试跑
  -> 检查图片、manifest 与 license
  -> 下载正式批次
  -> 自动解码、分辨率与去重
  -> 抽查 license 字段
  -> RIGHTS_OK
  -> 交给 B
```

任何新来源都必须先试跑，不能第一次运行就下载几千张。

## 3. 每批接收和交付

输入：官方开放数据 API、公开馆藏网站和授权信息清楚的线上图片页面。

交付：

```text
data/raw/<provider>/<batch_id>/
├── images/
├── manifest_raw.csv 或 manifest.jsonl
├── README.md
└── rights_or_consent/
```

首批程序化工作目录：

```text
data/work/batches/<batch_id>/01_rights_ok/
├── images/
├── manifest.csv
└── intake_report.json
```

## 4. 开始采集前

### 4.1 建立批次名

```text
YYYYMMDD_来源_序号
例如：20260802_gzlib_001
```

同一天同一来源的下一批依次使用 `002`、`003`。批次建立后不能随意改名。

### 4.2 优先来源

1. 公共领域或 CC0 的官方文化机构开放 API。
2. 授权范围清楚、元数据完整的 CC BY 在线数据。
3. Library of Congress、Smithsonian、Europeana 等官方开放馆藏。
4. Wikimedia Commons 中逐文件授权明确且可追溯的图片。

禁止使用：搜索引擎结果图、Pinterest、来源不明网盘、带商业图库水印的图片、找不到原始页面的图片。

## 5. 爬虫运行手册

### 5.1 现有爬虫位置

```text
scripts/data/download_loc_dataset.py
```

它从 Library of Congress 官方 API 获取达盖尔银版照片，已经具备：

- 官方 API 分页。
- License 白名单过滤。
- 请求间隔和超时控制。
- 指数退避重试。
- 下载后 Pillow 解码检查。
- 最低分辨率过滤。
- SHA-256 完全去重。
- 感知哈希记录。
- 断点续传和 `.part` 临时文件。
- manifest、拒绝记录、统计报告和抽样拼图。

### 5.2 查看参数

```powershell
cd C:\Users\24211\PycharmProjects\PythonProject
.\.venv\Scripts\python.exe -m scripts.data.download_loc_dataset --help
```

常用参数：

| 参数 | 作用 | 建议值 |
|---|---|---:|
| `--collection` | 选择已配置馆藏 | `daguerreotypes` |
| `--limit` | 本次新增合格图片数 | 试跑 5，正式批次 50-100 |
| `--min-short-side` | 最低短边分辨率 | 768 |
| `--query` | 可选搜索关键词 | 不确定时先不填 |
| `--page-size` | 每页 API 结果数 | 100 |
| `--delay` | 请求间隔秒数 | 0.35-1.0 |
| `--retries` | 网络失败重试次数 | 3-5 |
| `--timeout` | 单次请求超时秒数 | 45 |
| `--output-dir` | 自定义输出目录 | 默认即可 |

注意：`--limit 50` 表示在已有数据基础上再增加 50 张，不是把总数限制为 50。下面的 5 张试跑也会写入指定输出目录并成为正式原始记录；不想写入默认目录时，必须用 `--output-dir` 指定独立试验目录。

### 5.3 第一次试跑

```powershell
.\.venv\Scripts\python.exe -m scripts.data.download_loc_dataset `
  --limit 5 `
  --min-short-side 768 `
  --delay 0.5 `
  --output-dir data/raw/pilots/loc_daguerreotypes_pilot
```

试跑完成后必须检查：

1. `images` 中正好新增预期数量的可打开图片。
2. `manifest.jsonl` 每行都包含来源、License、尺寸和哈希。
3. 如果生成了 `rejected.jsonl`，其中的拒绝理由合理；没有拒绝项时该文件可能不存在。
4. `summary.json` 数量和分辨率统计正确。
5. `contact_sheet.jpg` 图片没有明显抓错。

### 5.4 正式运行

```powershell
.\.venv\Scripts\python.exe -m scripts.data.download_loc_dataset `
  --limit 100 `
  --min-short-side 768 `
  --delay 0.5 `
  --retries 5 `
  --timeout 60
```

下载过程中不要同时启动多个实例写入同一个目录，否则可能造成 manifest 竞争。需要并行时必须使用不同输出目录和批次 ID，最后由 D 合并去重。

### 5.5 中断和续传

程序会读取已有 `manifest.jsonl`，跳过已经成功记录的 ID 和 SHA-256。网络中断后使用同一条命令重新运行即可继续追加。

续传前检查：

1. 是否遗留 `.part` 文件。
2. manifest 最后一行是否为完整 JSON。
3. 图片数量是否与 manifest 记录基本一致。
4. 不要手动删除 manifest 后直接重跑，否则会重复下载。

### 5.6 常见失败处理

| 现象 | 原因 | 处理 |
|---|---|---|
| `HTTPError` | 服务器拒绝或暂时异常 | 增加 delay，稍后续传 |
| `URLError` | 网络或 DNS 问题 | 检查网络后原命令续传 |
| `TimeoutError` | 下载超时 | 将 timeout 调到 60-90 |
| `IncompleteRead` | 连接中断 | 脚本会重试，失败记录后再续传 |
| `license_not_accepted` | License 不在白名单 | 保留拒绝记录，不强行下载 |
| `access_restricted` | 资源禁止公开访问 | 跳过，不尝试绕过限制 |
| `short_side_*` | 分辨率不足 | 不降低正式数据标准来凑数量 |
| `exact_duplicate` | 文件内容重复 | 保留拒绝记录，不重复入库 |
| `no_image_url` | API 没有可下载资源 | 跳过并记录 |

### 5.7 爬虫运行后验收

每次运行结束，A 必须提交：

```text
images/
manifest.jsonl
rejected.jsonl（有拒绝记录时）
summary.json
contact_sheet.jpg
README.md（由 A 记录来源、参数和用途）
```

核对终端最后显示的 `accepted`、`inspected` 和 `manifest_total`，其中 `accepted` 应达到本次 `--limit`。如果馆藏耗尽，程序可能以状态码 2 结束并明确提示数量不足。拒绝文件不是垃圾日志，必须保留，用于解释为什么最终数量少于网页结果数。

## 6. 开发新来源爬虫的标准

新增 Smithsonian、Europeana 或 Wikimedia 等线上来源时，不能把网页 HTML 临时解析代码直接复制成正式脚本。优先使用官方 API，并至少实现：

1. 独立的来源配置和输出目录。
2. 明确的 User-Agent，不伪装浏览器攻击站点。
3. 请求限速、超时、重试和错误日志。
4. API 分页，不依赖只抓第一页。
5. 下载前按 License 白名单过滤，下载后保存原始 License 字段。
6. 使用临时文件下载，完整后再原子改名。
7. Pillow 解码和尺寸检查。
8. SHA-256 去重，建议同时保存感知哈希。
9. 可重复运行和断点续传。
10. JSONL 或 CSV manifest。
11. `summary.json` 和抽样拼图。
12. `--limit`、`--output-dir`、`--delay` 等命令行参数。

提交新爬虫前先运行 5 张试验，并让 D 检查 manifest。不得通过绕过登录、验证码、付费墙或访问控制获取数据。

## 7. 每张图片必须记录什么

在 manifest 中填写：

| 字段 | 填写内容 |
|---|---|
| `id` | 当前文件唯一 ID |
| `source_id` | 原始对象 ID，同一原图派生版本共用它 |
| `filename` | 实际文件名 |
| `provider` | 来源机构全称 |
| `source_url` | 图片详情页，不是搜索结果页 |
| `author` | 作者或拍摄者，不详写 `unknown` |
| `license` | Public Domain、CC0、CC BY 等 |
| `rights_url` | 可选存档链接，不参与版权判断 |
| `downloaded_at` | 下载日期 |
| `sha256` | 原始文件哈希 |
| `status` | 核查通过后填写 `RIGHTS_OK` |
| `notes` | 馆藏号、年代、特殊限制等 |

License 为空或不在白名单时状态保持 `NEW`，不能写成 `RIGHTS_OK`。

## 8. License 判断方法

版权判断只读取 manifest 的 `license` 字段。当前白名单：

```text
Public Domain
CC0
CC0 1.0
CC0 1.0 Universal
CC BY 3.0
CC BY 4.0
```

判断规则：

1. License 在白名单中：可以标记 `RIGHTS_OK`。
2. License 为空：保持 `NEW`。
3. License 不在白名单中：保持 `NEW`，交 D 决定是否扩充白名单。
4. `rights_url`、作者年代和网页版权描述只作存档，不参与版权通过判断。
5. 页面存在登录、付费墙或访问限制时仍不得下载，这是访问条件，不是版权判断。

任何人不得自行把相近文字改写成白名单 License。新增 License 必须由 D 统一修改程序和手册。

## 9. 将爬虫结果建立为工作批次

检查以下目录包含 `images`、`manifest.jsonl`、`summary.json`：

```text
data/raw/library_of_congress/daguerreotypes/
```

执行：

```powershell
cd C:\Users\24211\PycharmProjects\PythonProject
.\.venv\Scripts\python.exe -m scripts.data.team_dataset_workflow import-loc `
  --batch-id 20260727_loc_pilot_001 `
  --operator "你的真实姓名"
```

如果提示批次已存在，不要使用 `--force` 覆盖队友成果。先联系 D 确认是否需要新批次。

## 10. 人工抽查

每批至少抽查 10%，首批抽查 10 张：

1. 打开本地图片。
2. 打开对应 `source_url`。
3. 对照标题、馆藏号和图片内容。
4. 检查 manifest 的 `license` 是否在白名单。
5. 检查本地文件哈希是否已记录。
6. 在抽查记录中写下 `source_id`、License 和结论。

## 11. 常见退回原因

| 问题 | 处理方法 |
|---|---|
| 找不到原始页面 | 保持 `NEW`，不交给 B |
| License 为空或不在白名单 | 保持 `NEW`，不进入 B 阶段 |
| 网页与图片不对应 | 修正记录后重新核查 |
| 文件重复下载 | 保留一份，记录重复关系 |
| 页面要求登录或绕过访问控制 | 停止采集，改用公开 API 或其他来源 |

## 12. 交付前检查

- [ ] 图片数和 manifest 行数一致。
- [ ] 每张图有唯一 `source_id`。
- [ ] 每张图有来源页面和 License。
- [ ] SHA-256 非空。
- [ ] 原始图片没有被修改。
- [ ] License 不在白名单的图片没有标为 `RIGHTS_OK`。
- [ ] 已完成至少 10% 人工抽查。
- [ ] 爬虫运行参数和执行时间已记录。
- [ ] rejected、summary 和 contact sheet 已保存。
- [ ] 没有多个进程同时写同一 manifest。
- [ ] 新来源爬虫已先完成 5-10 张试跑。

## 13. 群内交接模板

```text
姓名/角色：张三 / A
批次：20260802_gzlib_001
来源：广州图书馆
采集方式：官方 API 爬虫
爬虫脚本：scripts/data/xxx.py
运行参数：--limit 100 --min-short-side 768 --delay 0.5
候选数量：100
RIGHTS_OK：92
待确认：8
自动拒绝：低分辨率 5、重复 2、下载失败 1
抽查数量：10
License 分布：Public Domain 92
交付目录：data/raw/gzlib/20260802_gzlib_001
下一位：B
```
