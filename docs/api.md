# 尘拂光回（PhotoRevival）API 接口文档

## 基本信息

- 默认服务地址：`http://localhost:5000`
- 接口前缀：`/api`
- 数据格式：除文件上传外，均使用 JSON
- 任务执行方式：异步

一次完整修复通常按以下顺序调用：

```text
POST /api/upload
       ↓
POST /api/detect（可选）
       ↓
POST /api/restore
       ↓
GET /api/task/{task_id}（轮询）
       ↓
GET /api/images/{filename}
```

## 服务状态

### `GET /api/status`

检查后端、CUDA 和当前任务状态。

```json
{
  "status": "ok",
  "pipeline_loaded": false,
  "active_tasks": 0,
  "cuda_available": true
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | string | 服务状态 |
| `pipeline_loaded` | boolean | 修复模型是否已加载 |
| `active_tasks` | number | 正在检测或修复的任务数量 |
| `cuda_available` | boolean | PyTorch 是否检测到 CUDA |

## 上传图片

### `POST /api/upload`

使用 `multipart/form-data` 上传图片，文件字段名必须为 `file`。支持 `.jpg`、`.jpeg`、`.png`、`.bmp`。

```powershell
curl.exe -X POST -F "file=@example.jpg" http://localhost:5000/api/upload
```

响应示例：

```json
{
  "id": "task_1710000000_a1b2c3",
  "filename": "example.jpg",
  "original_url": "/api/images/task_1710000000_a1b2c3.jpg",
  "mode": "quick",
  "status": "idle",
  "progress": 0,
  "created_at": "2026-07-19T10:00:00",
  "width": 1024,
  "height": 768,
  "file_size": 245760,
  "params": {
    "mode": "quick",
    "strength": 0.35,
    "guidanceScale": 7.5,
    "steps": 25,
    "seed": 42,
    "blendFactor": 0.8
  }
}
```

## 检测划痕

### `POST /api/detect`

```json
{
  "task_id": "task_1710000000_a1b2c3",
  "sensitivity": 0.5
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `task_id` | string | 是 | - | 上传接口返回的任务编号 |
| `sensitivity` | number | 否 | `0.5` | 传统检测器使用的灵敏度 |

如果根目录存在 `scratch_classifier_v2.pkl`，后端会优先使用学习型检测器；否则使用内置的传统划痕检测器。

```json
{
  "mask_url": "/api/images/task_1710000000_a1b2c3_detect_mask.png",
  "coverage": 3.27
}
```

`coverage` 表示遮罩覆盖图片的百分比。

## 提交修复任务

### `POST /api/restore`

```json
{
  "task_id": "task_1710000000_a1b2c3",
  "mode": "fine",
  "strength": 0.35,
  "guidanceScale": 7.5,
  "steps": 25,
  "seed": 42,
  "blendFactor": 0.8
}
```

| 参数 | 类型 | 必填 | 默认值 | 当前作用 |
| --- | --- | --- | --- | --- |
| `task_id` | string | 是 | - | 要处理的任务编号 |
| `mode` | string | 否 | `quick` | 决定两阶段修复强度 |
| `strength` | number | 否 | `0.35` | `fine` 模式下参与强度计算 |
| `seed` | number | 否 | `42` | 控制生成结果的随机种子 |
| `guidanceScale` | number | 否 | `7.5` | 当前保存该字段，但尚未传入流水线 |
| `steps` | number | 否 | `25` | 当前保存该字段，但尚未传入流水线 |
| `blendFactor` | number | 否 | `0.8` | 当前保存该字段，但尚未传入流水线 |

模式对应的实际强度：

| 模式 | 第一阶段强度 | 第二阶段强度 |
| --- | --- | --- |
| `quick` | `0.6` | `0.1` |
| `fine` | `strength + 0.2`，限制在 `0.3` 至 `1.0` | `strength` |
| 其他值 | `0.5` | `0.05` |

```json
{
  "task_id": "task_1710000000_a1b2c3",
  "status": "processing"
}
```

接口返回后，修复会在后台线程执行，客户端需要继续轮询任务状态。

## 查询任务

### `GET /api/task/{task_id}`

```powershell
curl.exe http://localhost:5000/api/task/task_1710000000_a1b2c3
```

完成后响应会包含结果地址：

```json
{
  "id": "task_1710000000_a1b2c3",
  "status": "completed",
  "progress": 100,
  "result_url": "/api/images/task_1710000000_a1b2c3_restored.jpg",
  "mask_url": "/api/images/task_1710000000_a1b2c3_mask.png",
  "stage1_url": "/api/images/task_1710000000_a1b2c3_stage1.jpg",
  "completed_at": "2026-07-19T10:05:00"
}
```

| 状态 | 说明 |
| --- | --- |
| `idle` | 已上传，尚未开始 |
| `pending` | 已提交，等待后台执行 |
| `detecting` | 正在准备或检测 |
| `processing` | 正在执行修复 |
| `completed` | 处理完成 |
| `error` | 处理失败，可读取 `error` 字段 |

## 历史记录

### `GET /api/history`

返回最多 50 条历史记录：

```json
{
  "tasks": []
}
```

### `DELETE /api/history/{task_id}`

```powershell
curl.exe -X DELETE http://localhost:5000/api/history/task_1710000000_a1b2c3
```

```json
{
  "success": true
}
```

此操作只删除内存和 `artifacts/api/history.json` 中的记录，不会删除对应图片。

## 获取图片

### `GET /api/images/{filename}`

用于读取上传图片、检测遮罩和修复结果。文件名由其他接口返回，不建议客户端自行拼接。

## 错误响应

```json
{
  "error": "Task not found"
}
```

| 状态码 | 说明 |
| --- | --- |
| `400` | 缺少文件、文件类型不支持或请求参数有误 |
| `404` | 任务或图片不存在 |
| `409` | 同一任务已在处理中 |
| `500` | 模型加载、检测或修复过程发生异常 |

## 当前限制

- 服务没有登录、权限控制和请求限流，仅适合本地或受信任网络。
- 任务状态主要保存在进程内存中，重启后会从历史文件恢复为 `idle`。
- 后台任务使用线程执行，没有独立任务队列，不适合高并发部署。
- 历史记录使用单个 JSON 文件保存，并发写入能力有限。
- `guidanceScale`、`steps` 和 `blendFactor` 是预留参数，当前尚未影响实际流水线。
