# 尘拂光回（PhotoRevival）老照片修复系统

尘拂光回（PhotoRevival）是一个面向老照片划痕检测、内容修复和清晰度增强的本地应用。项目由 React 前端、Flask 后端和 Python 图像修复流水线组成，主要使用 SDXL Inpainting、LoRA 和 SwinIR。

## 主要功能

- 上传 JPG、JPEG、PNG 或 BMP 图片
- 自动检测照片中的划痕和破损区域
- 提供快速修复、精细修复等处理模式
- 异步执行修复任务并显示处理进度
- 查看原图、划痕遮罩、中间结果和最终结果
- 保存最近 100 条任务记录，界面最多返回 50 条
- 提供模型训练、效果评估、诊断和数据可视化脚本

## 修复流程

```text
上传照片
   ↓
划痕/破损检测
   ↓
SwinIR 基础增强
   ↓
SDXL Inpainting + LoRA 内容修复
   ↓
融合并保存最终结果
```

模型在第一次提交修复任务时加载，因此第一次处理通常会等待更久。

## 项目结构

```text
PhotoRevival/
├── src/photo_revival/           # 后端核心 Python 包
│   ├── api/server.py            # Flask 服务及接口
│   ├── two_stage_restoration.py # 两阶段修复主流程
│   ├── hybrid_restoration_pipeline.py
│   ├── network_swinir.py        # SwinIR 网络定义
│   ├── *_detector.py            # 划痕和遮罩检测模块
│   └── paths.py                 # 项目路径配置
├── scripts/
│   ├── training/                # 模型训练脚本
│   ├── evaluation/              # 测试、评估和方法对比
│   ├── diagnostics/             # 模型和流水线诊断
│   └── visualization/           # 数据生成与可视化
├── frontend/                    # React + TypeScript + Vite 前端
├── docs/                        # 项目详细文档
├── data/                        # 数据集、纹理素材
├── models/                      # 基础模型、权重和训练检查点
│   ├── base/
│   ├── weights/
│   ├── classifiers/
│   └── checkpoints/
├── artifacts/                   # 缓存、评估输出、日志和 API 文件
├── third_party/                 # DiffBIR、BOPBTL 等外部项目
├── pyproject.toml               # Python 项目配置
└── requirements.txt             # Python 依赖列表
```

所有路径在 `src/photo_revival/paths.py` 中统一定义。`third_party/` 中的代码来自第三方或参考项目，当前保持原始目录结构。

## 环境要求

建议开发环境如下：

- Windows 10/11
- Python 3.10、3.11 或当前已验证的 Python 3.13 环境
- Node.js 18 或更高版本
- NVIDIA 显卡及匹配的 CUDA 环境
- 充足的显存和磁盘空间；SDXL 模型对显存要求较高

项目当前的 `.venv` 已验证可在 Python 3.13.7、PyTorch 2.6.0 和 CUDA 12.4 下导入全部核心依赖。Python 3.10/3.11 的第三方兼容性通常更稳妥；暂不支持 Python 3.14。仅查看前端可以不安装 CUDA，实际执行完整修复建议使用支持 CUDA 的显卡。

## 模型文件

后端默认按以下顺序查找模型：

```text
models/base/AI-ModelScope/stable-diffusion-xl-1___0-inpainting-0___1/

models/checkpoints/lora_inpainting_v29/epoch5/
models/checkpoints/lora_inpainting_v28/best/
models/checkpoints/lora_inpainting_v27/best/
models/checkpoints/lora_inpainting_v26/final/
models/checkpoints/lora_inpainting_v25/best/

models/weights/realesrgan_s4_swinir_100k.pth
models/checkpoints/swinir/best_model.pth
```

LoRA 和 SwinIR 会采用列表中第一个存在的路径。SDXL 基础模型路径是必需的；缺少 LoRA 或 SwinIR 权重时，具体行为取决于修复流水线的加载逻辑。

## 安装后端

在项目根目录打开 PowerShell：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

如果使用 NVIDIA 显卡，建议先根据本机 CUDA 版本安装对应的 PyTorch，再安装项目：

```powershell
pip install -e .
```

`-e` 表示以开发模式安装。修改 `src/photo_revival/` 内的代码后，不需要重新安装项目。

## 安装前端

```powershell
cd frontend
npm.cmd install
cd ..
```

## 开发模式运行

先在第一个终端启动后端：

```powershell
.\.venv\Scripts\activate
python -m photo_revival.api.server
```

后端地址为 `http://localhost:5000`。看到服务启动信息后，在第二个终端启动前端：

```powershell
cd frontend
npm.cmd run dev
```

浏览器访问 `http://localhost:5173`。开发服务器会把 `/api` 请求自动转发到 `http://localhost:5000`。

可通过以下地址检查后端状态：

```text
http://localhost:5000/api/status
```

## 生产构建运行

先构建前端：

```powershell
cd frontend
npm.cmd run build
cd ..
```

然后启动后端：

```powershell
.\.venv\Scripts\activate
python -m photo_revival.api.server
```

构建完成后，Flask 会直接提供 `frontend/dist/` 中的页面，可访问 `http://localhost:5000` 使用完整应用。

## 使用流程

1. 打开网页并上传照片。
2. 根据需要运行划痕检测，查看遮罩覆盖率。
3. 选择修复模式和强度后提交任务。
4. 等待任务状态变为 `completed`。
5. 查看或下载最终修复图、中间增强图及遮罩图。

上传文件保存在 `artifacts/api/uploads/`，结果保存在 `artifacts/api/results/`，历史记录保存在 `artifacts/api/history.json`。

## 常用脚本

所有脚本建议从项目根目录以模块方式运行：

```powershell
# 测试单张图片或目录
python -m scripts.evaluation.test_two_stage --mode single --input .\example.jpg

# 快速运行混合修复流水线
python -m scripts.diagnostics.run_hybrid_restoration --input .\example.jpg --output .\artifacts\evaluations\result.jpg

# 训练 SwinIR
python -m scripts.training.train_swinir

# 训练统一 SDXL Inpainting LoRA（SwinIR + BOPBTL 遮罩）
python -m scripts.training.train_lora

# 只准备 SwinIR 与遮罩缓存，不开始训练
python -m scripts.training.train_lora --prepare-only

# 不使用 SwinIR，并使用全图遮罩
python -m scripts.training.train_lora --preprocessor none --mask-mode full

# 查看某个脚本的参数
python -m scripts.training.train_lora --help
```

统一 LoRA 入口支持 JSON 配置、断点继续、EMA、8-bit Adam、Min-SNR、LPIPS 和 SSIM。LPIPS/SSIM 会增加显存消耗，默认关闭，可通过 `--lpips-weight` 和 `--ssim-weight` 启用。SwinIR 和 ControlNet 训练保留为独立入口，旧 LoRA 实验位于 `scripts/training/legacy/`。

训练脚本通常需要单独准备数据集、基础模型和较大的显存。详细说明见 [训练指南](docs/training.md)。

## 接口说明

完整接口、请求参数和响应示例见 [API 接口文档](docs/api.md)。代码目录的职责划分见 [项目结构说明](docs/project_structure.md)。

## 常见问题

### `No module named photo_revival`

确认虚拟环境已激活，并且在项目根目录执行过：

```powershell
pip install -e .
```

### 后端启动成功，但第一次修复很慢

这是模型首次载入造成的正常现象。后续请求会复用已经加载的模型。

### 显存不足或出现 CUDA out of memory

关闭占用显存的程序，降低同时处理的任务数量，并确认模型使用了适合显卡的精度配置。完整 SDXL 流水线不适合低显存环境。

### 前端能打开，但接口请求失败

确认后端正在 `5000` 端口运行，前端开发服务器正在 `5173` 端口运行。后端开发模式只允许来自 `http://localhost:5173` 的跨域请求。

### 页面只显示 API 提示信息

说明 `frontend/dist/` 尚未生成。执行：

```powershell
cd frontend
npm.cmd run build
```

### 历史记录存在，但原图或结果图打不开

历史信息和图片文件分开保存。不要只保留 `artifacts/api/history.json` 而删除同目录下 `uploads/` 或 `results/` 中对应的文件。

## 开发检查

```powershell
# Python 语法检查
python -m compileall -q src scripts

# 前端类型检查
cd frontend
npm.cmd run check

# 前端代码检查
npm.cmd run lint

# 前端生产构建
npm.cmd run build
```

## 数据与安全说明

- 项目默认在本地运行，上传照片和输出结果会保存在本机项目目录。
- API 当前没有用户认证，不应直接暴露到公网。
- `artifacts/api/history.json` 不是数据库，只适合本地开发和单机使用。
- 删除历史记录接口只删除历史条目，不会自动删除对应图片文件。
- 模型权重、数据集、缓存和输出目录通常体积较大，不建议提交到 Git。
