# 项目结构说明

PhotoRevive 使用 `src` 布局，代码、数据、模型、第三方依赖和运行产物彼此分离。

```text
PhotoRevive/
├── src/photo_revival/       # 可复用 Python 包
├── scripts/                 # 训练、评估、诊断和数据脚本
├── frontend/                # React/Vite 前端
├── docs/                    # 文档和项目报告
├── data/                    # 数据集与纹理素材
├── models/                  # 模型文件
├── artifacts/               # 可重新生成或运行时产生的文件
├── third_party/             # 外部项目源码
├── pyproject.toml
└── README.md
```

## 核心代码

- `src/photo_revival/api/server.py`：Flask API 服务。
- `src/photo_revival/two_stage_restoration.py`：主要两阶段修复流水线。
- `src/photo_revival/hybrid_restoration_pipeline.py`：混合修复实验流水线。
- `src/photo_revival/training/`：统一 LoRA 配置、预处理、损失与训练循环。
- `src/photo_revival/paths.py`：所有顶层资源目录的唯一配置位置。

## 脚本

- `scripts/training/train_lora.py`：推荐的 SDXL Inpainting LoRA 训练入口。
- `scripts/training/train_swinir.py`：SwinIR 训练入口。
- `scripts/training/train_controlnet_lora.py`：ControlNet 实验入口。
- `scripts/training/legacy/`：历史 LoRA 实验，只用于回溯。
- `scripts/evaluation/`：效果评估和方法比较。
- `scripts/diagnostics/`：模型、权重和流水线诊断。
- `scripts/visualization/`：数据生成及结果可视化。

## 数据和模型

- `data/datasets/`：训练、测试和采集数据。
- `data/textures/`：合成老化所需纹理素材。
- `models/base/`：SDXL 等基础模型。
- `models/weights/`：SwinIR 等独立权重。
- `models/classifiers/`：划痕分类器。
- `models/checkpoints/`：LoRA、SwinIR 等训练检查点。

## 运行产物

- `artifacts/api/`：上传图片、修复结果和 API 历史记录。
- `artifacts/cache/`：SwinIR 与 BOPBTL 预处理缓存。
- `artifacts/evaluations/`：测试图片、比较图和指标输出。
- `artifacts/logs/`：历史训练日志。
- `artifacts/figures/`：项目报告使用的图表。

## 第三方源码

- `third_party/DiffBIR/`
- `third_party/Bringing-Old-Photos-Back-to-Life/`

第三方目录不属于 `photo_revival` 包，只由明确需要它们的训练或诊断模块加载。
