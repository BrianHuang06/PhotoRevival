# 训练指南

## 训练入口

项目目前保留三个训练入口：

| 入口 | 用途 |
| --- | --- |
| `scripts.training.train_lora` | 推荐入口，训练 SDXL Inpainting LoRA |
| `scripts.training.train_swinir` | 单独训练 SwinIR 增强模型 |
| `scripts.training.train_controlnet_lora` | ControlNet 实验路线 |

`scripts/training/legacy/` 保存 v25-v28 的旧 LoRA 实验，仅用于回溯，不建议用于新训练。

## 统一 LoRA 流程

```text
退化图
  ↓
SwinIR 预处理（可关闭）
  ↓
全图遮罩或 BOPBTL 损伤遮罩
  ↓
SDXL Inpainting + LoRA
  ↓
评估、检查点、best 和 final 权重
```

训练器使用与 Diffusers 推理一致的 9 通道顺序：噪声 latent、遮罩、被遮挡图像 latent。

## 常用命令

默认使用 SwinIR 和 BOPBTL 遮罩：

```powershell
python -m scripts.training.train_lora
```

先生成缓存：

```powershell
python -m scripts.training.train_lora --prepare-only
```

使用退化原图和全图遮罩：

```powershell
python -m scripts.training.train_lora --preprocessor none --mask-mode full
```

启用图像空间损失：

```powershell
python -m scripts.training.train_lora --lpips-weight 0.1 --ssim-weight 0.05
```

LPIPS 和 SSIM 需要通过 VAE 解码预测结果，会明显增加显存使用。建议先用默认的 MSE + Min-SNR 配置确认训练稳定，再逐步启用。

关闭或开启布尔选项：

```powershell
python -m scripts.training.train_lora --no-ema --8bit-adam
python -m scripts.training.train_lora --no-gradient-checkpointing
```

继续已有检查点：

```powershell
python -m scripts.training.train_lora --resume-from .\models\checkpoints\lora_inpainting\epoch10 --start-epoch 10
```

## 数据目录

默认使用文件名一一对应的配对图片：

```text
data/datasets/archive/03_Synthetic_Dataset/
├── Train_GT_Clean/          # 干净目标图
└── Train_Input_Degraded/    # 退化输入图
```

只有两个目录中同名的 JPG、JPEG 或 PNG 文件会组成训练对。也可以通过 `--clean-dir` 和 `--degraded-dir` 指定其他目录。

## 输出内容

默认写入 `models/checkpoints/lora_inpainting/`：

```text
models/checkpoints/lora_inpainting/
├── config.json
├── loss_log.json
├── epoch5/
├── best/
└── final/
```

`config.json` 会记录本次训练的完整配置，`loss_log.json` 记录每轮有效步数和平均损失。

## 配置文件

可以将输出目录中的 `config.json` 作为下一次训练的配置：

```powershell
python -m scripts.training.train_lora --config .\models\checkpoints\lora_inpainting\config.json
```

命令行参数会覆盖 JSON 中的同名配置。

## 重要说明

- 当前统一 LoRA 训练要求 CUDA。
- BOPBTL 模式需要第三方项目及其检测权重保持在现有目录。
- SwinIR 模式默认读取 `models/weights/realesrgan_s4_swinir_100k.pth`。
- 全图遮罩符合标准 inpainting 语义，被遮挡条件图会变为全零；更适合整图重绘实验，不适合强调局部保真的任务。
- 新训练应使用统一入口；旧权重仍可由运行时加载，不受脚本归档影响。
