# sdxl_inpaint_detailed.py

import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import os

# ==================== 配置区域 ====================
# 在这里修改你的设置

# 设备设置：检查是否有可用的GPU，如果没有则使用CPU（会很慢）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {device}")

# 模型路径
model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

# 输入输出文件路径
input_image_path = "./input.jpg"  # 请替换为你的原始图片路径
mask_image_path = "./mask.png"  # 请替换为你的掩码图片路径
output_image_path = "./output.jpg"  # 输出图片的保存路径

# 生成参数
prompt = "a cute cat, detailed, high quality"  # 描述你希望在mask区域生成什么
negative_prompt = "blurry, low quality, distorted, ugly"  # 描述你不希望出现的内容
num_inference_steps = 30  # 生成步数，越多细节越好但越慢 (20-50)
strength = 0.99  # 修复强度，对于inpainting建议接近1.0
guidance_scale = 7.5  # 提示词相关性 (7-12之间效果较好)


# ==================== 配置结束 ====================

def main():
    # 检查输入文件是否存在
    if not os.path.exists(input_image_path):
        print(f"错误：找不到输入图片 '{input_image_path}'")
        return
    if not os.path.exists(mask_image_path):
        print(f"错误：找不到掩码图片 '{mask_image_path}'")
        return

    print("正在加载模型...（首次运行需要下载，请耐心等待）")

    # 1. 加载修复管道
    # torch_dtype=torch.float16 使用半精度，能大幅节省显存且质量损失很小
    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    # 将管道移动到GPU或CPU
    pipe = pipe.to(device)

    # 2. 启用内存优化 (如果安装了xformers)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("已启用 xformers 内存优化。")
    except Exception as e:
        print(f"无法启用 xformers: {e}。将继续使用默认注意力机制。")

    print("模型加载完成！")

    # 3. 加载并准备输入图像和掩码
    print("正在加载输入图像和掩码...")
    init_image = load_image(input_image_path).convert("RGB")
    mask_image = load_image(mask_image_path).convert("RGB")

    # 可选：调整图像大小以适应SDXL模型
    # SDXL在多种分辨率上训练，但1024x1024是标准尺寸之一
    # 如果你的图片很大，可以取消下面两行的注释来调整大小以加快速度和节省显存
    # init_image = init_image.resize((1024, 1024))
    # mask_image = mask_image.resize((1024, 1024))

    # 4. 开始生成！
    print("开始生成修复后的图像...（这可能需要一些时间）")
    with torch.inference_mode():  # 减少内存占用
        generated_image = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        ).images[0]  # .images[0] 获取生成的第一张图片

    # 5. 保存结果
    generated_image.save(output_image_path)
    print(f"生成完成！结果已保存为: {output_image_path}")

    # 6. 可选：创建一个对比图（原始、掩码、生成结果）
    try:
        # 将掩码转换为单通道并调整大小以匹配原图（为了显示）
        mask_for_display = mask_image.resize(init_image.size).convert("L")
        # 创建一个对比网格
        grid = make_image_grid([init_image, mask_for_display, generated_image], rows=1, cols=3)
        grid.save("./comparison_grid.png")
        print("对比图已保存为: ./comparison_grid.png")
    except Exception as e:
        print(f"无法创建对比图: {e}")


if __name__ == "__main__":
    main()