import os
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLInpaintPipeline

# ==================== 解决网络问题（国内用户可取消注释） ====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==================== 配置区域 ====================
# 模型路径：Hugging Face ID 或本地路径
MODEL_ID = "./local_models/sdxl-inpainting"  # 本地模型路径

# 输入输出文件夹
INPUT_DIR = "old_photos"       # 输入老照片目录
OUTPUT_DIR = "restored_photos"  # 输出修复后的照片目录

# 硬件配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 根据设备选择数据类型：GPU 用 float16，CPU 用 float32
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 修复参数
SEED = 42
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
STRENGTH = 0.7  # 修复强度，值越大修复效果越强但可能改变原始特征

# ==================== 加载模型 ====================
print("正在加载模型...")
try:
    # 加载模型（根据设备自动选择合适的数据类型）
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        variant="fp16" if DTYPE == torch.float16 else None,
        use_safetensors=True
    )
    pipe = pipe.to(DEVICE)
    # 仅在 GPU 且显存不足时启用 CPU offload（需要安装 accelerate）
    if DEVICE == "cuda":
        try:
            pipe.enable_model_cpu_offload()
            print("已启用 CPU offload（节省显存）")
        except ImportError:
            print("未安装 accelerate，跳过 CPU offload。如需节省显存请安装：pip install accelerate")
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败：{e}")
    print("请检查网络连接或模型路径")
    exit(1)

# ==================== 辅助函数：自动生成掩码 ====================
def generate_automatic_mask(image_path, sensitivity=0.3):
    """自动生成掩码，检测照片中的损坏区域"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测边缘和损坏区域
    edges = cv2.Canny(gray, 50, 150)
    
    # 检测噪点和霉斑（通过阈值处理）
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 合并边缘和阈值结果
    mask = cv2.bitwise_or(edges, threshold)
    
    # 膨胀操作，扩大掩码区域
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 高斯模糊，使掩码边缘更自然
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # 根据敏感度调整掩码
    mask = cv2.threshold(mask, int(255 * sensitivity), 255, cv2.THRESH_BINARY)[1]
    
    return Image.fromarray(mask)

# ==================== 单张图片修复 ====================
def restore_one_image(input_path, output_path, mask=None, seed=None):
    """修复单张老照片"""
    init_image = Image.open(input_path).convert("RGB")
    
    # 如果没有提供掩码，自动生成
    if mask is None:
        mask = generate_automatic_mask(input_path)
    
    # 调整掩码大小以匹配输入图片
    if mask.size != init_image.size:
        mask = mask.resize(init_image.size, Image.Resampling.NEAREST)

    # 修复提示词
    prompt = (
        "restored old photo, clear, sharp, vibrant colors, no scratches, no mold, "
        "no watermarks, high quality, realistic, detailed, professional restoration"
    )
    
    # 负面提示词
    negative_prompt = (
        "blurry, low quality, distorted, ugly, scratches, mold, watermarks, "
        "text, signature, frame, modern, digital artifacts"
    )

    # 设置生成器
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # 执行修复
    print(f"正在修复 {os.path.basename(input_path)}...")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask,
        strength=STRENGTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    # 保存结果
    result.save(output_path)
    print(f"修复完成！结果已保存为: {output_path}")

# ==================== 批量修复 ====================
def batch_restore(input_folder, output_folder):
    """批量修复老照片"""
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图片文件
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    
    if not files:
        print("未找到图片文件")
        return

    print(f"找到 {len(files)} 张图片，开始修复...")
    
    # 批量处理
    for idx, filename in enumerate(tqdm(files)):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # 确定种子
            if SEED is not None:
                cur_seed = SEED + idx
            else:
                cur_seed = None
            
            # 修复图片
            restore_one_image(input_path, output_path, seed=cur_seed)
        except Exception as e:
            print(f"修复 {filename} 时出错: {e}")

    print(f"批量修复完成！结果保存在 {output_folder}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 执行批量修复
    batch_restore(
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR
    )
