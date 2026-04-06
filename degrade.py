import cv2
import numpy as np
import random
import os
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# ==================== 配置 ====================
INPUT_DIR = "01_Clean_Candidates_GT"
OUTPUT_DIR = "aged_photos_traditional"
SEVERITY = "medium"   # low, medium, high

# ==================== 老化效果函数（不改变原图内容）====================
def add_fading(img, intensity=0.5):
    """降低饱和度 + 暖色调，不改变结构"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * (1 - intensity * 0.5)
    hsv[:, :, 0] = hsv[:, :, 0] + int(15 * intensity)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def add_grain(img, intensity=0.02):
    """添加胶片颗粒"""
    noise = np.random.randn(*img.shape) * intensity * 255
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def add_scratches(img, num_scratches=20):
    """绘制划痕（白色或黑色线条）"""
    h, w = img.shape[:2]
    result = img.copy()
    for _ in range(num_scratches):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        angle = random.uniform(0, 2*np.pi)
        length = random.randint(int(0.02*min(h,w)), int(0.1*min(h,w)))
        x2 = int(x1 + length*np.cos(angle))
        y2 = int(y1 + length*np.sin(angle))
        color = (random.randint(200,255), random.randint(200,255), random.randint(200,255)) if random.random()>0.5 else (0,0,0)
        thickness = random.randint(1, 2)
        cv2.line(result, (x1,y1), (x2,y2), color, thickness)
    return result

def add_mold_spots(img, num_spots=30):
    """添加霉斑（半透明黄褐色圆形）"""
    h, w = img.shape[:2]
    result = img.astype(np.float32)
    for _ in range(num_spots):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        radius = random.randint(5, 40)
        color = (random.randint(60,120), random.randint(40,80), random.randint(20,60))
        # 绘制半透明圆
        overlay = result.copy()
        cv2.circle(overlay, (cx, cy), radius, color, -1)
        alpha = random.uniform(0.2, 0.6)
        result = cv2.addWeighted(overlay, alpha, result, 1-alpha, 0)
    return np.clip(result, 0, 255).astype(np.uint8)

def add_tear_edges(img, num_tears=2):
    """模拟边缘破损（填充纸张色）"""
    h, w = img.shape[:2]
    result = img.copy()
    paper_color = (210, 180, 140)
    for _ in range(num_tears):
        edge = random.choice(['top','bottom','left','right'])
        if edge == 'top':
            y1, y2 = 0, random.randint(10, 60)
            x1, x2 = random.randint(0, w//3), random.randint(2*w//3, w)
            pts = np.array([[x1,0], [x2,0], [x2,y2], [x1,y2]], np.int32)
        elif edge == 'bottom':
            y1, y2 = h-random.randint(10,60), h
            x1, x2 = random.randint(0, w//3), random.randint(2*w//3, w)
            pts = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], np.int32)
        elif edge == 'left':
            x1, x2 = 0, random.randint(10, 60)
            y1, y2 = random.randint(0, h//3), random.randint(2*h//3, h)
            pts = np.array([[0,y1], [x2,y1], [x2,y2], [0,y2]], np.int32)
        else:
            x1, x2 = w-random.randint(10,60), w
            y1, y2 = random.randint(0, h//3), random.randint(2*h//3, h)
            pts = np.array([[x1,y1], [w,y1], [w,y2], [x1,y2]], np.int32)
        cv2.fillPoly(result, [pts], paper_color)
        cv2.polylines(result, [pts], True, (100,80,60), 1)
    return result

def add_creasing(img, num_creases=2):
    """折痕（暗线或亮线）"""
    h, w = img.shape[:2]
    result = img.copy()
    for _ in range(num_creases):
        x1, y1 = random.randint(0,w), random.randint(0,h)
        x2, y2 = random.randint(0,w), random.randint(0,h)
        color = (220,220,220) if random.random()>0.5 else (50,50,50)
        cv2.line(result, (x1,y1), (x2,y2), color, 2)
        result = cv2.GaussianBlur(result, (3,3), 0)
    return result

def degrade_image_preserve_content(img, severity='medium'):
    """主函数：所有操作都不改变原图结构，仅叠加纹理"""
    if severity == 'low':
        fade=0.3; grain=0.01; scratches=10; spots=15; tears=1; creases=1
    elif severity == 'medium':
        fade=0.6; grain=0.02; scratches=20; spots=30; tears=2; creases=2
    else:
        fade=0.9; grain=0.04; scratches=40; spots=50; tears=3; creases=3

    img = add_fading(img, fade)
    img = add_grain(img, grain)
    img = add_scratches(img, scratches)
    img = add_mold_spots(img, spots)
    img = add_tear_edges(img, tears)
    img = add_creasing(img, creases)
    # 轻微降低对比度
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(0.9)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==================== 批量处理 ====================
def batch_degrade(input_folder, output_folder, severity='medium'):
    os.makedirs(output_folder, exist_ok=True)
    exts = ('.jpg','.jpeg','.png','.bmp')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(exts)]
    for fname in tqdm(files):
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)
        img = cv2.imread(in_path)
        if img is None:
            continue
        aged = degrade_image_preserve_content(img, severity)
        cv2.imwrite(out_path, aged)
    print(f"完成！结果保存在 {output_folder}")

if __name__ == "__main__":
    batch_degrade(INPUT_DIR, OUTPUT_DIR, SEVERITY)