import os
import glob
import torch
import gc  # 🌟 新增：强制垃圾回收
import numpy as np
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from tqdm import tqdm

# 🌟 新增：允许 PIL 加载损坏/截断的图片，防止卡死
ImageFile.LOAD_TRUNCATED_IMAGES = True 

# ================= 配置区 =================
HAZY_DIR = "./data/RESIDE/OTS/hazy"  # ⚠️ 请确认你的真实路径
SAVE_DIR = "./data/RESIDE/OTS/depth_dav2" 
MAX_SIZE = 1024 
# =========================================

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("⏳ 正在加载 DepthAnything V2 (Small)...")
processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()

img_paths = glob.glob(os.path.join(HAZY_DIR, "*.*"))
img_paths = [p for p in img_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"📦 找到 {len(img_paths)} 张雾图，开始提炼...")

with torch.no_grad():
    for i, img_path in enumerate(tqdm(img_paths, desc="Processing")):
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        save_path = os.path.join(SAVE_DIR, f"{name_no_ext}.npy")
        
        # 断点续传：存在则跳过
        if os.path.exists(save_path):
            continue
            
        try:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            if max(w, h) > MAX_SIZE:
                r = MAX_SIZE / float(max(w, h))
                img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
            
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
            depth = outputs.predicted_depth
            
            depth_resized = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(img.size[1], img.size[0]),
                mode="bicubic",
                align_corners=False
            ).squeeze()
            
            d_min, d_max = depth_resized.min(), depth_resized.max()
            depth_norm = (depth_resized - d_min) / (d_max - d_min + 1e-8)
            
            np.save(save_path, depth_norm.cpu().numpy().astype(np.float16))
            
        except Exception as e:
            # 🌟 把报错的图片名字大声喊出来，方便排查
            print(f"\n❌ [毒图警告] 处理 {filename} 失败，跳过！原因: {e}")

        # 🌟🌟🌟 核心防爆机制：每处理 50 张图，强制打扫一次内存和显存垃圾！
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

print("\n✅ 全量深度图库构建完成！")