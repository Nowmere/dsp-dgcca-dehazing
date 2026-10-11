import os
import glob
import torch
import gc
import numpy as np
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from tqdm import tqdm

# ================= 🛡️ 终极防爆装甲 =================
# 免疫 RESIDE 数据集里的 61 bytes 截断毒图
ImageFile.LOAD_TRUNCATED_IMAGES = True 
# =================================================

# ================= ⚙️ 路径配置区 =================
# ⚠️ 请根据您当前的测试集或训练集修改此路径
HAZY_DIR = "./data/RESIDE/SOTS/outdoor/hazy" 
SAVE_DIR = "./data/RESIDE/SOTS/outdoor/depth_dav2" 
MAX_SIZE = 1024 # 限制最大分辨率防爆显存
# =================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("======================================================")
    print("🚀 启动 Depth Anything V2 (DAV2) 深度先验提炼引擎")
    print("======================================================")

    # 1. 加载 DAV2 模型 (默认使用 Small 版本，速度快且精度足够做去雾先验)
    # 如果您显存极大且追求极致，可以将 Small 换成 Base 或 Large
    model_repo = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"⏳ 正在从 HuggingFace 召唤 {model_repo} ...")
    
    processor = AutoImageProcessor.from_pretrained(model_repo)
    model = AutoModelForDepthEstimation.from_pretrained(model_repo).to(device).eval()

    # 2. 扫描数据集
    img_paths = glob.glob(os.path.join(HAZY_DIR, "*.*"))
    img_paths = [p for p in img_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"📦 共锁定 {len(img_paths)} 张雾天图像，准备提炼物理几何信息...\n")

    # 3. 开启无梯度推理模式
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(img_paths, desc="DAV2 提炼进度")):
            filename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(filename)[0]
            save_path = os.path.join(SAVE_DIR, f"{name_no_ext}.npy")
            
            # ⏭️ 核心机制：断点续传。如果 .npy 已存在且大小正常，直接跳过
            if os.path.exists(save_path) and os.path.getsize(save_path) > 128:
                continue
                
            try:
                # 安全读取图像
                img = Image.open(img_path).convert('RGB')
                
                # 动态缩放限制，防止高分图直接打穿显存
                w, h = img.size
                if max(w, h) > MAX_SIZE:
                    r = MAX_SIZE / float(max(w, h))
                    img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
                
                # 图像预处理与模型推理
                inputs = processor(images=img, return_tensors="pt").to(device)
                outputs = model(**inputs)
                
                # 获取预测的深度张量
                depth = outputs.predicted_depth
                
                # DAV2 的输出尺寸可能与原图不一致，需要严格插值对齐回原图分辨率！
                depth_resized = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=(h, w), # 恢复到真实的原始 W, H
                    mode="bicubic",
                    align_corners=False
                ).squeeze()
                
                # 归一化到 0-1 范围 (去雾先验标准操作)
                d_min, d_max = depth_resized.min(), depth_resized.max()
                depth_norm = (depth_resized - d_min) / (d_max - d_min + 1e-8)
                
                # 转换为 Float16 极大地节省硬盘空间，且精度损失对去雾可忽略不计
                depth_npy = depth_norm.cpu().numpy().astype(np.float16)
                np.save(save_path, depth_npy)
                
            except Exception as e:
                # 遇到彻底损坏的图不崩溃，直接跳过并在终端报警
                print(f"\n⚠️ [毒图警告] 彻底无法处理 {filename}，已自动跳过！原因: {e}")

            # 🧹 内存扫地机器人：每 100 张强制清理一次底层显存垃圾
            if (i + 1) % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    print("\n======================================================")
    print("✅ 全量 DAV2 深度图库构建完成！")
    print("您的多模态网络现在拥有了世界上最顶级的物理几何感知能力！")
    print("======================================================")

if __name__ == "__main__":
    main()