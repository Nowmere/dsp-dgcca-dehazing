"""
Generate DAV2 depth maps for CVPR paper datasets (NH-HAZE, O-HAZE, I-HAZE).
Used for quantitative analysis of infinity depth noise.
"""

import os
import glob
import torch
import gc
import numpy as np
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from tqdm import tqdm

# 终极防爆装甲
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 数据集配置
DATASETS = {
    'NH-HAZE': {
        'hazy_dir': './data/NH-HAZE/hazy',
        'save_dir': './data/NH-HAZE/depth_dav2',
    },
    'O-HAZE': {
        'hazy_dir': './data/O-HAZE/hazy',
        'save_dir': './data/O-HAZE/depth_dav2',
    },
    'I-HAZE': {
        'hazy_dir': './data/I-HAZE/hazy',
        'save_dir': './data/I-HAZE/depth_dav2',
    },
}

MAX_SIZE = 1024  # 限制最大分辨率

def generate_depths(hazy_dir, save_dir, model, processor, device):
    """Generate depth maps for all images in a directory."""
    os.makedirs(save_dir, exist_ok=True)
    
    img_paths = glob.glob(os.path.join(hazy_dir, "*.*"))
    img_paths = [p for p in img_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"  Found {len(img_paths)} images")
    
    if len(img_paths) == 0:
        print(f"  ⚠️ No images found in {hazy_dir}")
        return 0
    
    processed = 0
    with torch.no_grad():
        for img_path in tqdm(img_paths, desc=f"  Processing"):
            filename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(filename)[0]
            save_path = os.path.join(save_dir, f"{name_no_ext}.npy")
            
            # Skip if already exists
            if os.path.exists(save_path) and os.path.getsize(save_path) > 128:
                continue
            
            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                
                # Resize if too large
                if max(w, h) > MAX_SIZE:
                    r = MAX_SIZE / float(max(w, h))
                    img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
                
                inputs = processor(images=img, return_tensors="pt").to(device)
                outputs = model(**inputs)
                depth = outputs.predicted_depth
                
                # Resize back to original resolution
                depth_resized = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False
                ).squeeze()
                
                # Normalize to 0-1
                d_min, d_max = depth_resized.min(), depth_resized.max()
                depth_norm = (depth_resized - d_min) / (d_max - d_min + 1e-8)
                
                depth_npy = depth_norm.cpu().numpy().astype(np.float16)
                np.save(save_path, depth_npy)
                processed += 1
                
            except Exception as e:
                print(f"\n  ⚠️ Failed: {filename} - {e}")
    
    return processed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("🚀 DAV2 Depth Map Generator for CVPR Datasets")
    print("=" * 60)
    print(f"Device: {device}\n")
    
    # Load DAV2 model
    model_repo = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"⏳ Loading {model_repo}...")
    processor = AutoImageProcessor.from_pretrained(model_repo)
    model = AutoModelForDepthEstimation.from_pretrained(model_repo).to(device).eval()
    print("✅ Model loaded\n")
    
    total_processed = 0
    for name, cfg in DATASETS.items():
        print(f"\n📦 Processing {name}:")
        print(f"   Source: {cfg['hazy_dir']}")
        print(f"   Target: {cfg['save_dir']}")
        
        count = generate_depths(cfg['hazy_dir'], cfg['save_dir'], model, processor, device)
        total_processed += count
        print(f"   ✅ Generated {count} depth maps")
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! Generated {total_processed} depth maps total.")
    print("=" * 60)

if __name__ == "__main__":
    main()