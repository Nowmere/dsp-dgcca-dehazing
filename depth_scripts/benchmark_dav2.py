"""快速测量 DAV2-Small 的 FLOPs 和推理时间"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEVICE = torch.device("cuda")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 加载模型
print("[Benchmark] Loading DAV2-Small...")
processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(DEVICE).eval()

# 2. 计算参数量
params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")

# 3. 输入图片
img_path = os.path.join(BASE_DIR, "data", "NH-HAZE", "hazy", "nh1_hazy.png")
img = Image.open(img_path).convert("RGB")
print(f"  Input size: {img.size}")

# 4. FLOPs 计算 (用 fvcore 如果可用)
try:
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    flops = FlopCountAnalysis(model, inputs["pixel_values"])
    total_flops = flops.total()
    print(f"  FLOPs: {total_flops:,.0f} ({total_flops/1e9:.2f} GFLOPs)")
except ImportError:
    print("  [SKIP] fvcore not installed, estimating FLOPs from literature (~24M params)")
    total_flops = None

# 5. 推理时间 (warmup + average)
print(f"\n[Benchmark] Measuring inference time...")

inputs = processor(images=img, return_tensors="pt").to(DEVICE)

# Warmup
with torch.no_grad():
    for _ in range(10):
        _ = model(**inputs)
    torch.cuda.synchronize()

# Measure
times = []
with torch.no_grad():
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model(**inputs)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1 - t0) * 1000)  # ms

avg = np.mean(times)
std = np.std(times)
print(f"  Inference time: {avg:.1f} +/- {std:.1f} ms (avg over 50 runs)")
print(f"  FPS: {1000/avg:.1f}")

# 6. 不同分辨率
print(f"\n[Benchmark] Resolution sweep:")
for size_name, size in [("256x256", (256, 256)), ("512x512", (512, 512)), ("1024x768", (1024, 768)), ("1920x1080", (1920, 1080))]:
    img_resized = img.resize(size, Image.LANCZOS)
    inputs_rs = processor(images=img_resized, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        for _ in range(5):
            _ = model(**inputs_rs)
        torch.cuda.synchronize()
    
    ts = []
    with torch.no_grad():
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = model(**inputs_rs)
            torch.cuda.synchronize()
            t1 = time.time()
            ts.append((t1 - t0) * 1000)
    
    print(f"  {size_name}: {np.mean(ts):.1f} ms (FPS: {1000/np.mean(ts):.1f})")

print("\n[Benchmark] Done!")