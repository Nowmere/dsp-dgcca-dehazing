"""
从 RESIDE OTS 数据集随机抽取 10% 图像对，创建一个迷你训练集。
"""
import os
import random
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_HAZY  = os.path.join(BASE, "data", "RESIDE", "OTS", "hazy")
SRC_CLEAR = os.path.join(BASE, "data", "RESIDE", "OTS", "clear")
SRC_DEPTH = os.path.join(BASE, "data", "RESIDE", "OTS", "depth_dav2")

DST_HAZY  = os.path.join(BASE, "data", "RESIDE", "OTS_MINI", "hazy")
DST_CLEAR = os.path.join(BASE, "data", "RESIDE", "OTS_MINI", "clear")
DST_DEPTH = os.path.join(BASE, "data", "RESIDE", "OTS_MINI", "depth_dav2")

RATIO = 0.10
SEED = 42

random.seed(SEED)

# 收集 hazy 文件名
hazy_files = sorted([f for f in os.listdir(SRC_HAZY) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
total = len(hazy_files)
sample_n = max(1, int(total * RATIO))
sampled = sorted(random.sample(hazy_files, sample_n))

print(f"Total OTS images: {total}")
print(f"Sampled (10%): {sample_n}")

# 创建目标目录
for d in [DST_HAZY, DST_CLEAR, DST_DEPTH]:
    os.makedirs(d, exist_ok=True)

# 复制对应文件
prefix_map = {}
for clear_name in os.listdir(SRC_CLEAR):
    clear_name_lower = clear_name.lower()
    if clear_name_lower.endswith(".png") or clear_name_lower.endswith(".jpg") or clear_name_lower.endswith(".jpeg"):
        prefix = os.path.splitext(clear_name)[0]
        prefix_map[prefix] = clear_name

copied = 0
for hazy_name in sampled:
    hazy_name_no_ext = os.path.splitext(hazy_name)[0]
    
    # 复制 hazy
    src_h = os.path.join(SRC_HAZY, hazy_name)
    dst_h = os.path.join(DST_HAZY, hazy_name)
    if not os.path.exists(dst_h):
        shutil.copy2(src_h, dst_h)
    
    # 找对应的 clear 和 depth
    prefix = hazy_name.split('_')[0]
    
    if prefix in prefix_map:
        clear_name = prefix_map[prefix]
        src_c = os.path.join(SRC_CLEAR, clear_name)
        dst_c = os.path.join(DST_CLEAR, clear_name)
        if not os.path.exists(dst_c):
            shutil.copy2(src_c, dst_c)
    
    # 复制 depth (如果有)
    depth_name = hazy_name_no_ext + ".pt"
    depth_path = os.path.join(SRC_DEPTH, depth_name)
    if os.path.exists(depth_path):
        dst_d = os.path.join(DST_DEPTH, depth_name)
        if not os.path.exists(dst_d):
            shutil.copy2(depth_path, dst_d)
    
    copied += 1

print(f"Copied {copied} image pairs to {DST_HAZY}")
print("Done!")