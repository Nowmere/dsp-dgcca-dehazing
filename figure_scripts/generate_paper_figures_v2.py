"""
generate_paper_figures_v2.py
Task 1: Combine Top1/Top5/Top16 SOTS-Outdoor best-case panels into a single figure.
Task 2: Find worst-case SOTS-Outdoor images (Ours < Baseline) and generate comparison panels.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def get_font(size=20):
    paths = ["C:/Windows/Fonts/arial.ttf"]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def add_label(img_np, text, color=(255, 255, 255)):
    h, w = img_np.shape[:2]
    bar_h = 40
    labeled = np.ones((h + bar_h, w, 3), dtype=np.uint8) * 40
    labeled[bar_h:, :, :] = img_np
    pil = Image.fromarray(labeled)
    draw = ImageDraw.Draw(pil)
    try:
        font = get_font(18)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tx = (w - bbox[2] - bbox[0]) // 2
    ty = (bar_h - bbox[3] - bbox[1]) // 2
    draw.text((tx, ty), text, fill=color, font=font)
    return np.array(pil)

def vstack_images(imgs, gap=4):
    w = max(i.shape[1] for i in imgs)
    total_h = sum(i.shape[0] for i in imgs) + gap * (len(imgs) - 1)
    canvas = np.ones((total_h, w, 3), dtype=np.uint8) * 255
    y = 0
    for im in imgs:
        h_i = im.shape[0]
        canvas[y:y + h_i, :im.shape[1]] = im
        y += h_i + gap
    return canvas

# =====================================================
# Task 1: Combine Top1, Top5, Top16 SOTS-Outdoor panels
# =====================================================
print("=" * 70)
print("Task 1: Combining Top1/Top5/Top16 SOTS-Outdoor best-case panels...")

top_dir = os.path.join(BASE_DIR, "paper_figures", "Top20_outdoor")
top1_path = os.path.join(top_dir, "Top01_1909_0.85_0.2_Win_7.63dB.png")
top5_path = os.path.join(top_dir, "Top05_0329_0.8_0.12_Win_4.81dB.png")
top16_path = os.path.join(top_dir, "Top16_0215_0.8_0.08_Win_3.70dB.png")

panels = []
for path, label in [(top1_path, "(a) 1909_0.85_0.2 (Ours +7.63 dB)"), 
                     (top5_path, "(b) 0329_0.8_0.12 (Ours +4.81 dB)"),
                     (top16_path, "(c) 0215_0.8_0.08 (Ours +3.70 dB)")]:
    img = np.array(Image.open(path).convert("RGB"))
    panels.append(add_label(img, label, color=(100, 255, 100)))

combined = vstack_images(panels, gap=10)
out_path = os.path.join(OUT_DIR, "sots_best_cases.pdf")
Image.fromarray(combined).save(out_path)
print(f"Saved: {out_path}")

# =====================================================
# Task 2: Find worst-case SOTS-Outdoor images (Ours < Baseline)
# =====================================================
print("\n" + "=" * 70)
print("Task 2: Scanning SOTS-Outdoor for worst-case images...")

from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# Paths
base_dir = os.path.join(BASE_DIR, "results", "SOTS_outdoor_Baseline_epoch120")
ours_dir = os.path.join(BASE_DIR, "results", "sots_outdoor_e2e_epoch120")
gt_dir = os.path.join(BASE_DIR, "data", "RESIDE", "SOTS", "outdoor", "clear")
hazy_dir = os.path.join(BASE_DIR, "data", "RESIDE", "SOTS", "outdoor", "hazy")

# Scan hazy files
hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

results = []
for fname in tqdm(hazy_files, desc="Scanning"):
    # Baseline output naming: hazy "0001_0.8_0.2.jpg" → result "0001_0.8_0.2_dehazed.png"
    name_noext = os.path.splitext(fname)[0]
    result_name = f"{name_noext}_dehazed.png"
    base_path = os.path.join(base_dir, result_name)
    if not os.path.exists(base_path):
        continue
    
    # Ours output
    ours_path = os.path.join(ours_dir, result_name)
    if not os.path.exists(ours_path):
        continue
    
    # GT
    prefix = fname.split('_')[0]
    candidates = glob.glob(os.path.join(gt_dir, f"{prefix}.*"))
    if not candidates:
        continue
    gt_path = candidates[0]
    
    try:
        base_img = np.array(Image.open(base_path).convert("RGB"))
        ours_img = np.array(Image.open(ours_path).convert("RGB"))
        gt_img = np.array(Image.open(gt_path).convert("RGB"))
    except:
        continue
    
    h, w = gt_img.shape[:2]
    if base_img.shape[:2] != (h, w):
        base_img = np.array(Image.fromarray(base_img).resize((w, h), Image.LANCZOS))
    if ours_img.shape[:2] != (h, w):
        ours_img = np.array(Image.fromarray(ours_img).resize((w, h), Image.LANCZOS))
    
    try:
        b_psnr = psnr(gt_img, base_img, data_range=255)
        o_psnr = psnr(gt_img, ours_img, data_range=255)
        delta = o_psnr - b_psnr
        results.append((fname, b_psnr, o_psnr, delta, hazy_path, gt_path, base_path, ours_path))
    except:
        continue

# Sort by delta (ascending = worst cases where Baseline beats Ours the most)
results.sort(key=lambda x: x[3])

# Print top 10 worst cases
print(f"\nTop 10 WORST cases (Baseline > Ours, largest negative delta):")
print(f"{'Rank':<6} {'Filename':<30} {'Baseline':>10} {'Ours':>10} {'Delta':>10}")
for i, (name, b, o, d, *_) in enumerate(results[:10]):
    print(f"{i+1:<6} {name:<30} {b:>10.3f} {o:>10.3f} {d:>+10.3f}")

# Generate comparison panels for top 3 worst cases
print(f"\nGenerating comparison panels for top 3 worst cases...")
for i in range(min(3, len(results))):
    fname, b_psnr, o_psnr, delta, hazy_path, gt_path, base_path, ours_path = results[i]
    
    hazy_img = np.array(Image.open(hazy_path).convert("RGB"))
    gt_img = np.array(Image.open(gt_path).convert("RGB"))
    base_img = np.array(Image.open(base_path).convert("RGB"))
    ours_img = np.array(Image.open(ours_path).convert("RGB"))
    
    # Resize all to same size as GT
    h, w = gt_img.shape[:2]
    for arr in [hazy_img, base_img, ours_img]:
        if arr.shape[:2] != (h, w):
            arr = np.array(Image.fromarray(arr).resize((w, h), Image.LANCZOS))
    hazy_img = np.array(Image.fromarray(hazy_img).resize((w, h), Image.LANCZOS))
    base_img = np.array(Image.fromarray(base_img).resize((w, h), Image.LANCZOS))
    ours_img = np.array(Image.fromarray(ours_img).resize((w, h), Image.LANCZOS))
    
    panels = [
        add_label(hazy_img, "Hazy Input"),
        add_label(base_img, f"FFA-Net Baseline ({b_psnr:.2f} dB)", color=(255, 100, 100)),
        add_label(ours_img, f"Ours ({o_psnr:.2f} dB)", color=(255, 255, 100)),
        add_label(gt_img, "Ground Truth"),
    ]
    
    # Horizontal stack
    h_max = max(p.shape[0] for p in panels)
    total_w = sum(p.shape[1] for p in panels) + 4 * (len(panels) - 1)
    combined = np.ones((h_max, total_w, 3), dtype=np.uint8) * 255
    x = 0
    for p in panels:
        combined[:p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + 4
    
    out_name = f"worst_case_{i+1}_{os.path.splitext(fname)[0]}.pdf"
    out_path = os.path.join(OUT_DIR, out_name)
    Image.fromarray(combined).save(out_path)
    print(f"Saved: {out_path} (Baseline: {b_psnr:.2f}, Ours: {o_psnr:.2f}, Delta: {delta:+.2f})")

print("\n" + "=" * 70)
print("All tasks complete!")
print(f"Output directory: {OUT_DIR}")
print("=" * 70)