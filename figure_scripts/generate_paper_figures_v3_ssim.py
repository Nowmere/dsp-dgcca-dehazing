"""
generate_paper_figures_v3_ssim.py
基于 SSIM 差值重新扫描 SOTS-Outdoor 500 张图：
- Best cases: Ours SSIM 远高于 Baseline (结构保护最好的图)
- Worst cases: Baseline SSIM 远高于 Ours (结构退步的图)
生成多列对比面板，标签标明 SSIM 差值
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

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
    font = get_font(18)
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
        canvas[y:y + im.shape[0], :im.shape[1]] = im
        y += im.shape[0] + gap
    return canvas

# Paths
base_dir = os.path.join(BASE_DIR, "results", "SOTS_outdoor_Baseline_epoch120")
ours_dir = os.path.join(BASE_DIR, "results", "sots_outdoor_e2e_epoch120")
gt_dir = os.path.join(BASE_DIR, "data", "RESIDE", "SOTS", "outdoor", "clear")
hazy_dir = os.path.join(BASE_DIR, "data", "RESIDE", "SOTS", "outdoor", "hazy")

print("=" * 70)
print("SSIM-based comparison: SOTS-Outdoor (500 images)")
print("=" * 70)

hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

results = []
for fname in tqdm(hazy_files, desc="Computing SSIM"):
    name_noext = os.path.splitext(fname)[0]
    result_name = f"{name_noext}_dehazed.png"
    
    base_path = os.path.join(base_dir, result_name)
    ours_path = os.path.join(ours_dir, result_name)
    if not (os.path.exists(base_path) and os.path.exists(ours_path)):
        continue
    
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
    base_img = np.array(Image.fromarray(base_img).resize((w, h), Image.LANCZOS))
    ours_img = np.array(Image.fromarray(ours_img).resize((w, h), Image.LANCZOS))
    
    try:
        b_ssim = ssim(gt_img, base_img, data_range=255, channel_axis=2)
        o_ssim = ssim(gt_img, ours_img, data_range=255, channel_axis=2)
        delta = o_ssim - b_ssim
        results.append((fname, b_ssim, o_ssim, delta, base_path, ours_path, gt_path))
    except:
        continue

# Sort by SSIM delta: best first (positive = Ours better)
results.sort(key=lambda x: x[3], reverse=True)

# ======================
# Best SSIM cases (top 10)
# ======================
print(f"\nTop 10 BEST SSIM cases (Ours > Baseline, largest positive SSIM delta):")
print(f"{'Rank':<6} {'Filename':<30} {'Baseline':>10} {'Ours':>10} {'Delta':>10}")
for i, (name, b, o, d, *_) in enumerate(results[:10]):
    print(f"{i+1:<6} {name:<30} {b:>10.4f} {o:>10.4f} {d:>+10.4f}")

# Generate single multi-row figure: best 3
print(f"\nGenerating best SSIM cases multi-row figure...")
best_panels = []
for i in range(min(3, len(results))):
    fname, b_ssim, o_ssim, delta, base_path, ours_path, gt_path = results[i]
    hazy_path = os.path.join(hazy_dir, fname)
    
    hazy_img = np.array(Image.open(hazy_path).convert("RGB"))
    gt_img   = np.array(Image.open(gt_path).convert("RGB"))
    base_img = np.array(Image.open(base_path).convert("RGB"))
    ours_img = np.array(Image.open(ours_path).convert("RGB"))
    
    h, w = gt_img.shape[:2]
    hazy_img = np.array(Image.fromarray(hazy_img).resize((w, h), Image.LANCZOS))
    base_img = np.array(Image.fromarray(base_img).resize((w, h), Image.LANCZOS))
    ours_img = np.array(Image.fromarray(ours_img).resize((w, h), Image.LANCZOS))
    
    # Horizontal row for this image
    panels_row = [
        add_label(hazy_img, "Hazy Input"),
        add_label(base_img, f"FFA-Net Baseline (SSIM {b_ssim:.4f})", color=(255, 100, 100)),
        add_label(ours_img, f"Ours (SSIM {o_ssim:.4f}, +{delta:.4f})", color=(100, 255, 100)),
        add_label(gt_img,   "Ground Truth"),
    ]
    h_max = max(p.shape[0] for p in panels_row)
    total_w = sum(p.shape[1] for p in panels_row) + 4 * (len(panels_row) - 1)
    row = np.ones((h_max, total_w, 3), dtype=np.uint8) * 255
    x = 0
    for p in panels_row:
        row[:p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + 4
    # Add row label
    row_labeled = add_label(row, f"({chr(97+i)}) {fname}  (SSIM +{delta:.4f})", color=(100, 255, 100))
    best_panels.append(row_labeled)

best_combined = vstack_images(best_panels, gap=10)
best_out = os.path.join(OUT_DIR, "sots_best_ssim.pdf")
Image.fromarray(best_combined).save(best_out)
print(f"Saved: {best_out}")

# ======================
# Worst SSIM cases (bottom 10)
# ======================
worst_results = sorted(results, key=lambda x: x[3])  # ascending = worst first
print(f"\nTop 10 WORST SSIM cases (Baseline > Ours, largest negative SSIM delta):")
print(f"{'Rank':<6} {'Filename':<30} {'Baseline':>10} {'Ours':>10} {'Delta':>10}")
for i, (name, b, o, d, *_) in enumerate(worst_results[:10]):
    print(f"{i+1:<6} {name:<30} {b:>10.4f} {o:>10.4f} {d:>+10.4f}")

# Generate single multi-row figure: worst 3
print(f"\nGenerating worst SSIM cases multi-row figure...")
worst_panels = []
for i in range(min(3, len(worst_results))):
    fname, b_ssim, o_ssim, delta, base_path, ours_path, gt_path = worst_results[i]
    hazy_path = os.path.join(hazy_dir, fname)
    
    hazy_img = np.array(Image.open(hazy_path).convert("RGB"))
    gt_img   = np.array(Image.open(gt_path).convert("RGB"))
    base_img = np.array(Image.open(base_path).convert("RGB"))
    ours_img = np.array(Image.open(ours_path).convert("RGB"))
    
    h, w = gt_img.shape[:2]
    hazy_img = np.array(Image.fromarray(hazy_img).resize((w, h), Image.LANCZOS))
    base_img = np.array(Image.fromarray(base_img).resize((w, h), Image.LANCZOS))
    ours_img = np.array(Image.fromarray(ours_img).resize((w, h), Image.LANCZOS))
    
    panels_row = [
        add_label(hazy_img, "Hazy Input"),
        add_label(base_img, f"FFA-Net Baseline (SSIM {b_ssim:.4f})", color=(100, 255, 100)),
        add_label(ours_img, f"Ours (SSIM {o_ssim:.4f}, {delta:+.4f})", color=(255, 100, 100)),
        add_label(gt_img,   "Ground Truth"),
    ]
    h_max = max(p.shape[0] for p in panels_row)
    total_w = sum(p.shape[1] for p in panels_row) + 4 * (len(panels_row) - 1)
    row = np.ones((h_max, total_w, 3), dtype=np.uint8) * 255
    x = 0
    for p in panels_row:
        row[:p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + 4
    row_labeled = add_label(row, f"({chr(97+i)}) {fname}  (SSIM {delta:+.4f})", color=(255, 100, 100))
    worst_panels.append(row_labeled)

worst_combined = vstack_images(worst_panels, gap=10)
worst_out = os.path.join(OUT_DIR, "sots_worst_ssim.pdf")
Image.fromarray(worst_combined).save(worst_out)
print(f"Saved: {worst_out}")

print("\n" + "=" * 70)
print("All SSIM-based comparison figures generated!")
print(f"  Best SSIM:  {best_out}")
print(f"  Worst SSIM: {worst_out}")
print("=" * 70)