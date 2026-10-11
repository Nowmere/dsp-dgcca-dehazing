"""
generate_zoomed_comparisons.py
对 O-HAZE 和 NH-HAZE 的最佳 SSIM 案例图，计算 Baseline vs Ours 的
逐像素 MSE 热力图，定位差异最大的 256×256 区域，在原图上画红框，
并在对比面板右侧追加放大 callout (2×2: Baseline crop / Ours crop / Diff heatmap / GT crop)
"""
import os, numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob
from skimage.metrics import structural_similarity as ssim

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

FONT = None
def _font(size=18):
    global FONT
    if FONT is None:
        try: FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        except: FONT = ImageFont.load_default()
    return FONT

def label(img, txt, color=(255,255,255)):
    h,w = img.shape[:2]
    bar = 40
    lb = np.ones((h+bar, w, 3), dtype=np.uint8)*40
    lb[bar:,:,:] = img
    pi = Image.fromarray(lb)
    d = ImageDraw.Draw(pi)
    bbox = d.textbbox((0,0), txt, font=_font())
    tx = (w-bbox[2]-bbox[0])//2
    ty = (bar-bbox[3]-bbox[1])//2
    d.text((tx,ty), txt, fill=color, font=_font())
    return np.array(pi)

def hstack(imgs, gap=4):
    h = max(i.shape[0] for i in imgs)
    tw = sum(i.shape[1] for i in imgs)+gap*(len(imgs)-1)
    c = np.ones((h,tw,3),dtype=np.uint8)*255
    x=0
    for i in imgs: c[:i.shape[0],x:x+i.shape[1]]=i; x+=i.shape[1]+gap
    return c

def vstack(imgs, gap=6):
    w = max(i.shape[1] for i in imgs)
    th = sum(i.shape[0] for i in imgs)+gap*(len(imgs)-1)
    c = np.ones((th,w,3),dtype=np.uint8)*255
    y=0
    for i in imgs: c[y:y+i.shape[0],:i.shape[1]]=i; y+=i.shape[0]+gap
    return c

def load(path, size=None):
    im = np.array(Image.open(path).convert("RGB"))
    if size: im = np.array(Image.fromarray(im).resize(size, Image.LANCZOS))
    return im

def find_hotspot(base, ours, crop_size=256):
    """返回 (y, x) 使得 base 和 ours 在该 crop 内 MSE 最大"""
    diff = np.mean((base.astype(np.float32) - ours.astype(np.float32))**2, axis=2)
    h, w = diff.shape
    best_val, best_y, best_x = -1, 0, 0
    step = crop_size // 4
    for y in range(0, h - crop_size + 1, step):
        for x in range(0, w - crop_size + 1, step):
            val = np.mean(diff[y:y+crop_size, x:x+crop_size])
            if val > best_val:
                best_val, best_y, best_x = val, y, x
    # fine-tune around best
    y0 = max(0, best_y - crop_size//2)
    x0 = max(0, best_x - crop_size//2)
    y0 = min(y0, h - crop_size)
    x0 = min(x0, w - crop_size)
    return y0, x0

def draw_rect(img, y, x, crop_size, color=(255,0,0), thickness=3):
    out = img.copy()
    cv2 = __import__('cv2')
    return cv2.rectangle(out, (x, y), (x+crop_size, y+crop_size), color, thickness)

def diff_heatmap(base, ours):
    diff = np.mean((base.astype(np.float32) - ours.astype(np.float32))**2, axis=2)
    diff = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)

import cv2

# ========== NH-HAZE SSIM best cases ==========
print("="*60)
print("NH-HAZE zoomed comparisons...")
nh_targets = [
    ("nh34_hazy.png", 0.4039, 0.4584, 0.0545),
    ("nh38_hazy.png", 0.5234, 0.5667, 0.0432),
    ("nh32_hazy.png", 0.5334, 0.5677, 0.0342),
]

nh_dirs = {
    "hazy": os.path.join(BASE, "data", "NH-HAZE", "hazy"),
    "gt": os.path.join(BASE, "data", "NH-HAZE", "GT"),
    "base": os.path.join(BASE, "results", "NH-HAZE_Baseline_epoch120"),
    "ours": os.path.join(BASE, "results", "NH-HAZE_e2e_epoch120"),
}

rows = []
for fname, b_s, o_s, d in nh_targets:
    hp = os.path.join(nh_dirs["hazy"], fname)
    gt_name = fname.replace("_hazy", "")
    gp = glob.glob(os.path.join(nh_dirs["gt"], f"{os.path.splitext(gt_name)[0]}.*"))[0]
    bp = os.path.join(nh_dirs["base"], f"{os.path.splitext(fname)[0]}_dehazed.png")
    op = os.path.join(nh_dirs["ours"], f"{os.path.splitext(fname)[0]}_dehazed.png")
    
    hazy = load(hp)
    gt = load(gp)
    base = load(bp)
    ours = load(op)
    h,w = gt.shape[:2]
    hazy = load(hp, (w,h))
    base = load(bp, (w,h))
    ours = load(op, (w,h))
    
    # Find hotspot
    y0, x0 = find_hotspot(base, ours, crop_size=256)
    
    # Draw red rect on base and ours
    base_marked = draw_rect(base, y0, x0, 256)
    ours_marked = draw_rect(ours, y0, x0, 256)
    
    # Crop callouts
    base_crop = base[y0:y0+256, x0:x0+256]
    ours_crop = ours[y0:y0+256, x0:x0+256]
    gt_crop   = gt[y0:y0+256, x0:x0+256]
    diff_crop = diff_heatmap(base_crop, ours_crop)
    
    # Build row: hazy | base_marked | ours_marked | [callout: base_crop | ours_crop | diff_crop | gt_crop] | gt
    callout = hstack([
        label(base_crop, "Baseline (crop)", (255,100,100)),
        label(ours_crop, "Ours (crop)", (100,255,100)),
        label(diff_crop, "MSE Diff"),
        label(gt_crop, "GT (crop)"),
    ], gap=3)
    
    row_img = hstack([
        label(hazy, "Hazy Input"),
        label(base_marked, f"Baseline (SSIM {b_s:.4f})", (255,100,100)),
        label(ours_marked, f"Ours (SSIM {o_s:.4f}, +{d:.4f})", (100,255,100)),
        callout,
        label(gt, "Ground Truth"),
    ], gap=4)
    
    rows.append(label(row_img, f"({chr(97+len(rows))}) {fname}  (SSIM +{d:.4f}, hotspot marked)", (100,255,100)))

nh_fig = vstack(rows, gap=10)
out_path = os.path.join(OUT, "nh-haze_best_ssim_zoomed.pdf")
Image.fromarray(nh_fig).save(out_path)
print(f"Saved: {out_path}")

# ========== O-HAZE SSIM best cases ==========
print("\n" + "="*60)
print("O-HAZE zoomed comparisons...")
oh_targets = [
    ("20_outdoor.jpg", 0.4515, 0.4806, 0.0292),
    ("31_outdoor.jpg", 0.6577, 0.6847, 0.0269),
    ("28_outdoor.jpg", 0.5438, 0.5698, 0.0260),
]

oh_dirs = {
    "hazy": os.path.join(BASE, "data", "O-HAZE", "hazy"),
    "gt": os.path.join(BASE, "data", "O-HAZE", "GT"),
    "base": os.path.join(BASE, "results", "O-HAZE_Baseline_epoch120"),
    "ours": os.path.join(BASE, "results", "O-HAZE_e2e_epoch120"),
    "cplir": os.path.join(BASE, "CPLIR", "results_cpl_ohaze"),
    "c2p": os.path.join(BASE, "C2PNet", "results_ohaze"),
}

rows = []
for fname, b_s, o_s, d in oh_targets:
    hp = os.path.join(oh_dirs["hazy"], fname)
    gp = glob.glob(os.path.join(oh_dirs["gt"], f"{os.path.splitext(fname)[0]}.*"))[0]
    bp = os.path.join(oh_dirs["base"], f"{os.path.splitext(fname)[0]}_dehazed.png")
    op = os.path.join(oh_dirs["ours"], f"{os.path.splitext(fname)[0]}_dehazed.png")
    
    hazy = load(hp)
    gt = load(gp)
    base = load(bp)
    ours = load(op)
    h,w = gt.shape[:2]
    hazy = load(hp, (w,h))
    base = load(bp, (w,h))
    ours = load(op, (w,h))
    
    y0, x0 = find_hotspot(base, ours, crop_size=256)
    
    base_marked = draw_rect(base, y0, x0, 256)
    ours_marked = draw_rect(ours, y0, x0, 256)
    
    base_crop = base[y0:y0+256, x0:x0+256]
    ours_crop = ours[y0:y0+256, x0:x0+256]
    gt_crop   = gt[y0:y0+256, x0:x0+256]
    diff_crop = diff_heatmap(base_crop, ours_crop)
    
    callout = hstack([
        label(base_crop, "Baseline (crop)", (255,100,100)),
        label(ours_crop, "Ours (crop)", (100,255,100)),
        label(diff_crop, "MSE Diff"),
        label(gt_crop, "GT (crop)"),
    ], gap=3)
    
    row_img = hstack([
        label(hazy, "Hazy Input"),
        label(base_marked, f"Baseline (SSIM {b_s:.4f})", (255,100,100)),
        label(ours_marked, f"Ours (SSIM {o_s:.4f}, +{d:.4f})", (100,255,100)),
        callout,
        label(gt, "Ground Truth"),
    ], gap=4)
    
    rows.append(label(row_img, f"({chr(97+len(rows))}) {fname}  (SSIM +{d:.4f}, hotspot marked)", (100,255,100)))

oh_fig = vstack(rows, gap=10)
out_path = os.path.join(OUT, "o-haze_best_ssim_zoomed.pdf")
Image.fromarray(oh_fig).save(out_path)
print(f"Saved: {out_path}")

print("\nDone! Zoomed comparison figures generated.")