"""
generate_best_comparisons.py
Mengenerate panel perbandingan untuk gambar-gambar best-case (top-3 NH-HAZE + top-3 O-HAZE).
Menggunakan output Baseline dan Ours yang sudah ada; menambahkan label.
Menyimpan hasil ke paper/figures/ sebagai file PDF terpisah untuk setiap kandidat.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
#  Utilitas
# ----------------------------------------------------------------------
def get_font(size=20):
    paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
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
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tx = (w - bbox[2] - bbox[0]) // 2
    ty = (bar_h - bbox[3] - bbox[1]) // 2
    draw.text((tx, ty), text, fill=color, font=font)
    return np.array(pil)


def hstack_images(imgs, gap=4):
    h = max(i.shape[0] for i in imgs)
    total_w = sum(i.shape[1] for i in imgs) + gap * (len(imgs) - 1)
    canvas = np.ones((h, total_w, 3), dtype=np.uint8) * 255
    x = 0
    for im in imgs:
        canvas[:im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    return canvas


def load_image(path, size=None):
    """Load image, convert to RGB, optionally resize (width, height)."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return np.array(img)


def find_file(dir_path, prefix):
    """Cari file di direktori yang namanya dimulai dengan prefix."""
    candidates = glob.glob(os.path.join(dir_path, f"{prefix}*"))
    return candidates[0] if candidates else None


# ----------------------------------------------------------------------
#  NH-HAZE Teaser (hazy + baseline + ours + GT)
# ----------------------------------------------------------------------
NH_HAZY_DIR = os.path.join(BASE_DIR, "data", "NH-HAZE", "hazy")
NH_GT_DIR   = os.path.join(BASE_DIR, "data", "NH-HAZE", "GT")
NH_BASE_DIR = os.path.join(BASE_DIR, "results", "NH-HAZE_Baseline_epoch120")
NH_OURS_DIR = os.path.join(BASE_DIR, "results", "NH-HAZE_e2e_epoch120")

nh_candidates = ["nh51_hazy.png", "nh34_hazy.png", "nh38_hazy.png"]

for cand in nh_candidates:
    hazy_path = os.path.join(NH_HAZY_DIR, cand)
    gt_prefix = os.path.splitext(cand)[0].replace("_hazy", "")
    gt_path = find_file(NH_GT_DIR, gt_prefix)
    base_path = os.path.join(NH_BASE_DIR, f"{os.path.splitext(cand)[0]}_dehazed.png")
    ours_path = os.path.join(NH_OURS_DIR, f"{os.path.splitext(cand)[0]}_dehazed.png")
    
    if not all([os.path.exists(p) for p in [hazy_path, gt_path, base_path, ours_path]]):
        print(f"SKIP {cand}: missing files")
        continue
    
    hazy = load_image(hazy_path)
    gt   = load_image(gt_path, (hazy.shape[1], hazy.shape[0]))
    base = load_image(base_path, (hazy.shape[1], hazy.shape[0]))
    ours = load_image(ours_path, (hazy.shape[1], hazy.shape[0]))
    
    # panel
    panels = [
        add_label(hazy, "Hazy Input"),
        add_label(base, "FFA-Net (Baseline)", color=(255, 100, 100)),
        add_label(ours, "Ours (DG-CCA+DAV2+SkyTV)", color=(100, 255, 100)),
        add_label(gt,   "Ground Truth"),
    ]
    combined = hstack_images(panels, gap=8)
    
    out_name = f"teaser_{os.path.splitext(cand)[0]}.pdf"
    out_path = os.path.join(OUT_DIR, out_name)
    Image.fromarray(combined).save(out_path)
    print(f"Saved: {out_path}")

# ----------------------------------------------------------------------
#  O-HAZE Visual Comparison (hazy + baseline + cplir + c2pnet + ours + GT)
# ----------------------------------------------------------------------
OH_HAZY_DIR = os.path.join(BASE_DIR, "data", "O-HAZE", "hazy")
OH_GT_DIR   = os.path.join(BASE_DIR, "data", "O-HAZE", "GT")
OH_BASE_DIR = os.path.join(BASE_DIR, "results", "O-HAZE_Baseline_epoch120")
OH_OURS_DIR = os.path.join(BASE_DIR, "results", "O-HAZE_e2e_epoch120")
OH_CPLIR_DIR = os.path.join(BASE_DIR, "CPLIR", "results_cpl_ohaze")
OH_C2P_DIR   = os.path.join(BASE_DIR, "C2PNet", "results_ohaze")

oh_candidates = ["28_outdoor.jpg", "20_outdoor.jpg", "31_outdoor.jpg"]

for cand in oh_candidates:
    hazy_path = os.path.join(OH_HAZY_DIR, cand)
    gt_path   = find_file(OH_GT_DIR, os.path.splitext(cand)[0])
    base_dehazed = f"{os.path.splitext(cand)[0]}_dehazed.png"
    base_path = os.path.join(OH_BASE_DIR, base_dehazed)
    ours_path = os.path.join(OH_OURS_DIR, base_dehazed)
    cplir_path = os.path.join(OH_CPLIR_DIR, cand)
    c2p_path   = os.path.join(OH_C2P_DIR, cand)
    
    if not all([os.path.exists(p) for p in [hazy_path, gt_path, base_path, ours_path]]):
        print(f"SKIP {cand}: missing files")
        continue
    
    hazy  = load_image(hazy_path)
    gt    = load_image(gt_path, (hazy.shape[1], hazy.shape[0]))
    base  = load_image(base_path, (hazy.shape[1], hazy.shape[0]))
    ours  = load_image(ours_path, (hazy.shape[1], hazy.shape[0]))
    
    panels = [
        add_label(hazy, "Hazy Input"),
        add_label(base, "FFA-Net (Baseline)", color=(255, 100, 100)),
    ]
    
    # SOTA models jika tersedia
    if os.path.exists(cplir_path):
        cplir = load_image(cplir_path, (hazy.shape[1], hazy.shape[0]))
        panels.append(add_label(cplir, "CPLIR (CVPR 2024)"))
    if os.path.exists(c2p_path):
        c2p = load_image(c2p_path, (hazy.shape[1], hazy.shape[0]))
        panels.append(add_label(c2p, "C2PNet (CVPR 2023)"))
    
    panels.append(add_label(ours, "Ours (DG-CCA+DAV2+SkyTV)", color=(100, 200, 100)))
    panels.append(add_label(gt, "Ground Truth"))
    
    combined = hstack_images(panels, gap=6)
    
    out_name = f"visual_comparison_{os.path.splitext(cand)[0]}.pdf"
    out_path = os.path.join(OUT_DIR, out_name)
    Image.fromarray(combined).save(out_path)
    print(f"Saved: {out_path}")

print("\nAll best-case comparison figures generated.")