"""
select_best_cases.py
Scans existing NH-HAZE and O-HAZE output directories to find images where
Ours (D-SkyTV) outperforms Baseline by the largest PSNR margin.
"""
import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def compute_per_image_psnr(baseline_dir, ours_dir, gt_dir, hazy_dir, ds_name, gt_fn=None):
    """Compute per-image PSNR and return sorted list of (name, baseline_psnr, ours_psnr, delta)"""
    results = []
    hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    
    for fname in tqdm(hazy_files, desc=ds_name):
        # Baseline output (actual naming: nh10_hazy_dehazed.png)
        name_noext = os.path.splitext(fname)[0]
        base_name = f"{name_noext}_dehazed.png"
        base_path = os.path.join(baseline_dir, base_name)
        if not os.path.exists(base_path):
            base_name = fname  # fallback to exact match
            base_path = os.path.join(baseline_dir, base_name)
            if not os.path.exists(base_path):
                continue
            
        # Ours output
        ours_name = f"{name_noext}_dehazed.png"
        ours_path = os.path.join(ours_dir, ours_name)
        if not os.path.exists(ours_path):
            ours_name = fname
            ours_path = os.path.join(ours_dir, ours_name)
            if not os.path.exists(ours_path):
                continue
        
        # GT
        if gt_fn:
            gt_name = gt_fn(fname)
        else:
            gt_name = fname
        gt_name_noext = os.path.splitext(gt_name)[0]
        
        import glob
        candidates = glob.glob(os.path.join(gt_dir, f"{gt_name_noext}.*"))
        if not candidates:
            continue
        gt_path = candidates[0]
        
        try:
            base_img = np.array(Image.open(base_path).convert("RGB"))
            ours_img = np.array(Image.open(ours_path).convert("RGB"))
            gt_img = np.array(Image.open(gt_path).convert("RGB"))
        except Exception:
            continue
        
        # Resize to match GT
        h, w = gt_img.shape[:2]
        if base_img.shape[:2] != (h, w):
            base_img = np.array(Image.fromarray(base_img).resize((w, h), Image.LANCZOS))
        if ours_img.shape[:2] != (h, w):
            ours_img = np.array(Image.fromarray(ours_img).resize((w, h), Image.LANCZOS))
        
        try:
            b_psnr = psnr(gt_img, base_img, data_range=255)
            o_psnr = psnr(gt_img, ours_img, data_range=255)
            delta = o_psnr - b_psnr
            results.append((fname, b_psnr, o_psnr, delta))
        except Exception:
            continue
    
    results.sort(key=lambda x: x[3], reverse=True)
    return results

# === NH-HAZE ===
print("="*70)
print("NH-HAZE: Scanning existing outputs...")
print("="*70)
nh_baseline_dir = os.path.join(BASE_DIR, "results", "NH-HAZE_Baseline_epoch120")
nh_ours_dir = os.path.join(BASE_DIR, "results", "NH-HAZE_e2e_epoch120")
nh_gt_dir = os.path.join(BASE_DIR, "data", "NH-HAZE", "GT")
nh_hazy_dir = os.path.join(BASE_DIR, "data", "NH-HAZE", "hazy")
nh_gt_fn = lambda name: name.replace("_hazy", "")

nh_results = compute_per_image_psnr(nh_baseline_dir, nh_ours_dir, nh_gt_dir, nh_hazy_dir, "NH-HAZE", nh_gt_fn)
print(f"\nTop 10 NH-HAZE images (largest Ours-Baseline PSNR delta):")
print(f"{'Rank':<6} {'Filename':<25} {'Baseline':>10} {'Ours':>10} {'Delta':>10}")
for i, (name, b, o, d) in enumerate(nh_results[:10]):
    print(f"{i+1:<6} {name:<25} {b:>10.3f} {o:>10.3f} {d:>+10.3f}")

# === O-HAZE ===
print("\n" + "="*70)
print("O-HAZE: Scanning existing outputs...")
print("="*70)
oh_baseline_dir = os.path.join(BASE_DIR, "results", "O-HAZE_Baseline_epoch120")
oh_ours_dir = os.path.join(BASE_DIR, "results", "O-HAZE_e2e_epoch120")
oh_gt_dir = os.path.join(BASE_DIR, "data", "O-HAZE", "GT")
oh_hazy_dir = os.path.join(BASE_DIR, "data", "O-HAZE", "hazy")

oh_results = compute_per_image_psnr(oh_baseline_dir, oh_ours_dir, oh_gt_dir, oh_hazy_dir, "O-HAZE")
print(f"\nTop 10 O-HAZE images (largest Ours-Baseline PSNR delta):")
print(f"{'Rank':<6} {'Filename':<25} {'Baseline':>10} {'Ours':>10} {'Delta':>10}")
for i, (name, b, o, d) in enumerate(oh_results[:10]):
    print(f"{i+1:<6} {name:<25} {b:>10.3f} {o:>10.3f} {d:>+10.3f}")

print("\n" + "="*70)
print("Done! Select the top images from each list for paper figures.")
print("="*70)