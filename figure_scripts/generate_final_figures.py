"""
一次性生成所有论文最终对比图：
1. SOTS-Outdoor 3张 (1909, 0329, 0215) — 合并为一个多行图
2. O-HAZE Best SSIM Top3 + NH-HAZE Best SSIM Top3 — 生成对比面板
"""
import os, numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

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

def row(imgs, gap=4):
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

# ========== SOTS-Outdoor 3 selected images ==========
print("="*60)
print("Generating SOTS-Outdoor 3-image panel...")
base_dir = os.path.join(BASE, "results", "SOTS_outdoor_Baseline_epoch120")
ours_dir = os.path.join(BASE, "results", "sots_outdoor_e2e_epoch120")
gt_dir   = os.path.join(BASE, "data", "RESIDE", "SOTS", "outdoor", "clear")
hazy_dir = os.path.join(BASE, "data", "RESIDE", "SOTS", "outdoor", "hazy")

sots_targets = ["1909_0.85_0.2", "0329_0.8_0.12", "0215_0.8_0.08"]
rows = []
for t in sots_targets:
    hazy_path = os.path.join(hazy_dir, f"{t}.jpg")
    base_path = os.path.join(base_dir, f"{t}_dehazed.png")
    ours_path = os.path.join(ours_dir, f"{t}_dehazed.png")
    prefix = t.split("_")[0]
    gt_path = glob.glob(os.path.join(gt_dir, f"{prefix}.*"))[0]
    
    gt  = load(gt_path)
    h,w = gt.shape[:2]
    hazy = load(hazy_path, (w,h))
    base = load(base_path, (w,h))
    ours = load(ours_path, (w,h))
    
    b_s = ssim(gt, base, data_range=255, channel_axis=2)
    o_s = ssim(gt, ours, data_range=255, channel_axis=2)
    d = o_s - b_s
    
    r = row([
        label(hazy, "Hazy Input"),
        label(base, f"FFA-Net Baseline (SSIM {b_s:.4f})", (255,100,100)),
        label(ours, f"Ours (SSIM {o_s:.4f}, {d:+.4f})", (100,255,100)),
        label(gt, "Ground Truth"),
    ])
    rows.append(label(r, f"({chr(97+len(rows))}) {t}.jpg  (SSIM {d:+.4f})", (100,255,100)))
    
sots_fig = vstack(rows, gap=10)
out = os.path.join(OUT, "sots_selected_3.pdf")
Image.fromarray(sots_fig).save(out)
print(f"Saved: {out}")

# ========== O-HAZE & NH-HAZE SSIM scanning ==========
for ds, ds_config in [
    ("O-HAZE", {
        "hazy": os.path.join(BASE, "data", "O-HAZE", "hazy"),
        "gt":   os.path.join(BASE, "data", "O-HAZE", "GT"),
        "base": os.path.join(BASE, "results", "O-HAZE_Baseline_epoch120"),
        "ours": os.path.join(BASE, "results", "O-HAZE_e2e_epoch120"),
        "cplir": os.path.join(BASE, "CPLIR", "results_cpl_ohaze"),
        "c2p":   os.path.join(BASE, "C2PNet", "results_ohaze"),
    }),
    ("NH-HAZE", {
        "hazy": os.path.join(BASE, "data", "NH-HAZE", "hazy"),
        "gt":   os.path.join(BASE, "data", "NH-HAZE", "GT"),
        "base": os.path.join(BASE, "results", "NH-HAZE_Baseline_epoch120"),
        "ours": os.path.join(BASE, "results", "NH-HAZE_e2e_epoch120"),
        "cplir": None, "c2p": None,
        "gt_fn": lambda n: n.replace("_hazy",""),
    }),
]:
    print(f"\n{'='*60}")
    print(f"Scanning {ds} for SSIM best cases...")
    
    hazy_dir = ds_config["hazy"]
    gt_dir   = ds_config["gt"]
    base_dir_local = ds_config["base"]
    ours_dir_local = ds_config["ours"]
    cplir_dir = ds_config.get("cplir")
    c2p_dir   = ds_config.get("c2p")
    gt_fn     = ds_config.get("gt_fn", lambda n: n)
    
    hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    results = []
    
    for fname in tqdm(hazy_files, desc=ds):
        name_noext = os.path.splitext(fname)[0]
        base_result = f"{name_noext}_dehazed.png"
        ours_result = f"{name_noext}_dehazed.png"
        
        bp = os.path.join(base_dir_local, base_result)
        op = os.path.join(ours_dir_local, ours_result)
        if not (os.path.exists(bp) and os.path.exists(op)):
            continue
        
        gt_name = gt_fn(fname)
        gt_prefix = os.path.splitext(gt_name)[0]
        gc = glob.glob(os.path.join(gt_dir, f"{gt_prefix}.*"))
        if not gc: continue
        gp = gc[0]
        
        try:
            base_img = load(bp)
            ours_img = load(op)
            gt_img   = load(gp)
        except: continue
        
        h,w = gt_img.shape[:2]
        base_img = load(bp, (w,h))
        ours_img = load(op, (w,h))
        
        try:
            b_s = ssim(gt_img, base_img, data_range=255, channel_axis=2)
            o_s = ssim(gt_img, ours_img, data_range=255, channel_axis=2)
            d = o_s - b_s
            results.append((fname, b_s, o_s, d, bp, op, gp))
        except: continue
    
    results.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\nTop 3 {ds} SSIM best cases:")
    for i,(name,b,o,d,_,_,_) in enumerate(results[:3]):
        print(f"  {i+1}. {name}: Baseline={b:.4f}, Ours={o:.4f}, Delta={d:+.4f}")
    
    # Generate panels
    rows = []
    for i in range(min(3, len(results))):
        name, b_s, o_s, d, bp, op, gp = results[i]
        hp = os.path.join(hazy_dir, name)
        
        hazy = load(hp)
        gt   = load(gp)
        h,w = gt.shape[:2]
        hazy = load(hp, (w,h))
        base = load(bp, (w,h))
        ours = load(op, (w,h))
        
        panels = [
            label(hazy, "Hazy Input"),
            label(base, f"FFA-Net Baseline (SSIM {b_s:.4f})", (255,100,100)),
        ]
        # SOTA models if available
        if cplir_dir:
            cp = os.path.join(cplir_dir, name)
            if os.path.exists(cp):
                panels.append(label(load(cp, (w,h)), "CPLIR (CVPR 2024)"))
        if c2p_dir:
            c2 = os.path.join(c2p_dir, name)
            if os.path.exists(c2):
                panels.append(label(load(c2, (w,h)), "C2PNet (CVPR 2023)"))
        
        panels.append(label(ours, f"Ours (SSIM {o_s:.4f}, {d:+.4f})", (100,200,100)))
        panels.append(label(gt, "Ground Truth"))
        
        r = row(panels, gap=6)
        rows.append(label(r, f"({chr(97+len(rows))}) {name}  (SSIM {d:+.4f})", (100,255,100)))
    
    fig = vstack(rows, gap=10)
    out = os.path.join(OUT, f"{ds.lower()}_best_ssim.pdf")
    Image.fromarray(fig).save(out)
    print(f"Saved: {out}")

print("\n" + "="*60)
print("All final figures generated!")
print(f"Output: {OUT}")
print("="*60)