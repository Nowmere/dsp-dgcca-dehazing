"""
run_restormer_test.py
批量运行 Restormer (Deraining 权重) 在 O-HAZE / NH-HAZE / I-HAZE 上的推理
"""
import os, sys
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

BASE = os.path.dirname(os.path.abspath(__file__))
DEHAZE_ENV = os.path.join(BASE, "DehazeFormer", "DehazeFormer")

# ========== Load Restormer ==========
print("Loading Restormer Deraining model...")
sys.path.insert(0, os.path.join(BASE, "Restormer"))
import torch
import torch.nn.functional as F
from runpy import run_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture
arch_path = os.path.join(BASE, "Restormer", "basicsr", "models", "archs", "restormer_arch.py")
load_arch = run_path(arch_path)
parameters = {
    "inp_channels": 3, "out_channels": 3, "dim": 48,
    "num_blocks": [4, 6, 6, 8], "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8], "ffn_expansion_factor": 2.66,
    "bias": False, "LayerNorm_type": "WithBias", "dual_pixel_task": False
}
model = load_arch["Restormer"](**parameters)

# Weights
weights_path = os.path.join(BASE, "Restormer", "Deraining", "pretrained_models", "deraining.pth")
if not os.path.exists(weights_path):
    # Try downloading if not present
    print(f"ERROR: weights not found at {weights_path}")
    print("Download from: https://github.com/swz30/Restormer/releases")
    sys.exit(1)

ckpt = torch.load(weights_path, map_location=device)
model.load_state_dict(ckpt["params"])
model.to(device).eval()

img_multiple_of = 8

def restormer_infer(img_np, tile=1024, tile_overlap=128):
    """img_np: (H, W, 3) uint8 -> (H, W, 3) uint8. Uses tile-based inference for large images."""
    input_ = torch.from_numpy(img_np).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, h, w = input_.shape

    if tile is None:
        H = ((h + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
        W = ((w + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
        padh, padw = H - h, W - w
        input_ = F.pad(input_, (0, padw, 0, padh), "reflect")
        with torch.no_grad():
            restored = model(input_)
        restored = torch.clamp(restored, 0, 1)
        restored = restored[:, :, :h, :w]
        restored = restored.permute(0, 2, 3, 1).cpu().numpy()[0]
        return (restored * 255).astype(np.uint8)

    # Tile-based inference
    b, c, H, W = input_.shape
    tile = min(tile, H, W)
    stride = tile - tile_overlap
    h_idx_list = list(range(0, H - tile, stride)) + [H - tile]
    w_idx_list = list(range(0, W - tile, stride)) + [W - tile]
    E = torch.zeros(b, c, H, W).type_as(input_)
    W_sum = torch.zeros_like(E)

    with torch.no_grad():
        for hi in h_idx_list:
            for wi in w_idx_list:
                patch = input_[..., hi:hi + tile, wi:wi + tile]
                # Pad to multiple of img_multiple_of
                ph, pw = patch.shape[2], patch.shape[3]
                PH = ((ph + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
                PW = ((pw + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
                patch = F.pad(patch, (0, PW - pw, 0, PH - ph), "reflect")
                out_patch = model(patch)
                out_patch = out_patch[:, :, :ph, :pw]
                E[..., hi:hi + tile, wi:wi + tile] += out_patch
                W_sum[..., hi:hi + tile, wi:wi + tile] += 1.0

    restored = E / W_sum
    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]
    restored = restored.permute(0, 2, 3, 1).cpu().numpy()[0]
    return (restored * 255).astype(np.uint8)

# ========== Datasets ==========
DATASETS = {
    "O-HAZE": {
        "hazy": os.path.join(BASE, "data", "O-HAZE", "hazy"),
        "gt":   os.path.join(BASE, "data", "O-HAZE", "GT"),
        "gt_fn": lambda n: n,
    },
    "NH-HAZE": {
        "hazy": os.path.join(BASE, "data", "NH-HAZE", "hazy"),
        "gt":   os.path.join(BASE, "data", "NH-HAZE", "GT"),
        "gt_fn": lambda n: n.replace("_hazy", ""),
    },
    "I-HAZE": {
        "hazy": os.path.join(BASE, "data", "I-HAZE", "hazy"),
        "gt":   os.path.join(BASE, "data", "I-HAZE", "GT"),
        "gt_fn": lambda n: n.replace("_hazy", ""),
    },
}

# ========== Run ==========
all_results = {}

for ds_name, cfg in DATASETS.items():
    hazy_dir = cfg["hazy"]
    gt_dir = cfg["gt"]
    gt_fn = cfg["gt_fn"]
    out_dir = os.path.join(BASE, "results", f"Restormer_{ds_name}")
    os.makedirs(out_dir, exist_ok=True)

    hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])

    psnr_list, ssim_list = [], []
    valid = 0

    print(f"\n{'='*50}")
    print(f"Dataset: {ds_name} ({len(hazy_files)} images)")
    print(f"{'='*50}")

    for fname in tqdm(hazy_files, desc=ds_name):
        hp = os.path.join(hazy_dir, fname)
        try:
            hazy_np = np.array(Image.open(hp).convert("RGB"))
        except:
            continue

        # Inference
        out_np = restormer_infer(hazy_np)

        # Save
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(out_dir, out_name)
        Image.fromarray(out_np).save(out_path)

        # GT
        gt_name = gt_fn(fname)
        gt_prefix = os.path.splitext(gt_name)[0]
        gc = glob.glob(os.path.join(gt_dir, f"{gt_prefix}.*"))
        if not gc:
            continue
        gp = gc[0]
        try:
            gt_np = np.array(Image.open(gp).convert("RGB"))
        except:
            continue

        h, w = gt_np.shape[:2]
        out_np_resized = np.array(Image.fromarray(out_np).resize((w, h), Image.LANCZOS))

        try:
            p = psnr(gt_np, out_np_resized, data_range=255)
            s = ssim(gt_np, out_np_resized, data_range=255, channel_axis=2)
            psnr_list.append(p)
            ssim_list.append(s)
            valid += 1
        except:
            pass

        torch.cuda.empty_cache()

    avg_p = np.mean(psnr_list) if psnr_list else 0
    avg_s = np.mean(ssim_list) if ssim_list else 0
    all_results[ds_name] = (avg_p, avg_s, valid)
    print(f"  Valid: {valid}, PSNR: {avg_p:.2f}, SSIM: {avg_s:.4f}")

# ========== Summary ==========
print(f"\n{'='*60}")
print(f"Restormer (Deraining) Real-World Results")
print(f"{'='*60}")
print(f"{'Dataset':<12} {'PSNR':>8} {'SSIM':>8} {'Samples':>8}")
for ds in ["O-HAZE", "NH-HAZE", "I-HAZE"]:
    if ds in all_results:
        p, s, n = all_results[ds]
        print(f"{ds:<12} {p:>8.2f} {s:>8.4f} {n:>8}")
print(f"{'='*60}")