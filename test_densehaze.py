"""Quick Dense-Haze benchmark with all 4 ablation variants."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import glob
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from models.ffanet import FFA_Baseline_Net, FFA_DGCCA_Net

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dense-Haze config
HAZY_DIR = os.path.join(BASE_DIR, "data", "Dense_Haze", "hazy")
GT_DIR   = os.path.join(BASE_DIR, "data", "Dense_Haze", "GT")

VARIANTS = [
    ("A-Baseline", "checkpoints_ots/ffa_baseline_ots_epoch_100.pth", "baseline"),
    ("B-+DG-CCA",  "checkpoints_ots/ffa_dgcca_ots_epoch_100.pth", "dgcca"),
    ("C-+DAV2",    "checkpoints_ots/ffa_dgcca_ots_dav2_epoch_100.pth", "dgcca"),
    ("D-+SkyTV",   "checkpoints_ots/ffa_dgcca_ots_dav2_epoch_120.pth", "dgcca"),
]

print(f"Device: {DEVICE}")
print(f"Hazy dir: {HAZY_DIR}")
print(f"GT dir:   {GT_DIR}")

# Load DAV2 depth estimator once
print("Loading DepthAnything V2...")
depth_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
).to(DEVICE).eval()
print("DAV2 OK")

def get_depth(img_pil, h, w):
    inputs = depth_proc(images=img_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        d = depth_model(**inputs).predicted_depth
    d = torch.nn.functional.interpolate(d.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False).squeeze()
    dmin, dmax = d.min(), d.max()
    return (d - dmin) / (dmax - dmin + 1e-8)

# Load all models
models = {}
for name, ckpt_rel, mtype in VARIANTS:
    print(f"Loading {name}...")
    if mtype == "baseline":
        m = FFA_Baseline_Net(gps=3, blocks=3, channels=64).to(DEVICE).eval()
    else:
        m = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(DEVICE).eval()
    ckpt = torch.load(os.path.join(BASE_DIR, ckpt_rel), map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    m.load_state_dict(ckpt)
    models[name] = (m, mtype)
    print(f"  {name} OK")

# Gather hazy files
hazy_files = sorted([f for f in os.listdir(HAZY_DIR) if f.lower().endswith('.png')])
print(f"Found {len(hazy_files)} hazy images")

results = {name: {"psnr": [], "ssim": []} for name, _, _ in VARIANTS}
valid = 0

for fn in tqdm(hazy_files, desc="Dense-Haze"):
    hazy_path = os.path.join(HAZY_DIR, fn)
    gt_name = fn.replace("_hazy", "_GT")
    gt_path = os.path.join(GT_DIR, gt_name)
    if not os.path.exists(gt_path):
        continue

    hazy_pil = Image.open(hazy_path).convert('RGB')
    gt_pil = Image.open(gt_path).convert('RGB')

    w, h = hazy_pil.size
    max_side = 1024
    if max(w, h) > max_side:
        r = max_side / float(max(w, h))
        w, h = int(w * r), int(h * r)
    nw, nh = (w // 16) * 16, (h // 16) * 16
    hazy_pil = hazy_pil.resize((nw, nh), Image.LANCZOS)
    gt_pil   = gt_pil.resize((nw, nh), Image.LANCZOS)
    gt_np = np.array(gt_pil)

    hazy_t = TF.to_tensor(hazy_pil).unsqueeze(0).to(DEVICE)
    depth_map = get_depth(hazy_pil, nh, nw).unsqueeze(0).unsqueeze(0).to(DEVICE)

    for name, mtype in [(n, t) for n, _, t in VARIANTS]:
        model = models[name][0]
        with torch.no_grad():
            if mtype == "baseline":
                out = model(hazy_t)
            else:
                out = model(hazy_t, depth_map)
        out_np = (out.clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        try:
            p = psnr(gt_np, out_np, data_range=255)
            s = ssim(gt_np, out_np, data_range=255, channel_axis=2)
        except:
            p, s = 0, 0
        results[name]["psnr"].append(p)
        results[name]["ssim"].append(s)

    valid += 1
    if valid % 10 == 0:
        torch.cuda.empty_cache()

print(f"\nValid samples: {valid}")
print("=" * 70)
print(f"{'Variant':<14} {'PSNR (dB)':>10} {'SSIM':>10}")
print("-" * 70)
for name, _, _ in VARIANTS:
    avg_p = np.mean(results[name]["psnr"])
    avg_s = np.mean(results[name]["ssim"])
    print(f"{name:<14} {avg_p:>10.2f} {avg_s:>10.4f}")
print("=" * 70)
print("Dense-Haze benchmark complete!")