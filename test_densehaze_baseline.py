"""Dense-Haze baseline-only test (FFA-Net, no depth)."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from models.ffanet import FFA_Baseline_Net

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HAZY_DIR = os.path.join(BASE_DIR, "data", "Dense_Haze", "hazy")
GT_DIR   = os.path.join(BASE_DIR, "data", "Dense_Haze", "GT")
CKPT = os.path.join(BASE_DIR, "checkpoints_ots", "ffa_baseline_ots_epoch_100.pth")

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CKPT}")

# Load baseline model
model = FFA_Baseline_Net(gps=3, blocks=3, channels=64).to(DEVICE).eval()
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    ckpt = ckpt["model_state_dict"]
model.load_state_dict(ckpt)
print("Baseline model loaded OK")

hazy_files = sorted([f for f in os.listdir(HAZY_DIR) if f.lower().endswith('.png')])
print(f"Found {len(hazy_files)} hazy images")

psnr_list, ssim_list = [], []
valid = 0

for fn in tqdm(hazy_files, desc="Dense-Haze Baseline"):
    hazy_path = os.path.join(HAZY_DIR, fn)
    gt_name = fn.replace("_hazy", "_GT")
    gt_path = os.path.join(GT_DIR, gt_name)
    if not os.path.exists(gt_path):
        continue

    hazy_pil = Image.open(hazy_path).convert('RGB')
    gt_pil   = Image.open(gt_path).convert('RGB')

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

    with torch.no_grad():
        out = model(hazy_t)

    out_np = (out.clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    try:
        p = psnr(gt_np, out_np, data_range=255)
        s = ssim(gt_np, out_np, data_range=255, channel_axis=2)
    except:
        p, s = 0, 0
    psnr_list.append(p)
    ssim_list.append(s)
    valid += 1

avg_p = np.mean(psnr_list)
avg_s = np.mean(ssim_list)

print(f"\nValid samples: {valid}")
print(f"Baseline (FFA-Net) on Dense-Haze:")
print(f"  PSNR: {avg_p:.2f} dB")
print(f"  SSIM: {avg_s:.4f}")

# Also compute per-image max/min for variance sense
print(f"  PSNR range: [{np.min(psnr_list):.2f}, {np.max(psnr_list):.2f}]")
print(f"  SSIM range: [{np.min(ssim_list):.4f}, {np.max(ssim_list):.4f}]")
print("Done.")