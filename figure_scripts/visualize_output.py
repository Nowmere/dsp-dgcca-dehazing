import os
import torch
import glob
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from test_correct_block import OfficialFFANetStandard

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load('ots_train_ffa_3_19.pk', map_location=DEVICE)
sd = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}

model = OfficialFFANetStandard().to(DEVICE).eval()
model.load_state_dict(sd)

# Load a sample from SOTS-Outdoor
DATA_ROOT = "./data/RESIDE/SOTS/outdoor"
hazy_fn = sorted(os.listdir(os.path.join(DATA_ROOT, "hazy")))[0]
gt_candidates = glob.glob(os.path.join(DATA_ROOT, "clear", hazy_fn.split('_')[0] + "*.*"))
gt_fn = gt_candidates[0]

hazy_pil = Image.open(os.path.join(DATA_ROOT, "hazy", hazy_fn)).convert('RGB')
gt_pil = Image.open(gt_fn).convert('RGB')

w, h = hazy_pil.size
nw, nh = (w // 16) * 16, (h // 16) * 16
hazy_pil = hazy_pil.resize((nw, nh), Image.LANCZOS)
gt_pil = gt_pil.resize((nw, nh), Image.LANCZOS)

tensor = TF.to_tensor(hazy_pil).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    out = model(tensor)

print("Hazy tensor min/max:", tensor.min().item(), tensor.max().item())
print("Model output min/max/mean:", out.min().item(), out.max().item(), out.mean().item())

# Save comparison image
out_np = torch.clamp(out, 0, 1).squeeze().cpu().numpy().transpose(1, 2, 0)
hazy_np = np.array(hazy_pil)
gt_np = np.array(gt_pil)

print("Output image min/max:", out_np.min(), out_np.max())

# Let's see if the output looks like the dehazed image or something else
# We can save it as debug_output.png
Image.fromarray((out_np * 255).astype(np.uint8)).save("debug_output.png")
print("Saved debug_output.png")
