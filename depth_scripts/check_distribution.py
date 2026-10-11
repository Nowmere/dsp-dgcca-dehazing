import torch
import torch.nn as nn
from test_correct_block import OfficialFFANetStandard
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import os, glob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load('ots_train_ffa_3_19.pk', map_location=DEVICE)
sd = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}

model_std = OfficialFFANetStandard().to(DEVICE).eval()
model_std.load_state_dict(sd)

# 我们来网格搜索各种不同的输入尺度和偏移，看看它们的前向输出特征的值分布。
# 比如如果官方输入是 [0, 1]，而输出也是 [0, 1]，但是在某些尺度下更符合图像分布。
# 真实去雾图的均值一般和输入 hazy 的均值差不多或稍微暗一点。
# 让我们试着打印一些值看看。
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
gt_np = np.array(gt_pil).astype(np.float32) / 255.0
tensor = TF.to_tensor(hazy_pil).unsqueeze(0).to(DEVICE)

print("Hazy input mean:", tensor.mean().item(), "min/max:", tensor.min().item(), tensor.max().item())
print("GT clear mean:", gt_np.mean(), "min/max:", gt_np.min(), gt_np.max())

with torch.no_grad():
    # 假设无归一化前向：
    out_direct = model_std(tensor)
    print("Direct model output mean/min/max:", out_direct.mean().item(), out_direct.min().item(), out_direct.max().item())
    
    # 假设 x - 0.5 归一化：
    out_minus_05 = model_std(tensor - 0.5) + 0.5
    print("x-0.5 model output mean/min/max:", out_minus_05.mean().item(), out_minus_05.min().item(), out_minus_05.max().item())
    
    # 假设 (x - 0.5)/0.5 归一化：
    out_norm_05 = model_std((tensor - 0.5) / 0.5) * 0.5 + 0.5
    print("(x-0.5)/0.5 model output mean/min/max:", out_norm_05.mean().item(), out_norm_05.min().item(), out_norm_05.max().item())


