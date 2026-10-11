import os, re, torch, glob, numpy as np, cv2
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from PIL import Image
from torchvision.transforms import functional as TF
from collections import OrderedDict
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models.ffanet import FFA_Baseline_Net, FFA_DGCCA_Net

# ========== 1. 分析官方架构 ==========
ckpt = torch.load('ots_train_ffa_3_19.pk', map_location='cpu')
official_keys = list(ckpt['model'].keys())

g1_blocks = set()
g2_blocks = set()
g3_blocks = set()
for k in official_keys:
    m = re.match(r'module\.g1\.gp\.(\d+)', k)
    if m: g1_blocks.add(int(m.group(1)))
    m = re.match(r'module\.g2\.gp\.(\d+)', k)
    if m: g2_blocks.add(int(m.group(1)))
    m = re.match(r'module\.g3\.gp\.(\d+)', k)
    if m: g3_blocks.add(int(m.group(1)))

num_blocks = len(g1_blocks)
reduction = 64 // ckpt['model']['module.g1.gp.0.calayer.ca.0.weight'].shape[0]
print(f"官方架构: gps=3, blocks={num_blocks}, CA reduction={reduction}")

# ========== 2. 直接使用 state_dict 转换后加载 ==========
converted_sd = OrderedDict()
for k, v in ckpt['model'].items():
    key = k
    if key.startswith('module.'):
        key = key[7:]
    key = re.sub(r'\.gp\.(\d+)', r'.\1', key)
    key = key.replace('g1.19', 'g1_att')
    key = key.replace('g2.19', 'g2_att')
    key = key.replace('g3.19', 'g3_att')
    converted_sd[key] = v

print(f"转换后参数数: {len(converted_sd)}")
print("样本keys:", list(converted_sd.keys())[:5], "...", list(converted_sd.keys())[-5:])

# ========== 3. 构建完全匹配官方架构的模型 ==========
class FABlock(nn.Module):
    def __init__(self, c, r):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.ca = nn.Sequential(nn.Conv2d(c, c//r, 1, bias=True), nn.ReLU(True), nn.Conv2d(c//r, c, 1, bias=True), nn.Sigmoid())
        self.pa = nn.Sequential(nn.Conv2d(c, c//8, 1, bias=True), nn.ReLU(True), nn.Conv2d(c//8, 1, 1, bias=True), nn.Sigmoid())
    def forward(self, x):
        r = self.conv1(x); r = self.relu(r); r = self.conv2(r)
        r = self.ca(r) * r; r = self.pa(r) * r
        return r + x

# l1, l2, l3 = lower convs for each group; ca = concat attention
class OfficialFFANet(nn.Module):
    def __init__(self):
        super().__init__()
        c, r = 64, 8
        self.pre = nn.Sequential(nn.Conv2d(3, c, 3, 1, 1, bias=True))
        self.g1 = nn.ModuleList([FABlock(c, r) for _ in range(19)])
        self.g1_att = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.g2 = nn.ModuleList([FABlock(c, r) for _ in range(19)])
        self.g2_att = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.g3 = nn.ModuleList([FABlock(c, r) for _ in range(19)])
        self.g3_att = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(3*c, c//r, 1, bias=True), nn.ReLU(True), nn.Conv2d(c//r, 3*c, 1, bias=True), nn.Sigmoid())
        self.pa = nn.Sequential(nn.Conv2d(3*c, 3*c//8, 1, bias=True), nn.ReLU(True), nn.Conv2d(3*c//8, 3, 1, bias=True), nn.Sigmoid())
        self.post = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=True), nn.ReLU(True), nn.Conv2d(c, 3, 3, 1, 1, bias=True))
    def forward(self, rgb):
        x = self.pre(rgb)
        g1 = x
        for b in self.g1: g1 = b(g1)
        g1 = self.g1_att(g1) + x
        g2 = g1
        for b in self.g2: g2 = b(g2)
        g2 = self.g2_att(g2) + g1
        g3 = g2
        for b in self.g3: g3 = b(g3)
        g3 = self.g3_att(g3) + g2
        cat = torch.cat([g1, g2, g3], 1)
        w = self.ca(cat) * self.pa(cat)
        w = w.unsqueeze(-1).unsqueeze(-1)
        f = g1 * w[:,0:1] + g2 * w[:,1:2] + g3 * w[:,2:3]
        return self.post(f) + rgb

# ========== 4. 加载模型 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎯 硬件: {DEVICE}")

model = OfficialFFANet().to(DEVICE).eval()
# Check what keys our model expects
model_keys = list(model.state_dict().keys())
print(f"模型期望参数数: {len(model_keys)}")
print("模型keys前5:", model_keys[:5])
print("模型keys后5:", model_keys[-5:])

# Filter: only keep keys that match
usable = OrderedDict()
for k, v in converted_sd.items():
    if k in model_keys:
        usable[k] = v
    else:
        # Try matching nested Sequential names
        # e.g., g1.0.pa.0.weight should match g1.0.pa.0.weight
        # Check if key is already correct
        if k in model.state_dict():
            usable[k] = v

print(f"可匹配参数: {len(usable)}/{len(converted_sd)}")
missing = [k for k in model_keys if k not in usable]
print(f"缺失参数: {len(missing)}")
if missing:
    print("示例缺失:", missing[:5])
extra = [k for k in usable if k not in model_keys]
if extra:
    print("多余参数:", extra[:5])

model.load_state_dict(usable, strict=False)

# 加载我们自己的模型
base_net = FFA_Baseline_Net(gps=3, blocks=3, channels=64).to(DEVICE).eval()
base_net.load_state_dict(torch.load("./checkpoints_ots/ffa_baseline_ots_epoch_100.pth", map_location=DEVICE))

ours_net = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(DEVICE).eval()
ckpt_ours = torch.load("./checkpoints_ots/ffa_dgcca_ots_dav2_epoch_120.pth", map_location=DEVICE)
ours_net.load_state_dict(ckpt_ours["model_state_dict"] if "model_state_dict" in ckpt_ours else ckpt_ours)

print(f"官方模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# 测试三个数据集
for DATASET_NAME in ["O-HAZE", "NH-HAZE", "I-HAZE"]:
    DATA_ROOT = f"./data/{DATASET_NAME}"
    SAVE_DIR = f"./results/{DATASET_NAME}_Comparison_Official"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(DEVICE).eval()
    
    print(f"\n🚀 测试: {DATASET_NAME}")
    
    def get_depth(pil, h, w):
        inp = processor(images=pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            d = depth_model(**inp).predicted_depth
        d = nn.functional.interpolate(d.unsqueeze(1), (h,w), mode="bicubic").squeeze()
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        return d.unsqueeze(0).unsqueeze(0)
    
    hazy_list = sorted(os.listdir(os.path.join(DATA_ROOT, "hazy")))
    metrics = {"base_p":[],"base_s":[],"off_p":[],"off_s":[],"ours_p":[],"ours_s":[]}
    
    for fn in hazy_list:
        if not fn.endswith(('.png','.jpg','.jpeg')): continue
        prefix = fn.split('_')[0]
        gt_candidates = glob.glob(os.path.join(DATA_ROOT, "GT", f"{prefix}*.*"))
        if not gt_candidates: continue
        hazy = Image.open(os.path.join(DATA_ROOT, "hazy", fn)).convert('RGB')
        gt = Image.open(gt_candidates[0]).convert('RGB')
        w,h = hazy.size
        if max(w,h) > 1024:
            r = 1024/max(w,h); w,h = int(w*r), int(h*r)
        nw, nh = (w//16)*16, (h//16)*16
        hazy = hazy.resize((nw,nh), Image.LANCZOS)
        gt = gt.resize((nw,nh), Image.LANCZOS)
        gt_np = np.array(gt)
        depth = get_depth(hazy, nh, nw)
        tensor = TF.to_tensor(hazy).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            b = base_net(tensor)
            o = model(tensor)
            u = ours_net(tensor, depth)
        
        def to_np(t): return (t.clamp(0,1).squeeze().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        rb, ro, ru = to_np(b), to_np(o), to_np(u)
        
        metrics["base_p"].append(psnr(gt_np, rb)); metrics["base_s"].append(ssim(gt_np, rb, channel_axis=2))
        metrics["off_p"].append(psnr(gt_np, ro)); metrics["off_s"].append(ssim(gt_np, ro, channel_axis=2))
        metrics["ours_p"].append(psnr(gt_np, ru)); metrics["ours_s"].append(ssim(gt_np, ru, channel_axis=2))
        
        cv2.imwrite(os.path.join(SAVE_DIR, fn), cv2.cvtColor(np.hstack([np.array(hazy),rb,ro,ru,gt_np]), cv2.COLOR_RGB2BGR))
        print(f"  {fn} | Off:{metrics['off_p'][-1]:.2f} | Ours:{metrics['ours_p'][-1]:.2f}")
        torch.cuda.empty_cache()
    
    print(f"\n📊 {DATASET_NAME}")
    print(f"  Baseline(我们100轮) -> PSNR:{np.mean(metrics['base_p']):.4f}, SSIM:{np.mean(metrics['base_s']):.4f}")
    print(f"  Official(33.57dB)   -> PSNR:{np.mean(metrics['off_p']):.4f}, SSIM:{np.mean(metrics['off_s']):.4f}")
    print(f"  Ours(DAV2+TV 120轮) -> PSNR:{np.mean(metrics['ours_p']):.4f}, SSIM:{np.mean(metrics['ours_s']):.4f}")