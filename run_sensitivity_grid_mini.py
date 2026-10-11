"""
敏感性网格测试 V4（迷你数据集版）：4 τ × 4 λ = 16 组合
使用 OTS 10% 迷你数据集，冻结 FFA 骨干，仅训练 DG-CCA + post_conv + depth_pre_conv
每个组合：加载 epoch_100 checkpoint → 用对应 τ/λ 微调 20 epochs → 评估 5 数据集
结果写入 sensitivity_grid_mini.json
"""
import os, torch, itertools, json, time, argparse, glob
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ========== 可选命令行指定范围（用于分片运行）==========
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="起始组合索引 (0-15)")
parser.add_argument("--end", type=int, default=16, help="结束组合索引 (exclusive)")
args = parser.parse_args()

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

# ========== 参数网格 ==========
TAUS    = [0.05, 0.10, 0.15, 0.20]
LAMBDAS = [0.005, 0.01, 0.02, 0.05]
CHECKPOINT_EPOCH = 100
NUM_FT_EPOCHS = 20
CHECKPOINT_PATH = os.path.join(BASE, "checkpoints_ots", f"ffa_dgcca_ots_dav2_epoch_{CHECKPOINT_EPOCH}.pth")
RESULTS_JSON = os.path.join(BASE, "sensitivity_grid_mini.json")

# 使用迷你数据集
TRAIN_HAZY  = os.path.join(BASE, "data", "RESIDE", "OTS_MINI", "hazy")
TRAIN_CLEAR = os.path.join(BASE, "data", "RESIDE", "OTS_MINI", "clear")
TRAIN_DEPTH = os.path.join(BASE, "data", "RESIDE", "OTS_MINI", "depth_dav2")

BATCH_SIZE = 4
ACCUM_STEPS = 4          # effective batch = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4

DATASETS = {
    "SOTS-Outdoor": (os.path.join(BASE, "data", "sots", "outdoor", "hazy"),
                     os.path.join(BASE, "data", "sots", "outdoor", "clear")),
    "SOTS-Indoor":  (os.path.join(BASE, "data", "sots", "indoor", "hazy"),
                     os.path.join(BASE, "data", "sots", "indoor", "clear")),
    "O-HAZE":       (os.path.join(BASE, "data", "O-HAZE", "hazy"),
                     os.path.join(BASE, "data", "O-HAZE", "GT")),
    "NH-HAZE":      (os.path.join(BASE, "data", "NH-HAZE", "hazy"),
                     os.path.join(BASE, "data", "NH-HAZE", "GT")),
    "I-HAZE":       (os.path.join(BASE, "data", "I-HAZE", "hazy"),
                     os.path.join(BASE, "data", "I-HAZE", "GT")),
}

# ========== 数据集适配 ==========
from utils.dataset_ots import OTSDataset

# ========== 单个评估函数 ==========
def evaluate_model(model, tag):
    model.eval()
    results = {}
    for ds_name, (hazy_dir, gt_dir) in DATASETS.items():
        if not os.path.exists(hazy_dir):
            results[ds_name] = {"psnr": None, "ssim": None}
            continue
        hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
        psnr_list, ssim_list = [], []
        for fname in tqdm(hazy_files[:30], desc=f"[{tag}] Eval {ds_name}", leave=False):
            hp = os.path.join(hazy_dir, fname)
            try:
                hazy_img = Image.open(hp).convert("RGB")
            except:
                continue
            w0, h0 = hazy_img.size
            # 限制最大边 1024 防止 OOM
            max_size = 1024
            if max(w0, h0) > max_size:
                scale = max_size / max(w0, h0)
                w0, h0 = int(w0 * scale), int(h0 * scale)
                hazy_img = hazy_img.resize((w0, h0), Image.LANCZOS)
            nw, nh = (w0 // 16) * 16, (h0 // 16) * 16
            hazy_pil = hazy_img.resize((nw, nh), Image.LANCZOS)
            gt_prefix = os.path.splitext(fname)[0]
            gt_candidates = glob.glob(os.path.join(gt_dir, f"{gt_prefix}.*"))
            if not gt_candidates:
                alt = gt_prefix.replace("_hazy", "")
                gt_candidates = glob.glob(os.path.join(gt_dir, f"{alt}.*"))
            if not gt_candidates:
                continue
            gt_pil = Image.open(gt_candidates[0]).convert("RGB").resize((nw, nh), Image.LANCZOS)
            gt_np = np.array(gt_pil)

            # 创建假 depth（全 0.5 中等深度）
            depth_np = np.ones((nh, nw), dtype=np.float32) * 0.5
            depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

            hazy_t = TF.to_tensor(hazy_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(hazy_t, depth_t) if hasattr(model, 'forward') and 'depth' in model.forward.__code__.co_varnames else model(hazy_t)
                out_np = out.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            
            out_uint8 = (out_np * 255).clip(0, 255).astype(np.uint8)
            gt_uint8 = gt_np.astype(np.uint8)
            
            try:
                psnr_list.append(psnr(gt_uint8, out_uint8, data_range=255))
                ssim_list.append(ssim(gt_uint8, out_uint8, data_range=255, channel_axis=2, win_size=11))
            except:
                continue
        
        results[ds_name] = {
            "psnr": float(np.mean(psnr_list)) if psnr_list else None,
            "ssim": float(np.mean(ssim_list)) if ssim_list else None,
        }
    return results

# ========== 单个 τ/λ 组合训练 ==========
def run_one_combo(tau, lam, combo_idx):
    tag = f"tau{tau:.2f}_lam{lam:.4f}"
    print(f"\n{'='*60}")
    print(f"  [{combo_idx+1}/16] τ={tau}, λ={lam}  ({tag})")
    print(f"{'='*60}")

    # 1. 加载模型
    from models.ffanet import FFA_DGCCA_Net
    model = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    # 2. 冻结 FFA 骨干 (g1, g2, g3) 和 rgb_pre_conv
    for name, param in model.named_parameters():
        if any(k in name for k in ["g1.", "g2.", "g3.", "rgb_pre_conv."]):
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # 3. 数据集 & DataLoader
    train_ds = OTSDataset(TRAIN_HAZY, TRAIN_CLEAR, TRAIN_DEPTH, crop_size=256, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 4. 优化器（只优化 requires_grad=True 的参数）
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    num_steps_per_epoch = len(train_loader) // ACCUM_STEPS
    total_steps = num_steps_per_epoch * NUM_FT_EPOCHS
    sched = CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=LR * 0.1)

    # 5. Loss（带 TV 的混合 loss）
    #    参数映射: τ → sky_threshold (深度阈值), λ → tv_weight (TV 正则化系数)
    from utils.loss_v2_sky_tv import CompositeLoss_SkyTV
    criterion = CompositeLoss_SkyTV(edge_weight=0.05, tv_weight=lam, sky_threshold=tau).to(DEVICE)

    # 6. 训练循环
    all_results = {tag: {}}
    best_psnr_sots = -1
    best_ssim_sots = -1
    opt.zero_grad()
    nan_alert = False

    for epoch in range(CHECKPOINT_EPOCH + 1, CHECKPOINT_EPOCH + NUM_FT_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[{tag}] Epoch {epoch}/{CHECKPOINT_EPOCH+NUM_FT_EPOCHS}", leave=False)
        
        for i, (hazy, clean, depth) in pbar:
            hazy, clean, depth = hazy.to(DEVICE), clean.to(DEVICE), depth.to(DEVICE)
            out = model(hazy, depth)
            loss = criterion(out, clean, depth)
            loss = loss / ACCUM_STEPS
            if torch.isnan(loss) or torch.isinf(loss):
                if not nan_alert:
                    print(f"\n  ⚠️ [{tag}] NaN/inf loss! epoch {epoch}, step {i}")
                    nan_alert = True
                opt.zero_grad()
                continue
            loss.backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            
            epoch_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({"Loss": f"{loss.item()*ACCUM_STEPS:.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        
        # 每个 epoch 结束后评估
        eval_results = evaluate_model(model, f"{tag}_e{epoch}")
        sots_p = eval_results.get("SOTS-Outdoor", {}).get("psnr", 0) or 0
        sots_s = eval_results.get("SOTS-Outdoor", {}).get("ssim", 0) or 0
        print(f"  [{tag}] Epoch {epoch} | Loss={avg_loss:.4f} | SOTS PSNR={sots_p:.2f} SSIM={sots_s:.4f}")
        
        all_results[tag][f"epoch_{epoch}"] = {
            "loss": float(avg_loss),
            "eval": eval_results,
        }
        
        if sots_p + sots_s > best_psnr_sots + best_ssim_sots:
            best_psnr_sots = sots_p
            best_ssim_sots = sots_s

    all_results[tag]["best"] = {"SOTS_PSNR": best_psnr_sots, "SOTS_SSIM": best_ssim_sots}
    return all_results

# ========== 主程序 ==========
if __name__ == "__main__":
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    
    # 生成所有组合
    combos = list(itertools.product(TAUS, LAMBDAS))
    print(f"Total combos: {len(combos)} (running indices {args.start} to {args.end-1})")
    
    # 加载已有结果（支持断点续跑）
    all_results = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r") as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {RESULTS_JSON}")
    
    for idx in range(args.start, min(args.end, len(combos))):
        tau, lam = combos[idx]
        tag = f"tau{tau:.2f}_lam{lam:.4f}"
        
        if tag in all_results and "best" in all_results[tag]:
            print(f"  [{idx+1}/16] {tag} already done, skipping...")
            continue
        
        combo_results = run_one_combo(tau, lam, idx)
        all_results.update(combo_results)
        
        # 每个组合完成后保存
        with open(RESULTS_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved results to {RESULTS_JSON}")
    
    print(f"\n{'='*60}")
    print("  All done! Results in:", RESULTS_JSON)
    print(f"{'='*60}")