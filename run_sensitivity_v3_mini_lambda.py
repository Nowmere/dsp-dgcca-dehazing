"""
敏感性快速测试 V3-Mini-Lambda：固定 τ=0.10，扫描 λ
λ ∈ {0.001, 0.005, 0.01, 0.05}，每组合 20 epochs
使用 OTS-MINI 迷你数据集，冻结 FFA 骨干
结果写入 sensitivity_v3_mini_lambda.json
"""
import os, torch, time, json, glob
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 固定参数 ==========
TAU_FIXED = 0.10
LAMBDAS = [0.001, 0.005, 0.01, 0.05]
CHECKPOINT_EPOCH = 100
NUM_FT_EPOCHS = 20
CHECKPOINT_PATH = os.path.join(BASE, "checkpoints_ots", f"ffa_dgcca_ots_dav2_epoch_{CHECKPOINT_EPOCH}.pth")

# OTS-MINI
TRAIN_HAZY  = os.path.join(BASE, "data", "RESIDE", "OTS_MINI", "hazy")
TRAIN_CLEAR = os.path.join(BASE, "data", "RESIDE", "OTS_MINI", "clear")
TRAIN_DEPTH = os.path.join(BASE, "data", "RESIDE", "OTS_MINI", "depth_dav2")

BATCH_SIZE = 4
ACCUM_STEPS = 4
FINAL_EPOCH = CHECKPOINT_EPOCH + NUM_FT_EPOCHS
LR = 1e-4
WD = 1e-4

DATASETS = {
    "SOTS-Outdoor": (os.path.join(BASE, "data", "sots", "outdoor", "hazy"),
                     os.path.join(BASE, "data", "sots", "outdoor", "gt")),
    "SOTS-Indoor":  (os.path.join(BASE, "data", "sots", "indoor", "hazy"),
                     os.path.join(BASE, "data", "sots", "indoor", "gt")),
    "O-HAZE":       (os.path.join(BASE, "data", "O-HAZE", "hazy"),
                     os.path.join(BASE, "data", "O-HAZE", "GT")),
    "NH-HAZE":      (os.path.join(BASE, "data", "NH-HAZE", "hazy"),
                     os.path.join(BASE, "data", "NH-HAZE", "GT")),
    "I-HAZE":       (os.path.join(BASE, "data", "I-HAZE", "hazy"),
                     os.path.join(BASE, "data", "I-HAZE", "GT")),
}


def evaluate_model(model, tag):
    model.eval()
    results = {}
    for ds_name, (hazy_dir, gt_dir) in DATASETS.items():
        if not os.path.exists(hazy_dir):
            results[ds_name] = {"psnr": None, "ssim": None}
            continue
        hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        psnr_list, ssim_list = [], []
        for fname in tqdm(hazy_files[:30], desc=f"[{tag}] Eval {ds_name}", leave=False):
            hp = os.path.join(hazy_dir, fname)
            try:
                hazy_img = Image.open(hp).convert("RGB")
            except:
                continue
            w0, h0 = hazy_img.size
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
            depth_path = os.path.join(BASE, "data", "depth_cache", ds_name, f"{os.path.splitext(fname)[0]}.pt")
            if os.path.exists(depth_path):
                depth_tensor = torch.load(depth_path).unsqueeze(0).to(DEVICE)
            else:
                depth_tensor = torch.zeros(1, 1, nh, nw).to(DEVICE)
            hazy_t = TF.to_tensor(hazy_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(hazy_t, depth_tensor)
            out_np = (out.clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            try:
                p = psnr(gt_np, out_np, data_range=255)
                s = ssim(gt_np, out_np, data_range=255, channel_axis=2)
                psnr_list.append(p)
                ssim_list.append(s)
            except:
                pass
        results[ds_name] = {
            "psnr": float(np.mean(psnr_list)) if psnr_list else None,
            "ssim": float(np.mean(ssim_list)) if ssim_list else None,
        }
        if results[ds_name]["psnr"] is not None:
            print(f"  [{tag}] {ds_name}: PSNR={results[ds_name]['psnr']:.2f}, SSIM={results[ds_name]['ssim']:.4f}")
    return results


def run_one(lam):
    tag = f"lam{lam:.3f}_tau{TAU_FIXED:.2f}"
    print(f"\n{'='*60}\n  λ={lam}, τ={TAU_FIXED}  ({tag})\n{'='*60}")

    from models.ffanet import FFA_DGCCA_Net
    model = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    for name, param in model.named_parameters():
        if any(k in name for k in ["g1.", "g2.", "g3.", "rgb_pre_conv."]):
            param.requires_grad = False
        else:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    from utils.dataset_ots import OTSDataset
    dataset = OTSDataset(TRAIN_HAZY, TRAIN_CLEAR, TRAIN_DEPTH, crop_size=256, is_train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    print(f"  Dataset size: {len(dataset)} images, {len(dataloader)} batches/epoch")

    from utils.loss_v2_sky_tv import CompositeLoss_SkyTV
    criterion = CompositeLoss_SkyTV(edge_weight=0.05, tv_weight=lam, sky_threshold=TAU_FIXED).to(DEVICE)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
    total_steps = len(dataloader) // ACCUM_STEPS * NUM_FT_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=LR * 0.1)
    scaler = torch.amp.GradScaler("cuda")

    nan_alert = False
    start_time = time.time()
    for epoch in range(CHECKPOINT_EPOCH, FINAL_EPOCH):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"[{tag}] Epoch {epoch+1}/{FINAL_EPOCH}")
        optimizer.zero_grad()
        for i, (hazy, clear, depth) in enumerate(pbar):
            hazy, clear, depth = hazy.to(DEVICE), clear.to(DEVICE), depth.to(DEVICE)
            with torch.amp.autocast("cuda"):
                preds = model(hazy, depth)
                loss = criterion(preds, clear, depth) / ACCUM_STEPS
            if torch.isnan(loss) or torch.isinf(loss):
                if not nan_alert:
                    print(f"\n  ⚠️ [{tag}] NaN/inf loss! epoch {epoch+1}, step {i}")
                    nan_alert = True
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            cur = loss.item() * ACCUM_STEPS
            epoch_loss += cur
            if i % 50 == 0:
                pbar.set_postfix({"Loss": f"{cur:.4f}"})
        remainder = len(dataloader) % ACCUM_STEPS
        if remainder != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        avg_loss = epoch_loss / len(dataloader)
        print(f"  [{tag}] Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"  [{tag}] Training done! {elapsed:.1f}s")

    save_path = os.path.join(BASE, "checkpoints_ots", f"ffa_sens_v3mini_{tag}_epoch{FINAL_EPOCH}.pth")
    torch.save({"epoch": FINAL_EPOCH, "model_state_dict": model.state_dict(), "lambda": lam, "tau": TAU_FIXED}, save_path)

    metrics = evaluate_model(model, tag)
    return {
        "lambda": lam, "tau": TAU_FIXED, "tag": tag,
        "elapsed_sec": elapsed, "metrics": metrics,
    }


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE, "checkpoints_ots"), exist_ok=True)
    print(f"DEVICE: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    assert os.path.exists(CHECKPOINT_PATH), "Checkpoint not found!"

    print(f"\n{'#'*60}")
    print(f"V3-Mini-Lambda 敏感性测试：1 τ × {len(LAMBDAS)} λ = {len(LAMBDAS)} 组合")
    print(f"τ 固定: {TAU_FIXED}, λ ∈ {LAMBDAS}")
    print(f"数据集: OTS_MINI (10% 采样)")
    print(f"每组合: epoch {CHECKPOINT_EPOCH}→{FINAL_EPOCH} ({NUM_FT_EPOCHS} epochs)")
    print(f"骨干冻结: g1/g2/g3/rgb_pre_conv")
    print(f"{'#'*60}\n")

    all_results = []
    total_start = time.time()
    for idx, lam in enumerate(LAMBDAS):
        tag = f"lam{lam:.3f}_tau{TAU_FIXED:.2f}"
        save_path = os.path.join(BASE, "checkpoints_ots", f"ffa_sens_v3mini_{tag}_epoch{FINAL_EPOCH}.pth")

        if os.path.exists(save_path):
            print(f"\n{'='*60}\n  [{idx+1}/{len(LAMBDAS)}] λ={lam} — 权重已存在，跳过训练，仅评估\n{'='*60}")
            from models.ffanet import FFA_DGCCA_Net
            model = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(DEVICE)
            ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            metrics = evaluate_model(model, tag)
            result = {"lambda": lam, "tau": TAU_FIXED, "tag": tag, "metrics": metrics, "skipped_training": True}
        else:
            print(f"\n{'='*60}\n  [{idx+1}/{len(LAMBDAS)}] λ={lam}\n{'='*60}")
            result = run_one(lam)
        all_results.append(result)

    total_elapsed = time.time() - total_start

    json_path = os.path.join(BASE, "sensitivity_v3_mini_lambda.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": {"tau_fixed": TAU_FIXED, "lambdas": LAMBDAS, "epochs": NUM_FT_EPOCHS, "dataset": "OTS_MINI"},
            "total_elapsed_sec": total_elapsed,
            "results": all_results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  All {len(LAMBDAS)} combos done! Total: {total_elapsed/60:.1f} min")
    print(f"  Results saved: {json_path}")
    print(f"{'='*60}")