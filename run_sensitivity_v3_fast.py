"""
敏感性快速测试 V3：按 τ 值的单独组合训练（λ 固定 0.01）
用法（串行/单组合）： python run_sensitivity_v3_fast.py --tau 0.05
用法（3 个终端并行）： 在 3 个终端窗口分别执行 python run_sensitivity_v3_fast.py --tau 0.05 / 0.10 / 0.15
结果分别写入 sensitivity_v3_tau{tau}.json
若不加 --tau 则按顺序串行运行所有预设值
使用 dehaze 环境
"""
import os, torch, time, json, argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import glob

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 固定参数 ==========
LAMBDA_FIXED = 0.01
CHECKPOINT_EPOCH = 100
NUM_FT_EPOCHS = 10  # 仅微调 10 epoch
CHECKPOINT_PATH = os.path.join(BASE, "checkpoints_ots", f"ffa_dgcca_ots_dav2_epoch_{CHECKPOINT_EPOCH}.pth")
TRAIN_HAZY  = os.path.join(BASE, "data", "RESIDE", "OTS", "hazy")
TRAIN_CLEAR = os.path.join(BASE, "data", "RESIDE", "OTS", "clear")
TRAIN_DEPTH = os.path.join(BASE, "data", "RESIDE", "OTS", "depth_dav2")
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
    """评估"""
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

def run_one(tau):
    tag = f"tau{tau:.2f}_lam{LAMBDA_FIXED:.2f}"
    print(f"\n{'='*60}\n  τ={tau}, λ={LAMBDA_FIXED}  ({tag})\n{'='*60}")

    from models.ffanet import FFA_DGCCA_Net
    model = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt else ckpt["model_state_dict"])

    from utils.dataset_ots import OTSDataset
    dataset = OTSDataset(TRAIN_HAZY, TRAIN_CLEAR, TRAIN_DEPTH, crop_size=256, is_train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    from utils.loss_v2_sky_tv import CompositeLoss_SkyTV
    criterion = CompositeLoss_SkyTV(edge_weight=0.05, tv_weight=LAMBDA_FIXED, sky_threshold=tau).to(DEVICE)

    # 2. 冻结 FFA 骨干 (g1, g2, g3) 和 rgb_pre_conv，仅微调融合/深度/后处理模块
    for name, param in model.named_parameters():
        if any(k in name for k in ["g1.", "g2.", "g3.", "rgb_pre_conv."]):
            param.requires_grad = False
        else:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
    # T_max 使用实际总步数而非 epoch 总数，避免学习率塌缩
    total_steps = len(dataloader) // ACCUM_STEPS * NUM_FT_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=LR * 0.1)
    scaler = torch.amp.GradScaler("cuda")

    best_loss = float("inf")
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
            # NaN 检测：loss 中出现 NaN 时跳过该步并报警
            if torch.isnan(loss) or torch.isinf(loss):
                if not nan_alert:
                    print(f"\n  ⚠️ [{tag}] NaN/inf loss detected at epoch {epoch+1}, step {i} — 跳过该 batch")
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
            if i % 100 == 0:
                pbar.set_postfix({"Loss": f"{cur:.4f}"})
        # epoch 末端残余梯度处理
        remainder = len(dataloader) % ACCUM_STEPS
        if remainder != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        avg = epoch_loss / len(dataloader)
        best_loss = min(best_loss, avg)
        print(f"  [{tag}] Epoch {epoch+1}: Avg Loss = {avg:.4f}")
    elapsed = time.time() - start_time
    print(f"  [{tag}] Done! {elapsed:.1f}s, Best Loss = {best_loss:.4f}")

    # 权重保存
    save_path = os.path.join(BASE, "checkpoints_ots", f"ffa_sens_v3_{tag}_epoch{FINAL_EPOCH}.pth")
    torch.save({"epoch": FINAL_EPOCH, "model_state_dict": model.state_dict(), "tau": tau, "lambda": LAMBDA_FIXED}, save_path)

    # 评估并保存单独 JSON
    metrics = evaluate_model(model, tag)
    result = {"tau": tau, "lambda": LAMBDA_FIXED, "tag": tag, "elapsed_sec": elapsed, "best_loss": best_loss, "metrics": metrics}
    json_path = os.path.join(BASE, f"sensitivity_v3_tau{tau:.2f}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"💾 结果已保存: {json_path}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=None, help="指定单个 τ 值，如不加则运行所有预设值")
    args = parser.parse_args()

    os.makedirs(os.path.join(BASE, "checkpoints_ots"), exist_ok=True)
    print(f"DEVICE: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    assert os.path.exists(CHECKPOINT_PATH), "Checkpoint not found!"

    if args.tau is not None:
        # 单组合模式
        print(f"\n{'#'*60}")
        print(f"V3 快速测试：单组合 τ={args.tau}")
        print(f"τ={args.tau}，λ={LAMBDA_FIXED}，{NUM_FT_EPOCHS} epochs")
        print(f"{'#'*60}\n")
        run_one(args.tau)
    else:
        # 串行模式（旧行为）
        TAUS = [0.05, 0.10, 0.15]
        print(f"\n{'#'*60}")
        print(f"V3 快速测试：{len(TAUS)} τ × 1 λ = {len(TAUS)} 组合")
        print(f"每组合：{CHECKPOINT_EPOCH}→{FINAL_EPOCH} epoch ({NUM_FT_EPOCHS} epochs) + 5 数据集评估")
        print(f"{'#'*60}\n")
        total_start = time.time()
        for idx, tau in enumerate(TAUS):
            print(f"\n{'='*60}\n  [{idx+1}/{len(TAUS)}] τ={tau}\n{'='*60}")
            run_one(tau)
        total_elapsed = time.time() - total_start
        print(f"\n全部 {len(TAUS)} 组合完成！总耗时 {total_elapsed/3600:.1f} 小时")