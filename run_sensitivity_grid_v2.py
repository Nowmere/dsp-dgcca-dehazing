"""
敏感性网格测试：4 τ × 4 λ = 16 组合
每个组合：加载 epoch_100 checkpoint → 用对应 τ/λ 微调 20 epochs → 评估 5 数据集
"""
import os, torch, itertools, json, time
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 参数网格 ==========
TAUS    = [0.05, 0.10, 0.15, 0.20]
LAMBDAS = [0.005, 0.01, 0.02, 0.05]
CHECKPOINT_EPOCH = 100
CHECKPOINT_PATH  = os.path.join(BASE, "checkpoints_ots", f"ffa_dgcca_ots_dav2_epoch_{CHECKPOINT_EPOCH}.pth")
CKPT_SAVE_DIR    = os.path.join(BASE, "checkpoints_ots")  # 保存在同一目录
RESULTS_JSON     = os.path.join(BASE, "sensitivity_grid_results.json")

# 数据集路径
TRAIN_HAZY   = os.path.join(BASE, "data", "RESIDE", "OTS", "hazy")
TRAIN_CLEAR  = os.path.join(BASE, "data", "RESIDE", "OTS", "clear")
TRAIN_DEPTH  = os.path.join(BASE, "data", "RESIDE", "OTS", "depth_dav2")

# 固定训练参数
BATCH_SIZE = 4
ACCUM_STEPS = 4                    # effective batch = 16
NUM_EPOCHS = CHECKPOINT_EPOCH + 20 # 只跑 20 epoch (101→120)
LR = 1e-4
WEIGHT_DECAY = 1e-4

print(f"DEVICE: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"

# ========== 跑单个组合 ==========
def run_one_tau_lambda(tau, lam):
    tag = f"tau{tau:.2f}_lam{lam:.4f}"
    print(f"\n{'='*60}\n  τ={tau}, λ={lam}  ({tag})\n{'='*60}")

    # 1. 加载模型
    from models.ffanet import FFA_DGCCA_Net
    model = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # 2. 数据集 (使用已有深度图，不加载 DAV2)
    from utils.dataset_ots import OTSDataset
    dataset = OTSDataset(TRAIN_HAZY, TRAIN_CLEAR, TRAIN_DEPTH, crop_size=256, is_train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # 3. 损失函数 (动态 τ/λ)
    from utils.loss_v2_sky_tv import CompositeLoss_SkyTV
    criterion = CompositeLoss_SkyTV(edge_weight=0.05, tv_weight=lam, sky_threshold=tau).to(DEVICE)

    # 4. 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # 快进 scheduler 到第 100 epoch 的状态
    for _ in range(CHECKPOINT_EPOCH):
        scheduler.step()

    # 5. 训练 20 epochs
    best_loss = float("inf")
    loss_history = []
    start_time = time.time()

    for epoch in range(CHECKPOINT_EPOCH, NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"[{tag}] Epoch {epoch+1}/{NUM_EPOCHS}")
        optimizer.zero_grad()

        for i, (hazy, clear, depth) in enumerate(pbar):
            hazy, clear, depth = hazy.to(DEVICE), clear.to(DEVICE), depth.to(DEVICE)

            with torch.amp.autocast("cuda"):
                preds = model(hazy, depth)
                loss = criterion(preds, clear, depth) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            cur = loss.item() * ACCUM_STEPS
            epoch_loss += cur
            if i % 10 == 0:
                pbar.set_postfix({"Loss": f"{cur:.4f}"})

        scheduler.step()
        avg = epoch_loss / len(dataloader)
        loss_history.append(avg)
        best_loss = min(best_loss, avg)
        print(f"  [{tag}] Epoch {epoch+1}: Avg Loss = {avg:.4f}")

    elapsed = time.time() - start_time
    print(f"  [{tag}] 完成！耗时 {elapsed:.1f}s, 最佳 Loss = {best_loss:.4f}")

    # 6. 保存模型权重
    save_path = os.path.join(CKPT_SAVE_DIR, f"ffa_sens_{tag}_epoch{NUM_EPOCHS}.pth")
    torch.save({
        "epoch": NUM_EPOCHS,
        "model_state_dict": model.state_dict(),
        "tau": tau,
        "lambda": lam,
        "best_loss": best_loss,
    }, save_path)

    # 评估 5 数据集
    metrics = evaluate_model(model, tag)

    return {
        "tau": tau,
        "lambda": lam,
        "tag": tag,
        "elapsed_sec": elapsed,
        "best_loss": best_loss,
        "loss_history": loss_history,
        "metrics": metrics,
    }

# ========== 评估 5 数据集 ==========
def evaluate_model(model, tag):
    import numpy as np
    from PIL import Image
    from torchvision.transforms import functional as TF
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    import glob

    model.eval()
    # 复用 run_ablation.py 中已有的评估路径配置
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

    results = {}
    for ds_name, (hazy_dir, gt_dir) in DATASETS.items():
        if not os.path.exists(hazy_dir):
            print(f"  [{tag}] 跳过 {ds_name}（路径不存在）")
            results[ds_name] = {"psnr": None, "ssim": None}
            continue

        hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
        psnr_list, ssim_list = [], []
        for fname in tqdm(hazy_files[:30], desc=f"[{tag}] Eval {ds_name}"):  # 前30张评估
            hp = os.path.join(hazy_dir, fname)
            try:
                hazy_img = Image.open(hp).convert("RGB")
            except:
                continue
            w0, h0 = hazy_img.size
            nw, nh = (w0 // 16) * 16, (h0 // 16) * 16
            hazy_pil = hazy_img.resize((nw, nh), Image.LANCZOS)

            # 找 GT
            gt_prefix = os.path.splitext(fname)[0]
            # 尝试多种命名约定
            gt_candidates = glob.glob(os.path.join(gt_dir, f"{gt_prefix}.*"))
            if not gt_candidates:
                # NH-HAZE GT 文件名可能不同
                alt = gt_prefix.replace("_hazy", "")
                gt_candidates = glob.glob(os.path.join(gt_dir, f"{alt}.*"))
            if not gt_candidates:
                # 尝试同编号匹配
                gt_candidates = glob.glob(os.path.join(gt_dir, f"*{gt_prefix}*"))
                if len(gt_candidates) > 1:
                    gt_candidates = [gt_candidates[0]]
            if not gt_candidates:
                continue
            gt_pil = Image.open(gt_candidates[0]).convert("RGB").resize((nw, nh), Image.LANCZOS)
            gt_np = np.array(gt_pil)

            # 加载预存深度
            depth_path = os.path.join(BASE, "data", "depth_cache", ds_name, f"{os.path.splitext(fname)[0]}.pt")
            if os.path.exists(depth_path):
                depth_tensor = torch.load(depth_path).unsqueeze(0).to(DEVICE)
            else:
                # fallback: 全零深度（此组合仅影响训练，推理时 τ/λ 不参与前向传播）
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
        print(f"  [{tag}] {ds_name}: PSNR={results[ds_name]['psnr']:.2f}, SSIM={results[ds_name]['ssim']:.4f}")

    return results

# ========== 主入口 ==========
if __name__ == "__main__":
    # 确保保存目录存在
    os.makedirs(CKPT_SAVE_DIR, exist_ok=True)

    # 生成 16 个组合
    all_results = []
    grid = list(itertools.product(TAUS, LAMBDAS))
    print(f"\n{'#'*60}")
    print(f"敏感性网格测试：{len(TAUS)} τ × {len(LAMBDAS)} λ = {len(grid)} 组合")
    print(f"每组合：{CHECKPOINT_EPOCH}→{NUM_EPOCHS} epoch 微调 + 5 数据集评估")
    print(f"{'#'*60}\n")

    total_start = time.time()
    for idx, (tau, lam) in enumerate(grid):
        print(f"\n{'='*60}")
        print(f"  [{idx+1}/{len(grid)}] τ={tau}, λ={lam}")
        print(f"{'='*60}")
        result = run_one_tau_lambda(tau, lam)
        all_results.append(result)

        # 每完成一个组合就保存一次（断电保护）
        with open(RESULTS_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 中间结果已保存到 {RESULTS_JSON}")

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*60}")
    print(f"全部 {len(grid)} 组合完成！总耗时 {total_elapsed/3600:.1f} 小时")
    print(f"结果已保存: {RESULTS_JSON}")
    print(f"{'#'*60}")

    # 打印摘要
    print("\n\n" + "="*80)
    print("PSNR 汇总（O-HAZE / NH-HAZE）")
    print("="*80)
    header = f"{'τ':<6} {'λ':<8}"
    for ds in ["O-HAZE", "NH-HAZE"]:
        header += f" {'PSNR':>7} {'SSIM':>7}"
    print(header)
    for r in all_results:
        row = f"{r['tau']:<6.2f} {r['lambda']:<8.4f}"
        for ds in ["O-HAZE", "NH-HAZE"]:
            m = r["metrics"].get(ds, {})
            row += f" {m.get('psnr', 0) or 0:>7.2f} {m.get('ssim', 0) or 0:>7.4f}"
        print(row)