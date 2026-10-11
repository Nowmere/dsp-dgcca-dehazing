import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import glob
import numpy as np
import cv2
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

# ==================== 数据集配置 ====================
DATASETS_CFG = {
    "SOTS-Outdoor": {
        "hazy_dir": os.path.join(BASE_DIR, "data", "RESIDE", "SOTS", "outdoor", "hazy"),
        "gt_dir": os.path.join(BASE_DIR, "data", "RESIDE", "SOTS", "outdoor", "clear"),
        "gt_from_hazy": lambda name: name.split("_")[0],
        # B 变体用 hazy depth (训练时的深度图)
        "hazy_depth_dir": os.path.join(BASE_DIR, "data", "RESIDE", "SOTS", "outdoor", "depth_hazy"),
    },
    "SOTS-Indoor": {
        "hazy_dir": os.path.join(BASE_DIR, "data", "SOTS", "indoor", "hazy"),
        "gt_dir": os.path.join(BASE_DIR, "data", "SOTS", "indoor", "clear"),
        "gt_from_hazy": lambda name: name.split("_")[0],
    },
    "O-HAZE": {
        "hazy_dir": os.path.join(BASE_DIR, "data", "O-HAZE", "hazy"),
        "gt_dir": os.path.join(BASE_DIR, "data", "O-HAZE", "GT"),
        "gt_from_hazy": lambda name: name,  # 同名匹配
    },
    "NH-HAZE": {
        "hazy_dir": os.path.join(BASE_DIR, "data", "NH-HAZE", "hazy"),
        "gt_dir": os.path.join(BASE_DIR, "data", "NH-HAZE", "GT"),
        # nh10_hazy.png -> nh10.png
        "gt_from_hazy": lambda name: name.replace("_hazy", ""),
    },
    "I-HAZE": {
        "hazy_dir": os.path.join(BASE_DIR, "data", "I-HAZE", "hazy"),
        "gt_dir": os.path.join(BASE_DIR, "data", "I-HAZE", "GT"),
        # ih10_hazy.png -> ih10.png
        "gt_from_hazy": lambda name: name.replace("_hazy", ""),
    },
    "Dense-Haze": {
        "hazy_dir": os.path.join(BASE_DIR, "data", "Dense_Haze", "hazy"),
        "gt_dir": os.path.join(BASE_DIR, "data", "Dense_Haze", "GT"),
        # 01_hazy.png -> 01_GT.png
        "gt_from_hazy": lambda name: name.replace("_hazy", "_GT"),
    },
}

# ==================== 消融变体配置 ====================
VARIANTS_CFG = [
    {
        "name": "A-Baseline",
        "ckpt": "checkpoints_ots/ffa_baseline_ots_epoch_100.pth",
        "model_cls": "baseline",  # FFA_Baseline_Net, 不需要 depth
    },
    {
        "name": "B-+DG-CCA",
        "ckpt": "checkpoints_ots/ffa_dgcca_ots_epoch_100.pth",
        "model_cls": "dgcca",  # FFA_DGCCA_Net, 需要 depth
    },
    {
        "name": "C-+DAV2",
        "ckpt": "checkpoints_ots/ffa_dgcca_ots_dav2_epoch_100.pth",
        "model_cls": "dgcca",
    },
    {
        "name": "D-+SkyTV",
        "ckpt": "checkpoints_ots/ffa_dgcca_ots_dav2_epoch_120.pth",
        "model_cls": "dgcca",
    },
]


def load_model(variant, device):
    """根据变体配置加载对应模型和权重"""
    if variant["model_cls"] == "baseline":
        model = FFA_Baseline_Net(gps=3, blocks=3, channels=64).to(device).eval()
    else:
        model = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(device).eval()

    ckpt_path = os.path.join(BASE_DIR, variant["ckpt"])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    return model


def find_gt_path(gt_dir, hazy_name, gt_from_hazy_fn):
    """模糊搜索匹配 GT 文件"""
    base = gt_from_hazy_fn(hazy_name)
    # 去除后缀
    base_no_ext = os.path.splitext(base)[0]
    candidates = glob.glob(os.path.join(gt_dir, f"{base_no_ext}.*"))
    if candidates:
        return candidates[0]
    # 尝试直接拼接
    direct = os.path.join(gt_dir, base)
    if os.path.exists(direct):
        return direct
    return None


def run_ablation():
    print("=" * 70)
    print("Ablation Study | 4 variants x 5 datasets")
    print("=" * 70)

    # 1. 加载 DAV2 深度估计器 (只加载一次, B/C/D 共用)
    print("\n[Step 1] Loading DepthAnything V2 (Small) from cache...", flush=True)
    depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    print("  ImageProcessor OK", flush=True)
    depth_estimator = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    ).to(DEVICE).eval()
    print("  DepthAnything V2 model OK", flush=True)

    # 2. 预加载所有模型变体
    models = {}
    for v_cfg in VARIANTS_CFG:
        print(f"[Step 2] Loading variant: {v_cfg['name']} <- {v_cfg['ckpt']}", flush=True)
        models[v_cfg["name"]] = load_model(v_cfg, DEVICE)
        print(f"  {v_cfg['name']} OK", flush=True)

    # 3. 存储所有结果: results[dataset][variant] = {"psnr": [], "ssim": []}
    all_results = {}

    def get_dav2_depth(img_pil, target_h, target_w):
        """复用 cvpr 测试脚本的深度提取逻辑"""
        inputs = depth_processor(images=img_pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = depth_estimator(**inputs)
            depth = outputs.predicted_depth

        depth_rescaled = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(target_h, target_w),
            mode="bicubic", align_corners=False
        ).squeeze()

        d_min, d_max = depth_rescaled.min(), depth_rescaled.max()
        depth_norm = (depth_rescaled - d_min) / (d_max - d_min + 1e-8)
        return depth_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # 4. 逐数据集测试
    for ds_name, ds_cfg in DATASETS_CFG.items():
        hazy_dir = ds_cfg["hazy_dir"]
        gt_dir = ds_cfg["gt_dir"]
        gt_fn = ds_cfg["gt_from_hazy"]

        if not os.path.isdir(hazy_dir):
            print(f"\n[SKIP] {ds_name}: hazy dir not found ({hazy_dir})")
            continue

        hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not hazy_files:
            print(f"\n[SKIP] {ds_name}: no valid images")
            continue

        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name} ({len(hazy_files)} images)")
        print(f"{'='*70}")

        ds_results = {v["name"]: {"psnr": [], "ssim": []} for v in VARIANTS_CFG}

        valid_count = 0
        for filename in tqdm(hazy_files, desc=ds_name):
            hazy_path = os.path.join(hazy_dir, filename)
            gt_path = find_gt_path(gt_dir, filename, gt_fn)
            if gt_path is None:
                continue

            try:
                hazy_pil = Image.open(hazy_path).convert('RGB')
                gt_pil = Image.open(gt_path).convert('RGB')
            except Exception:
                continue

            w, h = hazy_pil.size
            # 限制显存
            max_side = 1024
            if max(w, h) > max_side:
                r = max_side / float(max(w, h))
                w, h = int(w * r), int(h * r)

            nw, nh = (w // 16) * 16, (h // 16) * 16
            hazy_pil = hazy_pil.resize((nw, nh), Image.LANCZOS)
            gt_pil = gt_pil.resize((nw, nh), Image.LANCZOS)
            gt_np = np.array(gt_pil)

            hazy_tensor = TF.to_tensor(hazy_pil).unsqueeze(0).to(DEVICE)

            # 预提取 DAV2 深度图 (C/D 共用, B 在 SOTS-Outdoor 用 hazy depth)
            dav2_depth = get_dav2_depth(hazy_pil, nh, nw)

            # B 变体专用: 尝试加载 hazy depth (训练时匹配的深度图)
            b_depth = dav2_depth  # 默认 fallback
            hazy_depth_dir = ds_cfg.get("hazy_depth_dir", None)
            if hazy_depth_dir and os.path.isdir(hazy_depth_dir):
                # hazy depth 文件名与 hazy 图片同名 (如 0001_0.8_0.2.jpg)
                base_no_ext = os.path.splitext(filename)[0]
                candidates = glob.glob(os.path.join(hazy_depth_dir, f"{base_no_ext}.*"))
                if not candidates:
                    # 尝试用 hazy 文件名前缀匹配 (SOTS-Indoor naming)
                    prefix = filename.split("_")[0]
                    candidates = glob.glob(os.path.join(hazy_depth_dir, f"{prefix}*"))
                if candidates:
                    try:
                        depth_pil = Image.open(candidates[0]).convert('L')
                        depth_pil = depth_pil.resize((nw, nh), Image.LANCZOS)
                        b_depth = TF.to_tensor(depth_pil).unsqueeze(0).to(DEVICE)
                    except Exception:
                        pass  # fallback to dav2

            for v_cfg in VARIANTS_CFG:
                v_name = v_cfg["name"]
                model = models[v_name]

                with torch.no_grad():
                    if v_cfg["model_cls"] == "baseline":
                        out = model(hazy_tensor)
                    elif v_name == "B-+DG-CCA":
                        out = model(hazy_tensor, b_depth)
                    else:
                        out = model(hazy_tensor, dav2_depth)

                out_np = (out.clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                try:
                    p_val = psnr(gt_np, out_np, data_range=255)
                    s_val = ssim(gt_np, out_np, data_range=255, channel_axis=2)
                except Exception:
                    p_val, s_val = 0, 0

                ds_results[v_name]["psnr"].append(p_val)
                ds_results[v_name]["ssim"].append(s_val)

            valid_count += 1
            torch.cuda.empty_cache()

        # 计算平均值
        for v_name in ds_results:
            psnrs = ds_results[v_name]["psnr"]
            ssims = ds_results[v_name]["ssim"]
            ds_results[v_name]["avg_psnr"] = np.mean(psnrs) if psnrs else 0
            ds_results[v_name]["avg_ssim"] = np.mean(ssims) if ssims else 0

        all_results[ds_name] = ds_results
        print(f"  Valid samples: {valid_count}")

    # 5. 打印汇总报告
    print("\n\n")
    print("=" * 90)
    print("Ablation Study Final Report")
    print("=" * 90)
    print(f"{'Dataset':<18} {'Metric':<8} {'A-Baseline':>12} {'B-+DG-CCA':>12} {'C-+DAV2':>12} {'D-+SkyTV':>12}")
    print("-" * 90)

    for ds_name in DATASETS_CFG:
        if ds_name not in all_results:
            continue
        res = all_results[ds_name]
        names = [v["name"] for v in VARIANTS_CFG]

        psnr_line = f"{ds_name:<18} {'PSNR':<8}"
        ssim_line = f"{'':<18} {'SSIM':<8}"
        for v_name in names:
            psnr_line += f" {res[v_name]['avg_psnr']:>11.2f}"
            ssim_line += f" {res[v_name]['avg_ssim']:>11.4f}"
        print(psnr_line)
        print(ssim_line)
        print("-" * 90)

    # 6. 打印 LaTeX 表格
    print("\n\n" + "=" * 90)
    print("LaTeX Table (copy to paper)")
    print("=" * 90)

    ds_order = ["SOTS-Outdoor", "SOTS-Indoor", "O-HAZE", "NH-HAZE", "I-HAZE"]
    v_names = [v["name"] for v in VARIANTS_CFG]

    # PSNR 表格
    print("\n% --- PSNR (dB) ---")
    header = "Dataset & " + " & ".join(v_names) + " \\\\"
    print(header)
    print("\\midrule")
    for ds_name in ds_order:
        if ds_name not in all_results:
            continue
        line = ds_name
        for v_name in v_names:
            line += f" & {all_results[ds_name][v_name]['avg_psnr']:.2f}"
        line += " \\\\"
        print(line)

    # SSIM 表格
    print("\n% --- SSIM ---")
    print(header)
    print("\\midrule")
    for ds_name in ds_order:
        if ds_name not in all_results:
            continue
        line = ds_name
        for v_name in v_names:
            line += f" & {all_results[ds_name][v_name]['avg_ssim']:.4f}"
        line += " \\\\"
        print(line)

    print("\n" + "=" * 90)
    print("Ablation complete! Tables ready for paper Section 4.4")
    print("=" * 90)


if __name__ == "__main__":
    run_ablation()