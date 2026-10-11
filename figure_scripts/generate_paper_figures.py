"""
generate_paper_figures.py
一次性生成论文所需的全部 4 张图片到 CVPR_2026_Submission_Template/figures/
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from models.ffanet import FFA_Baseline_Net, FFA_DGCCA_Net

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "CVPR_2026_Submission_Template", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 字体配置 ==========
def get_font(size=20):
    """兼容中英文的字体查找"""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()

# ========== 工具函数 ==========
def load_model_and_depth_estimator():
    """加载 4 个模型变体 + DAV2"""
    print("[Loading] Models...")
    models = {}

    # A - Baseline
    m = FFA_Baseline_Net(gps=3, blocks=3, channels=64).to(DEVICE).eval()
    ckpt = torch.load(os.path.join(BASE_DIR, "checkpoints_ots", "ffa_baseline_ots_epoch_100.pth"),
                       map_location=DEVICE, weights_only=True)
    m.load_state_dict(ckpt)
    models["A-Baseline"] = m

    # D - Ours full
    m = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(DEVICE).eval()
    ckpt = torch.load(os.path.join(BASE_DIR, "checkpoints_ots", "ffa_dgcca_ots_dav2_epoch_120.pth"),
                       map_location=DEVICE, weights_only=False)
    if "model_state_dict" in ckpt:
        m.load_state_dict(ckpt["model_state_dict"])
    else:
        m.load_state_dict(ckpt)
    models["D-SkyTV"] = m

    # DAV2
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    estimator = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(DEVICE).eval()

    return models, processor, estimator


def compute_dav2_depth(img_pil, processor, estimator, h, w):
    """DAV2 深度提取"""
    inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = estimator(**inputs)
        depth = outputs.predicted_depth
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze()
    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    return depth_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def to_np(tensor):
    return (tensor.clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def add_label(img_np, text, color=(255, 255, 255)):
    """在图片顶部加白色标签条"""
    h, w = img_np.shape[:2]
    bar_h = 40
    labeled = np.ones((h + bar_h, w, 3), dtype=np.uint8) * 40
    labeled[bar_h:, :, :] = img_np
    # PIL drawing
    pil_img = Image.fromarray(labeled)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = get_font(18)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tx = (w - bbox[2] - bbox[0]) // 2
    ty = (bar_h - bbox[3] - bbox[1]) // 2
    draw.text((tx, ty), text, fill=color, font=font)
    return np.array(pil_img)


def hstack_images(imgs, gap=4):
    """水平拼接多张图"""
    h = max(i.shape[0] for i in imgs)
    total_w = sum(i.shape[1] for i in imgs) + gap * (len(imgs) - 1)
    canvas = np.ones((h, total_w, 3), dtype=np.uint8) * 255
    x = 0
    for im in imgs:
        canvas[:im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    return canvas


# ============================================================
#  Fig.4: Mask Ablation
# ============================================================
def generate_fig4_mask_ablation(models_dict, processor, estimator):
    print("\n" + "=" * 60)
    print("[Fig.4] Mask Ablation: hazy -> depth -> sky mask")
    print("=" * 60)

    # 选 NH-HAZE 第一张
    hazy_dir = os.path.join(BASE_DIR, "data", "NH-HAZE", "hazy")
    hazy_name = "nh1_hazy.png"
    hazy_path = os.path.join(hazy_dir, hazy_name)

    hazy_pil = Image.open(hazy_path).convert("RGB")
    w, h = hazy_pil.size
    nw, nh = (w // 16) * 16, (h // 16) * 16
    hazy_pil = hazy_pil.resize((nw, nh), Image.LANCZOS)

    # 提取 DAV2 深度
    depth_np = compute_dav2_depth(hazy_pil, processor, estimator, nh, nw)
    depth_np = depth_np.squeeze().cpu().numpy()

    # 生成 sky mask
    mask = (depth_np < 0.1).astype(np.uint8) * 255

    # 转可视化
    hazy_viz = np.array(hazy_pil.resize((nw, nh), Image.LANCZOS))
    depth_viz = (depth_np * 255).astype(np.uint8)
    depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_INFERNO)

    # mask 显示为红色半透明叠加
    mask_overlay = hazy_viz.copy()
    mask_overlay[mask > 0] = mask_overlay[mask > 0] * 0.5 + np.array([200, 50, 50], dtype=np.uint8) * 0.5

    hazy_l = add_label(hazy_viz, "(a) Hazy Input (NH-HAZE)")
    depth_l = add_label(depth_viz, "(b) DAV2 Depth Map")
    mask_l = add_label(mask_overlay, "(c) Sky Mask (tau < 0.1)")

    combined = hstack_images([hazy_l, depth_l, mask_l], gap=8)

    out_path = os.path.join(OUT_DIR, "mask_ablation.pdf")
    Image.fromarray(combined).save(out_path)
    print(f"  Saved: {out_path}")


# ============================================================
#  Fig.1: Teaser (NH-HAZE: Baseline vs Ours)
# ============================================================
def generate_fig1_teaser(models_dict, processor, estimator):
    print("\n" + "=" * 60)
    print("[Fig.1] Teaser: NH-HAZE Baseline vs Ours")
    print("=" * 60)

    hazy_dir = os.path.join(BASE_DIR, "data", "NH-HAZE", "hazy")
    gt_dir = os.path.join(BASE_DIR, "data", "NH-HAZE", "GT")
    hazy_name = "nh1_hazy.png"
    hazy_path = os.path.join(hazy_dir, hazy_name)
    gt_path = os.path.join(gt_dir, "nh1.png")

    hazy_pil = Image.open(hazy_path).convert("RGB")
    gt_pil = Image.open(gt_path).convert("RGB")
    w, h = hazy_pil.size
    nw, nh = (w // 16) * 16, (h // 16) * 16
    hazy_pil = hazy_pil.resize((nw, nh), Image.LANCZOS)
    gt_pil = gt_pil.resize((nw, nh), Image.LANCZOS)

    hazy_tensor = TF.to_tensor(hazy_pil).unsqueeze(0).to(DEVICE)
    depth_map = compute_dav2_depth(hazy_pil, processor, estimator, nh, nw)

    # Baseline
    with torch.no_grad():
        out_base = models_dict["A-Baseline"](hazy_tensor)
    base_np = to_np(out_base)

    # Ours
    with torch.no_grad():
        out_ours = models_dict["D-SkyTV"](hazy_tensor, depth_map)
    ours_np = to_np(out_ours)

    hazy_np = np.array(hazy_pil.resize((nw, nh), Image.LANCZOS))
    gt_np = np.array(gt_pil.resize((nw, nh), Image.LANCZOS))

    hazy_l = add_label(hazy_np, "Hazy Input")
    base_l = add_label(base_np, "FFA-Net (Baseline)", color=(255, 100, 100))
    ours_l = add_label(ours_np, "Ours (DG-CCA+DAV2+SkyTV)", color=(100, 255, 100))
    gt_l = add_label(gt_np, "Ground Truth")

    combined = hstack_images([hazy_l, base_l, ours_l, gt_l], gap=8)

    out_path = os.path.join(OUT_DIR, "teaser.pdf")
    Image.fromarray(combined).save(out_path)
    print(f"  Saved: {out_path}")


# ============================================================
#  Fig.2: Architecture Diagram
# ============================================================
def generate_fig2_architecture():
    print("\n" + "=" * 60)
    print("[Fig.2] Architecture Diagram")
    print("=" * 60)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Colors
    c_rgb = "#4A90D9"
    c_depth = "#E67E22"
    c_cca = "#27AE60"
    c_sky = "#E74C3C"
    c_arrow = "#555555"

    # --- RGB Backbone (left) ---
    ax.add_patch(plt.Rectangle((0.5, 2.0), 4.0, 2.5, facecolor=c_rgb, alpha=0.2, edgecolor=c_rgb, linewidth=2))
    ax.text(2.5, 3.6, "RGB Backbone\n(FFA-Net)", ha="center", va="center", fontsize=11, fontweight="bold", color=c_rgb)

    # --- DAV2 Branch (top) ---
    ax.add_patch(plt.Rectangle((0.5, 4.8), 4.0, 0.9, facecolor=c_depth, alpha=0.2, edgecolor=c_depth, linewidth=2))
    ax.text(2.5, 5.25, "Frozen DAV2\nDepth Foundation Model", ha="center", va="center", fontsize=9, color=c_depth)

    # --- DG-CCA (center) ---
    ax.add_patch(plt.Rectangle((5.2, 2.0), 3.6, 2.5, facecolor=c_cca, alpha=0.15, edgecolor=c_cca, linewidth=2))
    ax.text(7.0, 4.0, "DG-CCA Module", ha="center", va="center", fontsize=11, fontweight="bold", color=c_cca)
    ax.text(7.0, 3.5, "Global AvgPool", ha="center", va="center", fontsize=8, color="#555")
    ax.text(7.0, 3.1, "MLP + Sigmoid Gate", ha="center", va="center", fontsize=8, color="#555")
    ax.text(7.0, 2.7, "$\\mathbf{A} \\!=\\! \\sigma(W_2\\delta(W_1[z_{rgb};z_d]))$", ha="center", va="center", fontsize=8, color="#555")
    ax.text(7.0, 2.3, "$F_{out} \\!=\\! \\mathbf{A} \\odot F_{rgb}$", ha="center", va="center", fontsize=8, color="#555")

    # --- Decoupled Optimization (right) ---
    ax.add_patch(plt.Rectangle((9.5, 2.0), 4.0, 2.5, facecolor=c_sky, alpha=0.12, edgecolor=c_sky, linewidth=2))
    ax.text(11.5, 4.0, "Regional Decoupled\nFine-Tuning", ha="center", va="center", fontsize=11, fontweight="bold", color=c_sky)
    ax.text(11.5, 3.3, "Sky Mask ($\\tau \\!<\\! 0.1$)", ha="center", va="center", fontsize=8, color="#555")
    ax.text(11.5, 2.9, "$\\mathcal{L}_{TV} \\!=\\! \\sum (1 - \\mathcal{M}) \\|\\nabla \\hat{I}\\|_1$", ha="center", va="center", fontsize=8, color="#555")
    ax.text(11.5, 2.5, "$\\mathcal{L}_{total} \\!=\\! \\mathcal{L}_{rec} \\!+\\! 0.01\\mathcal{L}_{TV}$", ha="center", va="center", fontsize=8, color="#555")

    # --- Input / Output ---
    ax.text(2.5, 1.5, "Hazy Input $I$", ha="center", fontsize=10, fontweight="bold")
    ax.text(11.5, 1.5, "Restored $\\hat{I}$", ha="center", fontsize=10, fontweight="bold", color=c_sky)

    # --- Arrows ---
    # input -> RGB
    ax.annotate("", xy=(2.0, 2.0), xytext=(2.0, 1.7), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))
    # input -> DAV2
    ax.annotate("", xy=(2.0, 4.8), xytext=(2.0, 1.7), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))
    # DAV2 -> DGCCA
    ax.annotate("", xy=(5.2, 4.5), xytext=(4.5, 5.25), arrowprops=dict(arrowstyle="->", color=c_depth, lw=1.5))
    ax.text(4.85, 5.0, "$D$", fontsize=9, color=c_depth)
    # RGB -> DGCCA
    ax.annotate("", xy=(5.2, 3.5), xytext=(4.5, 3.5), arrowprops=dict(arrowstyle="->", color=c_rgb, lw=1.5))
    ax.text(4.85, 3.7, "$F_{rgb}$", fontsize=9, color=c_rgb)
    # DGCCA -> Right
    ax.annotate("", xy=(9.5, 3.5), xytext=(8.8, 3.5), arrowprops=dict(arrowstyle="->", color=c_cca, lw=1.5))
    ax.text(9.15, 3.7, "$F_{out}$", fontsize=9, color=c_cca)
    # Right -> output
    ax.annotate("", xy=(11.5, 2.0), xytext=(11.5, 1.7), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))

    # Legend
    ax.text(0.5, 0.4, "Stage 1 (100 epochs): Global Training    |    Stage 2 (20 epochs): Regional Decoupled Fine-Tuning",
            fontsize=9, color="#777")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "architecture.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
#  Fig.3: Visual Comparison (O-HAZE: 5 methods)
# ============================================================
def generate_fig3_visual_comparison(models_dict, processor, estimator):
    print("\n" + "=" * 60)
    print("[Fig.3] Visual Comparison: O-HAZE 5 methods")
    print("=" * 60)

    # 选 O-HAZE 第一张
    hazy_dir = os.path.join(BASE_DIR, "data", "O-HAZE", "hazy")
    gt_dir = os.path.join(BASE_DIR, "data", "O-HAZE", "GT")
    hazy_name = "01_outdoor.jpg"  # O-HAZE uses .jpg
    hazy_path = os.path.join(hazy_dir, hazy_name)
    gt_path = os.path.join(gt_dir, "01_outdoor.JPG")  # GT is .JPG
    # fallback if .JPG not found
    if not os.path.exists(gt_path):
        gt_path = os.path.join(gt_dir, "01_outdoor.jpg")
    if not os.path.exists(gt_path):
        # glob search
        import glob as _g
        candidates = _g.glob(os.path.join(gt_dir, "01_outdoor.*"))
        gt_path = candidates[0] if candidates else None

    hazy_pil = Image.open(hazy_path).convert("RGB")
    gt_pil = Image.open(gt_path).convert("RGB")
    w, h = hazy_pil.size
    nw, nh = (w // 16) * 16, (h // 16) * 16
    hazy_pil = hazy_pil.resize((nw, nh), Image.LANCZOS)
    gt_pil = gt_pil.resize((nw, nh), Image.LANCZOS)

    hazy_tensor = TF.to_tensor(hazy_pil).unsqueeze(0).to(DEVICE)
    depth_map = compute_dav2_depth(hazy_pil, processor, estimator, nh, nw)

    hazy_np = np.array(hazy_pil)
    gt_np = np.array(gt_pil)

    # 1. Baseline
    with torch.no_grad():
        out_base = models_dict["A-Baseline"](hazy_tensor)
    base_np = to_np(out_base)

    # 2. Ours
    with torch.no_grad():
        out_ours = models_dict["D-SkyTV"](hazy_tensor, depth_map)
    ours_np = to_np(out_ours)

    # 3. CPLIR (已有结果, 读取)
    cplir_path = os.path.join(BASE_DIR, "CPLIR", "results_cpl_ohaze", hazy_name)
    cplir_np = np.array(Image.open(cplir_path).resize((nw, nh), Image.LANCZOS)) if os.path.exists(cplir_path) else None

    # 4. C2PNet
    c2p_path = os.path.join(BASE_DIR, "C2PNet", "results_ohaze", hazy_name)
    c2p_np = np.array(Image.open(c2p_path).resize((nw, nh), Image.LANCZOS)) if os.path.exists(c2p_path) else None

    # 5. DehazeFormer
    df_path = os.path.join(BASE_DIR, "DehazeFormer", "ohaze_results", hazy_name)
    df_np = np.array(Image.open(df_path).resize((nw, nh), Image.LANCZOS)) if os.path.exists(df_path) else None

    panels = [
        add_label(hazy_np, "Hazy Input"),
        add_label(base_np, "FFA-Net (Baseline)", color=(255, 100, 100)),
    ]
    if cplir_np is not None:
        panels.append(add_label(cplir_np, "CPLIR (CVPR 2024)"))
    if c2p_np is not None:
        panels.append(add_label(c2p_np, "C2PNet (CVPR 2023)"))
    if df_np is not None:
        panels.append(add_label(df_np, "DehazeFormer (2023)"))
    panels.append(add_label(ours_np, "Ours (DG-CCA+DAV2+SkyTV)", color=(100, 200, 100)))
    panels.append(add_label(gt_np, "Ground Truth"))

    combined = hstack_images(panels, gap=6)

    out_path = os.path.join(OUT_DIR, "visual_comparison.pdf")
    Image.fromarray(combined).save(out_path)
    print(f"  Saved: {out_path}")


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating ALL paper figures...")
    print("=" * 60)

    models_dict, depth_processor, depth_estimator = load_model_and_depth_estimator()

    # Fig.2: Architecture (no models needed)
    generate_fig2_architecture()

    # Fig.4: Mask Ablation
    generate_fig4_mask_ablation(models_dict, depth_processor, depth_estimator)

    # Fig.1: Teaser
    generate_fig1_teaser(models_dict, depth_processor, depth_estimator)

    # Fig.3: Visual Comparison
    generate_fig3_visual_comparison(models_dict, depth_processor, depth_estimator)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUT_DIR}")
    print("=" * 60)