"""Figure 2 Version 1: 水平流线型 (Horizontal Flow Architecture)
简洁清晰的水平数据流布局，突出两条并行分支。

Author: Architecture Visualization
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper" / "CVPR_2026_Submission_Template" / "figures"
OUTPUT_STEM = OUTPUT_DIR / "fig2_v1_horizontal_flow"

# ── Color Palette ──
INK = "#1a1a2e"
MUTED = "#555f6e"
RGB_FILL = "#e8f0fe"
RGB_EDGE = "#3b5998"
DEPTH_FILL = "#e6f5e6"
DEPTH_EDGE = "#4a8f5c"
FUSION_FILL = "#fce4ec"
FUSION_EDGE = "#c62858"
BACKBONE_FILL = "#ede7f6"
BACKBONE_EDGE = "#5e35b1"
LOSSFILL = "#fff3e0"
LOSS_EDGE = "#e65100"
PANEL_FILL = "#f5f7fa"
PANEL_EDGE = "#bac0c9"
SKY_FILL = "#e3f2fd"
SKY_EDGE = "#1565c0"

def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavuserif",
        "savefig.facecolor": "white",
        "figure.dpi": 200,
    })

def add_box(ax, x, y, w, h, label, sublabel=None, *, fill=RGB_FILL, edge=RGB_EDGE,
            lw=1.5, dashed=False, fontsize=8.5, subfontsize=7.0, bold=True, pad=0.2):
    ls = (0, (4, 2.5)) if dashed else "solid"
    patch = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad={pad},rounding_size=0.08",
        facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=4
    )
    ax.add_patch(patch)
    if sublabel:
        ax.text(x, y + 0.10, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=INK, zorder=5)
        ax.text(x, y - 0.14, sublabel, ha="center", va="center",
                fontsize=subfontsize, color=MUTED, zorder=5)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=INK, zorder=5)

def add_arrow(ax, x0, y0, x1, y1, color=RGB_EDGE, lw=1.2, zorder=3, dashed=False):
    ls = (0, (4, 2.5)) if dashed else "solid"
    patch = FancyArrowPatch((x0, y0), (x1, y1),
                            arrowstyle="-|>", mutation_scale=12,
                            linewidth=lw, linestyle=ls, color=color,
                            shrinkA=0, shrinkB=0, zorder=zorder)
    ax.add_patch(patch)

def add_curved_arrow(ax, points, color=RGB_EDGE, lw=1.0, zorder=2):
    """Draw polyline via a list of (x,y). Last segment gets arrowhead."""
    for i in range(len(points) - 2):
        ax.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]],
                color=color, linewidth=lw, solid_capstyle="round", zorder=zorder)
    # arrowhead on final segment
    add_arrow(ax, points[-2][0], points[-2][1], points[-1][0], points[-1][1],
              color=color, lw=lw, zorder=zorder)

def add_panel(ax, x, y, w, h, title):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.01,rounding_size=0.06",
                           facecolor=PANEL_FILL, edgecolor=PANEL_EDGE,
                           linewidth=1.0, zorder=0)
    ax.add_patch(patch)
    ax.text(x + 0.25, y + h - 0.22, title, ha="left", va="center",
            fontsize=9.0, fontweight="bold", color=INK, zorder=6)

def add_plus(ax, x, y):
    c = Circle((x, y), 0.18, facecolor="white", edgecolor=RGB_EDGE, linewidth=1.3, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, "+", ha="center", va="center", fontsize=13, fontweight="bold", color=INK, zorder=6)

# =====================================================================
# MAIN DRAWING
# =====================================================================
def draw():
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.0, 6.8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # ─── TITLE ───
    ax.text(0.3, 6.50, "Figure 2. Overall architecture of the proposed geometry-aware physical dehazing framework.",
            fontsize=12.0, fontweight="bold", color=INK, zorder=6)

    # ─── UPPER TRACK: RGB FLOW (y ≈ 4.8) ───
    y_up = 4.8

    # Hazy Input
    add_box(ax, 0.80, y_up, 1.20, 0.72, "Hazy Input", r"$I \in \mathbb{R}^{3\times H\times W}$",
            fill=RGB_FILL, edge=RGB_EDGE, fontsize=8.8, subfontsize=7.0)

    # RGB Encoder
    add_box(ax, 2.90, y_up, 1.30, 0.72, "RGB Encoder", r"$3 \rightarrow 64$",
            fill=RGB_FILL, edge=RGB_EDGE, fontsize=8.8, subfontsize=7.0)
    add_arrow(ax, 1.40, y_up, 2.25, y_up, RGB_EDGE, 1.2)

    # DG-CCA (fusion)
    add_box(ax, 5.30, y_up, 1.70, 0.82, "DG-CCA", "Channel Fusion",
            fill=FUSION_FILL, edge=FUSION_EDGE, lw=2.0, fontsize=10.0, subfontsize=7.3)
    add_arrow(ax, 3.55, y_up, 4.45, y_up, RGB_EDGE, 1.2)

    # FFA Backbone
    add_box(ax, 7.80, y_up, 2.10, 0.82, "FFA Backbone", "3 Groups × 3 PA Blocks",
            fill=BACKBONE_FILL, edge=BACKBONE_EDGE, fontsize=9.0, subfontsize=7.2)
    add_arrow(ax, 6.15, y_up, 6.75, y_up, FUSION_EDGE, 1.2)

    # Reconstruct
    add_box(ax, 10.70, y_up, 1.30, 0.72, "Reconstruct", r"$3\times 3$ Conv",
            fill=RGB_FILL, edge=RGB_EDGE, fontsize=8.8, subfontsize=7.0)
    add_arrow(ax, 8.85, y_up, 10.05, y_up, BACKBONE_EDGE, 1.2)

    # ⊕ node
    add_plus(ax, 12.20, y_up)

    # Restored Output
    add_box(ax, 13.30, y_up, 1.10, 0.72, "Restored", r"$\hat I = I + R(I)$",
            fill=RGB_FILL, edge=RGB_EDGE, fontsize=8.8, subfontsize=7.0)
    add_arrow(ax, 12.38, y_up, 12.75, y_up, RGB_EDGE, 1.2)

    # Global residual arrow (below RGB track)
    add_curved_arrow(ax,
        [(1.20, y_up + 0.36), (1.20, 6.20), (12.20, 6.20), (12.20, y_up + 0.60)],
        color=RGB_EDGE, lw=1.0, zorder=2)
    ax.text(6.70, 6.26, "Global Residual Connection", fontsize=7.2, color=RGB_EDGE,
            ha="center", va="bottom", zorder=5)

    # ─── LOWER TRACK: DEPTH FLOW (y ≈ 2.8) ───
    y_dn = 2.8

    # Frozen DAV2
    add_box(ax, 2.00, y_dn, 2.00, 0.78, "Depth Anything V2", "Frozen · 24.8M Params",
            fill=DEPTH_FILL, edge=DEPTH_EDGE, lw=1.5, dashed=True, fontsize=9.0, subfontsize=7.0)

    # Depth branch from hazy input
    add_curved_arrow(ax,
        [(0.80, y_up - 0.36), (0.80, y_dn + 0.39), (1.00, y_dn + 0.39)],
        color=DEPTH_EDGE, lw=1.2, zorder=2)
    ax.text(0.55, 3.90, "share", fontsize=7.0, color=DEPTH_EDGE,
            ha="center", rotation=90, zorder=5)

    # Depth Prior
    add_box(ax, 4.10, y_dn, 1.50, 0.78, "Depth Prior", "Min-Max Norm",
            fill=DEPTH_FILL, edge=DEPTH_EDGE, fontsize=8.5, subfontsize=7.0)
    add_arrow(ax, 3.00, y_dn, 3.35, y_dn, DEPTH_EDGE, 1.2)

    # Depth Encoder
    add_box(ax, 6.00, y_dn, 1.40, 0.78, "Depth Encoder", r"$1\times 1$ Conv: $1\rightarrow 64$",
            fill=DEPTH_FILL, edge=DEPTH_EDGE, fontsize=8.3, subfontsize=7.0)
    add_arrow(ax, 4.85, y_dn, 5.30, y_dn, DEPTH_EDGE, 1.2)

    # Injection arrow: depth → DG-CCA
    add_curved_arrow(ax,
        [(6.70, y_dn + 0.39), (6.85, y_dn + 0.39), (6.85, 3.70),
         (5.30, 4.39)],
        color=DEPTH_EDGE, lw=1.3, zorder=2)
    ax.text(6.98, 3.70, "Depth Feature $F_d$", fontsize=7.0, color=DEPTH_EDGE,
            ha="left", va="center", zorder=5)

    # ─── PANEL A: DG-CCA DETAIL (bottom-left) ───
    py = 0.35
    add_panel(ax, 0.30, py, 8.80, 2.00, "DG-CCA Detail: Depth-Guided Cross-modal Channel Attention")

    # F_rgb
    bx = 0.80
    add_box(ax, bx, py + 1.20, 1.50, 0.52, r"$F_{rgb}$", r"$C\!\times\!H\!\times\!W$",
            fill=RGB_FILL, edge=RGB_EDGE, fontsize=8.8, subfontsize=7.0)
    # F_d
    add_box(ax, bx + 5.70, py + 1.20, 1.50, 0.52, r"$F_{d}$", r"$C\!\times\!H\!\times\!W$",
            fill=DEPTH_FILL, edge=DEPTH_EDGE, fontsize=8.8, subfontsize=7.0)

    # GAP arrows & boxes
    add_arrow(ax, bx + 0.75, py + 1.20, bx + 1.25, py + 1.20, MUTED, 0.9)
    add_curve = [(bx + 1.40, py + 1.20), (bx + 1.40, py + 0.82)]
    for i in range(len(add_curve)-1):
        ax.plot([add_curve[i][0], add_curve[i+1][0]], [add_curve[i][1], add_curve[i+1][1]],
                color=MUTED, linewidth=0.9, zorder=2)
    add_arrow(ax, bx + 1.40, py + 0.94, bx + 1.40, py + 0.82, MUTED, 0.9)

    add_curve2 = [(bx + 6.45, py + 1.20), (bx + 6.45, py + 0.82)]
    for i in range(len(add_curve2)-1):
        ax.plot([add_curve2[i][0], add_curve2[i+1][0]], [add_curve2[i][1], add_curve2[i+1][1]],
                color=MUTED, linewidth=0.9, zorder=2)
    add_arrow(ax, bx + 6.45, py + 0.94, bx + 6.45, py + 0.82, MUTED, 0.9)
    add_arrow(ax, bx + 5.70, py + 1.20, bx + 6.20, py + 1.20, MUTED, 0.9)

    # GAP boxes
    add_box(ax, bx + 0.90, py + 0.62, 1.00, 0.42, r"$\mathbf{z}_{rgb}$", r"$\mathbb{R}^{C}$",
            fill="#f0f4ff", edge=RGB_EDGE, fontsize=8.0, subfontsize=6.8, lw=1.0)
    add_box(ax, bx + 5.95, py + 0.62, 1.00, 0.42, r"$\mathbf{z}_{d}$", r"$\mathbb{R}^{C}$",
            fill="#f0fef0", edge=DEPTH_EDGE, fontsize=8.0, subfontsize=6.8, lw=1.0)

    # Concat arrow
    add_arrow(ax, bx + 1.40, py + 0.62, bx + 2.60, py + 0.62, MUTED, 0.9)
    add_arrow(ax, bx + 6.45, py + 0.62, bx + 5.35, py + 0.62, MUTED, 0.9)

    # V arrows down to concat
    add_curve3 = [(bx + 2.60, py + 0.41), (bx + 2.60, py + 0.12)]
    for i in range(len(add_curve3)-1):
        ax.plot([add_curve3[i][0], add_curve3[i+1][0]], [add_curve3[i][1], add_curve3[i+1][1]],
                color=MUTED, linewidth=0.9, zorder=2)
    add_arrow(ax, bx + 2.60, py + 0.24, bx + 2.60, py + 0.12, MUTED, 0.9)

    add_curve4 = [(bx + 5.35, py + 0.41), (bx + 5.35, py + 0.12)]
    for i in range(len(add_curve4)-1):
        ax.plot([add_curve4[i][0], add_curve4[i+1][0]], [add_curve4[i][1], add_curve4[i+1][1]],
                color=MUTED, linewidth=0.9, zorder=2)
    add_arrow(ax, bx + 5.35, py + 0.24, bx + 5.35, py + 0.12, MUTED, 0.9)

    # Concat box
    add_box(ax, bx + 3.60, py - 0.15, 2.40, 0.55, r"Concat $[\mathbf{z}_{rgb};\,\mathbf{z}_{d}]$",
            r"$\mathbb{R}^{2C}$", fill=FUSION_FILL, edge="#d81b60", fontsize=8.0, subfontsize=6.8, lw=1.3)

    # MLP (bottleneck)
    add_arrow(ax, bx + 4.80, py + 0.12, bx + 4.80, py - 0.50, FUSION_EDGE, 0.9)
    add_box(ax, bx + 3.40, py - 0.80, 2.80, 0.62, r"Bottleneck MLP",
            r"$W_1\in\mathbb{R}^{C\times 2C},\, \mathrm{ReLU},\, W_2\in\mathbb{R}^{C\times C},\, \sigma$",
            fill="#fff0f5", edge=FUSION_EDGE, fontsize=8.6, subfontsize=6.6, lw=1.5)

    # Attention output
    add_arrow(ax, bx + 4.80, py - 1.11, bx + 4.80, py - 1.45, FUSION_EDGE, 0.9)
    ax.text(bx + 5.20, py - 1.28, r"$\mathbf{A}\!\in\!\mathbb{R}^{C}$", fontsize=8.0, color=FUSION_EDGE,
            ha="left", va="center", fontweight="bold", zorder=5)
    ax.text(bx + 5.20, py - 1.48, "channel-wise gate", fontsize=7.0, color=MUTED,
            ha="left", va="center", zorder=5)

    # Multiply symbol
    add_arrow(ax, bx + 6.40, py - 1.28, bx + 6.90, py - 1.28, FUSION_EDGE, 0.9)
    ax.text(bx + 7.45, py - 1.28, r"$F_{out} = \mathbf{A} \odot F_{rgb}$", fontsize=9.0, color=INK,
            ha="center", va="center", fontweight="bold", zorder=5)

    ax.text(bx + 4.00, py - 1.72, r"Per module: $2C^2$$+$$2C = 8{,}448$ parameters, $C=64$",
            fontsize=7.2, color=MUTED, ha="center", zorder=5)

    # ─── PANEL B: SKY TV LOSS & TRAINING ───
    sx = 9.40
    sw = 5.30
    add_panel(ax, sx, py, sw, 2.00, "Regional Decoupled Fine-Tuning (Epoch 101–120)")

    # Stage 1
    ax.text(sx + 0.30, py + 1.70, "Stage 1 (Epoch 1–100)", fontsize=8.5, fontweight="bold", color=INK, zorder=5)
    ax.text(sx + 0.40, py + 1.43, r"$\mathcal{L}_{rec} = \|\hat I - J\|_1 + 0.05\,\mathcal{L}_{edge}$",
            fontsize=8.8, color=LOSS_EDGE, zorder=5)
    ax.text(sx + 0.40, py + 1.20, "All parameters trainable · DG-CCA fully active",
            fontsize=7.0, color=MUTED, zorder=5)

    # Stage 2
    ax.text(sx + 0.30, py + 0.95, "Stage 2 (Epoch 101–120)", fontsize=8.5, fontweight="bold", color=INK, zorder=5)
    ax.text(sx + 0.40, py + 0.68, r"$\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda\,\mathcal{L}_{TV},\quad \lambda=0.01$",
            fontsize=8.5, color=INK, zorder=5)
    ax.text(sx + 0.40, py + 0.48, "Backbone frozen · Only DG-CCA + Adapter trainable",
            fontsize=7.0, color=MUTED, zorder=5)

    # Sky TV formula
    ax.text(sx + 0.40, py + 0.22, "Hard Mask & TV Loss:",
            fontsize=7.8, fontweight="bold", color=INK, zorder=5)
    ax.text(sx + 0.40, py - 0.02, r"$M(x,y)=\mathbf{1}[D(x,y)\!\geq\!\tau],\;\tau=0.1$",
            fontsize=7.8, color=SKY_EDGE, fontweight="bold", zorder=5)
    ax.text(sx + 0.40, py - 0.24, r"$\mathcal{L}_{TV}=\sum_{x,y}(1\!-\!M)(\|\nabla_x\hat I\|_1+\|\nabla_y\hat I\|_1)$",
            fontsize=7.8, color=LOSS_EDGE, fontweight="bold", zorder=5)

    # Save
    for fmt, dpi in [("pdf", None), ("png", 360)]:
        path = OUTPUT_STEM.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08, dpi=dpi)
        print(f"Saved: {path}")
    plt.close(fig)

if __name__ == "__main__":
    draw()