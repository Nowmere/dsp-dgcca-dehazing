"""Figure 2 — Neural Network Style Architecture Overview
Features:
- 3D-like tensor blocks for feature maps
- Color-coded processing stages
- Clean DG-CCA detail with SE-style channel gating
- Two-stage training strategy on the right
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "CVPR_2026_Submission_Template" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1a1a2e"
GRAY = "#556677"
# Color scheme
BLUE_BG    = "#d6e6f5"
BLUE_EDGE  = "#2e609a"
BLUE_DARK  = "#1a4a7a"
GREEN_BG   = "#d4ead8"
GREEN_EDGE = "#3d7f53"
GREEN_DARK = "#2a5c3a"
PINK_BG    = "#fce4ec"
PINK_EDGE  = "#b71c50"
PINK_DARK  = "#8a1030"
PURPLE_BG  = "#e8e0f0"
PURPLE_EDGE= "#5e35b1"
PURPLE_DARK= "#3d2080"
ORANGE_BG  = "#fff3e0"
ORANGE_EDGE= "#d84315"
PANEL_BG   = "#f7f9fc"
PANEL_EDGE = "#c4cad4"
WHITE      = "#ffffff"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
    "savefig.facecolor": "white",
})


def tensor_block(ax, cx, cy, w, h, label, sub=None, *,
                 bg=BLUE_BG, edge=BLUE_EDGE, lw=2.0, fs=9.0, fs2=7.2, z=5):
    """Draw a 3D-style tensor block (flat rounded rectangle with shadow accent)."""
    # Main block
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.08,rounding_size=0.06",
                       facecolor=bg, edgecolor=edge, linewidth=lw,
                       zorder=z)
    ax.add_patch(p)
    # Subtle 3D accent (right+top edge highlight)
    p2 = FancyBboxPatch((cx - w/2 + 0.02, cy - h/2 + 0.02), w - 0.04, h - 0.04,
                        boxstyle="round,pad=0.08,rounding_size=0.06",
                        facecolor="none", edgecolor="white", linewidth=0.5,
                        zorder=z + 1)
    ax.add_patch(p2)
    if sub:
        ax.text(cx, cy + 0.08, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=INK, zorder=z + 2)
        ax.text(cx, cy - 0.14, sub, ha="center", va="center",
                fontsize=fs2, color=GRAY, zorder=z + 2)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=INK, zorder=z + 2)


def dashed_block(ax, cx, cy, w, h, label, sub=None, *,
                 bg=GREEN_BG, edge=GREEN_EDGE, lw=2.0, fs=9.0, fs2=7.2, z=5):
    """Dashed tensor block for frozen modules."""
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.08,rounding_size=0.06",
                       facecolor=bg, edgecolor=edge, linewidth=lw,
                       linestyle=(0, (5, 3)), zorder=z)
    ax.add_patch(p)
    if sub:
        ax.text(cx, cy + 0.08, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=INK, zorder=z + 2)
        ax.text(cx, cy - 0.14, sub, ha="center", va="center",
                fontsize=fs2, color=GRAY, zorder=z + 2)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=INK, zorder=z + 2)


def arrow(ax, x0, y0, x1, y1, color=BLUE_EDGE, lw=1.5, z=4):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=14, linewidth=lw, color=color,
                        shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)


def path_arrow(ax, pts, color=BLUE_EDGE, lw=1.0, z=3):
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                color=color, linewidth=lw, solid_capstyle="round", zorder=z)
    arrow(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color, lw=lw, z=z)


def plus_node(ax, x, y):
    c = Circle((x, y), 0.16, facecolor=WHITE, edgecolor=BLUE_EDGE, lw=1.5, zorder=6)
    ax.add_patch(c)
    ax.text(x, y, "+", ha="center", va="center", fontsize=13,
            fontweight="bold", color=INK, zorder=7)


def text(ax, x, y, s, fs=8.0, c=INK, b=False, ha="left", va="center", z=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs,
            fontweight="bold" if b else "normal", color=c, zorder=z)


# =====================================================================
def draw():
    W, H = 14.8, 7.8
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal"); ax.axis("off")

    text(ax, 0.25, H - 0.25,
         "Figure 2.  Overall architecture of the proposed geometry-aware physical dehazing framework.",
         fs=11.5, b=True)

    # =========== TOP: RGB Pipeline (y=6.10) ===========
    ya = 6.10
    x_boxes = [0.85, 2.75, 5.05, 7.70, 10.15, 12.20, 13.35]
    w_boxes = [1.30, 1.35, 1.60, 2.00, 1.35, 0.00, 1.10]

    # Hazy Input
    tensor_block(ax, x_boxes[0], ya, w_boxes[0], 0.72, "Hazy Input", r"$I$", bg=BLUE_BG, edge=BLUE_EDGE, fs=9.2)
    # RGB Encoder
    tensor_block(ax, x_boxes[1], ya, w_boxes[1], 0.72, "RGB Encoder", r"$3\to 64$", bg=BLUE_BG, edge=BLUE_EDGE, fs=9.0)
    # DG-CCA (PINK — fusion module)
    tensor_block(ax, x_boxes[2], ya, w_boxes[2], 0.80, "DG-CCA", bg=PINK_BG, edge=PINK_EDGE, lw=2.5, fs=10.5)
    # FFA Backbone (PURPLE)
    tensor_block(ax, x_boxes[3], ya, w_boxes[3], 0.80, "FFA Backbone", "3 Groups × 3 PA Blocks",
                 bg=PURPLE_BG, edge=PURPLE_EDGE, lw=2.2, fs=9.2)
    # Reconstruct
    tensor_block(ax, x_boxes[4], ya, w_boxes[4], 0.72, "Reconstruct", r"$3\times 3$ Conv", bg=BLUE_BG, edge=BLUE_EDGE, fs=9.0)
    # ⊕ node
    plus_node(ax, x_boxes[5], ya)
    # Restored
    tensor_block(ax, x_boxes[6], ya, w_boxes[6], 0.72, "Restored", r"$\hat I$", bg=BLUE_BG, edge=BLUE_EDGE, fs=9.2)

    # Horizontal arrows
    arrows = [
        (x_boxes[0]+w_boxes[0]/2, x_boxes[1]-w_boxes[1]/2, BLUE_EDGE),
        (x_boxes[1]+w_boxes[1]/2, x_boxes[2]-w_boxes[2]/2, BLUE_EDGE),
        (x_boxes[2]+w_boxes[2]/2, x_boxes[3]-w_boxes[3]/2, PINK_EDGE),
        (x_boxes[3]+w_boxes[3]/2, x_boxes[4]-w_boxes[4]/2, PURPLE_EDGE),
        (x_boxes[4]+w_boxes[4]/2, x_boxes[5]-0.18, BLUE_EDGE),
        (x_boxes[5]+0.18, x_boxes[6]-w_boxes[6]/2, BLUE_EDGE),
    ]
    for x0, x1, ec in arrows:
        arrow(ax, x0, ya, x1, ya, color=ec)

    # Global residual
    path_arrow(ax, [
        (x_boxes[0] - w_boxes[0]/2 - 0.05, ya + 0.38),
        (x_boxes[0] - w_boxes[0]/2 - 0.05, H - 0.55),
        (x_boxes[5], H - 0.55),
        (x_boxes[5], ya + 0.40)
    ], color=BLUE_EDGE, lw=0.9, z=1)
    text(ax, 6.35, H - 0.51, "Global Residual", fs=7.5, c=BLUE_EDGE, ha="center", va="bottom")

    # =========== MIDDLE: Depth Branch (y=4.50) ===========
    yb = 4.50
    dashed_block(ax, 2.10, yb, 2.10, 0.78, "Depth Anything V2", "Frozen · 24.8M params",
                 bg=GREEN_BG, edge=GREEN_EDGE, fs=9.2, fs2=7.3)
    tensor_block(ax, 4.65, yb, 1.70, 0.78, "Depth Prior", "Min-Max + Adapter",
                 bg=GREEN_BG, edge=GREEN_EDGE, fs=9.0, fs2=7.2)
    path_arrow(ax, [(x_boxes[0], ya - 0.38), (x_boxes[0], yb + 0.40), (1.05, yb + 0.40)],
               color=GREEN_EDGE, lw=1.2)
    arrow(ax, 3.15, yb, 3.80, yb, color=GREEN_EDGE)
    # F_d vertical arrow
    arrow(ax, 5.50, yb + 0.39, 5.50, ya - 0.38, color=GREEN_EDGE, lw=1.4)
    text(ax, 5.75, (yb + ya)/2, r"$F_d$", fs=8.5, c=GREEN_EDGE, b=True)

    # =========== BOTTOM PANEL ===========
    px, py, pw, ph = 0.25, 0.25, 14.30, 3.20
    ptop = py + ph
    ax.add_patch(FancyBboxPatch((px, py), pw, ph,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor=PANEL_BG, edgecolor=PANEL_EDGE, linewidth=0.9, zorder=0))
    # Two column titles
    text(ax, px + 0.25, ptop - 0.20, "(a) DG-CCA Module Detail", fs=9.2, b=True)
    text(ax, 8.70, ptop - 0.20, "(b) Training Strategy", fs=9.2, b=True)

    # ----- DG-CCA Detail Grid -----
    rA = py + 2.70   # 2.95  F_rgb / F_d
    rB = py + 1.85   # 2.10  z_rgb / z_d
    rC = py + 1.10   # 1.35  Concat
    rD = py + 0.45   # 0.70  MLP
    rE = py + 0.12   # 0.37  Output

    hA = 0.52
    hB = 0.44
    hC = 0.46
    hD = 0.56

    # F_rgb & F_d (row A)
    tensor_block(ax, 1.30, rA, 1.55, hA, r"$F_{rgb}$", r"$C\!\times\!H\!\times\!W$",
                 bg=BLUE_BG, edge=BLUE_EDGE, lw=1.2, fs=8.8, fs2=7.2)
    tensor_block(ax, 6.90, rA, 1.55, hA, r"$F_d$", r"$C\!\times\!H\!\times\!W$",
                 bg=GREEN_BG, edge=GREEN_EDGE, lw=1.2, fs=8.8, fs2=7.2)

    # GAP labels vertically placed
    text(ax, 3.05, (rA + rB)/2, "GAP", fs=7.3, c=GRAY, b=True, ha="left")
    text(ax, 5.05, (rA + rB)/2, "GAP", fs=7.3, c=GRAY, b=True, ha="right")

    arrow(ax, 3.05, rA - hA/2, 3.05, rB + hB/2, color=GRAY, lw=1.0)
    arrow(ax, 5.05, rA - hA/2, 5.05, rB + hB/2, color=GRAY, lw=1.0)

    # z_rgb & z_d (row B)
    tensor_block(ax, 3.05, rB, 1.30, hB, r"$z_{rgb}$", r"$\mathbb{R}^{C}$",
                 bg="#eef4ff", edge=BLUE_EDGE, lw=1.0, fs=7.8, fs2=6.5)
    tensor_block(ax, 5.05, rB, 1.30, hB, r"$z_d$", r"$\mathbb{R}^{C}$",
                 bg="#edf7ee", edge=GREEN_EDGE, lw=1.0, fs=7.8, fs2=6.5)

    # z → Concat (row C)
    arrow(ax, 3.05, rB - hB/2, 3.05, rC + hC/2, color=GRAY, lw=0.9)
    arrow(ax, 5.05, rB - hB/2, 5.05, rC + hC/2, color=GRAY, lw=0.9)

    tensor_block(ax, 4.05, rC, 2.80, hC, r"Concat $[z_{rgb};\,z_d]$", r"$\mathbb{R}^{2C}$",
                 bg=PINK_BG, edge=PINK_EDGE, lw=1.5, fs=8.2, fs2=6.6)

    # MLP (row D)
    arrow(ax, 4.05, rC - hC/2, 4.05, rD + hD/2, color=PINK_EDGE, lw=1.0)
    tensor_block(ax, 4.05, rD, 4.80, hD, "Bottleneck MLP",
                 r"$W_1\!\in\!\mathbb{R}^{C\times 2C} \to \mathrm{ReLU} \to W_2\!\in\!\mathbb{R}^{C\times C} \to \sigma$",
                 bg=PINK_BG, edge=PINK_EDGE, lw=2.0, fs=8.8, fs2=6.5)

    # Output (row E)
    ox = 4.05 + 2.40  # MLP right edge
    arrow(ax, ox, rD - hD/2, ox, rE + 0.10, color=PINK_EDGE, lw=1.0)
    text(ax, ox + 0.35, rE, r"$\mathbf{A}\!\in\!\mathbb{R}^{C}$  (channel gate)", fs=8.2, c=PINK_EDGE, b=True)
    arrow(ax, ox + 2.80, rE, ox + 3.80, rE, color=PINK_EDGE, lw=1.0)
    text(ax, ox + 4.80, rE, r"$F_{out} = \mathbf{A} \odot F_{rgb}$", fs=9.5, c=INK, b=True, ha="center")

    # ----- Training text (right) -----
    tx = 8.70; ty = ptop - 0.45; dd = 0.30
    lines = [
        ("Two-Stage Training Curriculum", 9.5, True, INK),
        ("Stage 1 (Epoch 1–100)", 8.5, True, INK),
        ("  All parameters jointly optimized", 7.8, False, GRAY),
        ("  " + r"$\mathcal{L}_{rec} = \|\hat I - J\|_1 + 0.05\,\mathcal{L}_{edge}$", 8.5, True, PINK_EDGE),
        ("Stage 2 (Epoch 101–120)", 8.5, True, INK),
        ("  Backbone frozen · Only DG-CCA + Adapter trainable", 7.8, False, GRAY),
        ("  " + r"$M(x,y)=\mathbf{1}[D(x,y)\geq 0.1]$   (Hard sky mask, $\tau=0.1$)", 8.0, False, INK),
        ("  " + r"$\mathcal{L}_{TV} = \sum_{x,y}(1-M)(\|\nabla_x\hat I\|_1 + \|\nabla_y\hat I\|_1)$", 7.8, False, INK),
        ("  " + r"$\mathcal{L}_{total} = \mathcal{L}_{rec} + 0.01\,\mathcal{L}_{TV}$", 8.8, True, PINK_EDGE),
    ]
    for i, (s, fs, bld, clr) in enumerate(lines):
        text(ax, tx, ty - i * dd, s, fs=fs, c=clr, b=bld)

    # SAVE
    for fmt, dpi in [("pdf", None), ("png", 400)]:
        fp = OUT / f"fig2_nnstyle.{fmt}"
        fig.savefig(fp, bbox_inches="tight", pad_inches=0.06, dpi=dpi)
        print(f"Saved: {fp}")
    plt.close(fig)


if __name__ == "__main__":
    draw()