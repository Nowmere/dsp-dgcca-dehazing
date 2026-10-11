"""Figure 2: Final clean version. Minimal elements, precise layout.
Main pipeline (top) + depth branch (middle) + DG-CCA detail (bottom panel).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "CVPR_2026_Submission_Template" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ──
INK = "#1a1a2e"
GRAY = "#556677"
BLUE_FILL = "#e3ecf7"
BLUE_EDGE = "#2e609a"
GREEN_FILL = "#e2f0e5"
GREEN_EDGE = "#3d7f53"
PINK_FILL = "#fce4ec"
PINK_EDGE = "#b71c50"
PURPLE_FILL = "#ede7f6"
PURPLE_EDGE = "#5e35b1"
PANEL_FILL = "#f7f9fc"
PANEL_EDGE = "#c4cad4"
WHITE = "#ffffff"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
    "savefig.facecolor": "white",
})


def bbox(ax, x, y, w, h, t1, t2=None, *, fill=BLUE_FILL, edge=BLUE_EDGE, lw=1.5,
         dash=False, fs1=9.0, fs2=7.2, fw1="bold", c1=INK, c2=GRAY):
    ls = (0, (4, 3)) if dash else "solid"
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.12,rounding_size=0.07",
                       facecolor=fill, edgecolor=edge, linewidth=lw,
                       linestyle=ls, zorder=5)
    ax.add_patch(p)
    if t2:
        ax.text(x, y + 0.10, t1, ha="center", va="center",
                fontsize=fs1, fontweight=fw1, color=c1, zorder=6)
        ax.text(x, y - 0.14, t2, ha="center", va="center",
                fontsize=fs2, color=c2, zorder=6)
    else:
        ax.text(x, y, t1, ha="center", va="center",
                fontsize=fs1, fontweight=fw1, color=c1, zorder=6)


def arr(ax, x0, y0, x1, y1, color=BLUE_EDGE, lw=1.2, dash=False, z=4):
    ls = (0, (4, 3)) if dash else "solid"
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=11, linewidth=lw, linestyle=ls,
                        color=color, shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)


def path(ax, pts, color=BLUE_EDGE, lw=1.0, z=3):
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=color, linewidth=lw, solid_capstyle="round", zorder=z)
    arr(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1],
        color=color, lw=lw, z=z)


def oplus(ax, x, y):
    c = Circle((x, y), 0.16, facecolor=WHITE, edgecolor=BLUE_EDGE,
               linewidth=1.3, zorder=6)
    ax.add_patch(c)
    ax.text(x, y, "+", ha="center", va="center", fontsize=12,
            fontweight="bold", color=INK, zorder=7)


# =====================================================================
# MAIN
# =====================================================================
def draw():
    W, H = 14.5, 5.0
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(0.25, 4.80, "Figure 2.  Overall architecture of the proposed geometry-aware physical dehazing framework.",
            fontsize=11.5, fontweight="bold", color=INK)

    # =================================================================
    # ROW 1 — RGB Pipeline (y = 3.70)
    # =================================================================
    Y1 = 3.70

    # Box layout:  0.75, 2.60, 4.80, 7.40, 9.80, 11.40, 12.60
    bl = [0.75, 2.60, 4.80, 7.40, 9.80, 12.00, 13.10]
    bw = [1.20, 1.30, 1.55, 2.00, 1.30, 0.65, 1.05]
    bh = [0.72, 0.72, 0.78, 0.78, 0.72, 0.00, 0.72]

    bbox(ax, bl[0], Y1, bw[0], bh[0], "Hazy $I$", r"$3\times H\times W$", fill=BLUE_FILL, edge=BLUE_EDGE)
    bbox(ax, bl[1], Y1, bw[1], bh[1], "RGB Encoder", r"$3\rightarrow 64$", fill=BLUE_FILL, edge=BLUE_EDGE)
    bbox(ax, bl[2], Y1, bw[2], bh[2], "DG-CCA", fill=PINK_FILL, edge=PINK_EDGE, lw=2.0, fs1=10.5)
    bbox(ax, bl[3], Y1, bw[3], bh[3], "FFA Backbone", "3 Groups x 3 PA Blocks", fill=PURPLE_FILL, edge=PURPLE_EDGE, lw=1.8)
    bbox(ax, bl[4], Y1, bw[4], bh[4], "Reconstruct", r"$3\times 3$ Conv", fill=BLUE_FILL, edge=BLUE_EDGE)
    # bl[5] is ⊕
    oplus(ax, bl[5], Y1)
    bbox(ax, bl[6], Y1, bw[6], bh[6], "Restored", r"$\hat I$", fill=BLUE_FILL, edge=BLUE_EDGE)

    # Horizontal arrows
    arr(ax, 1.35, Y1, 1.95, Y1, BLUE_EDGE)
    arr(ax, 3.25, Y1, 4.02, Y1, BLUE_EDGE)
    arr(ax, 5.58, Y1, 6.40, Y1, PINK_EDGE)
    arr(ax, 8.40, Y1, 9.15, Y1, PURPLE_EDGE)
    arr(ax, 10.45, Y1, 11.68, Y1, BLUE_EDGE)
    arr(ax, 12.16, Y1, 12.58, Y1, BLUE_EDGE)

    # Global residual (top)
    path(ax, [
        (bl[0] - 0.60, Y1 + 0.36), (bl[0] - 0.60, 4.50),
        (bl[5], 4.50), (bl[5], Y1 + 0.66)
    ], color=BLUE_EDGE, lw=0.9, z=1)
    ax.text(6.30, 4.54, "Global Residual", fontsize=7.5, color=BLUE_EDGE,
            ha="center", va="bottom")

    # =================================================================
    # ROW 2 — Depth Branch (y = 2.20)
    # =================================================================
    Y2 = 2.20

    bbox(ax, 2.10, Y2, 2.10, 0.78, "Depth Anything V2", "Frozen (24.8M params)",
         fill=GREEN_FILL, edge=GREEN_EDGE, dash=True, fs1=9.2, fs2=7.3)
    bbox(ax, 4.60, Y2, 1.60, 0.78, "Depth Prior", "Min-Max + $1\\times 1$ Conv",
         fill=GREEN_FILL, edge=GREEN_EDGE, fs1=9.0, fs2=7.2)

    # Hazel → DAV2
    path(ax, [(bl[0], Y1 - 0.36), (bl[0], Y2 + 0.39), (1.05, Y2 + 0.39)],
         color=GREEN_EDGE, lw=1.2)
    # DAV2 → Depth Prior
    arr(ax, 3.15, Y2, 3.80, Y2, GREEN_EDGE)
    # Depth → DG-CCA (vertical injection)
    path(ax, [
        (5.40, Y2 + 0.39), (5.40, 3.05), (4.80, Y1 - 0.39)
    ], color=GREEN_EDGE, lw=1.3)
    ax.text(5.65, 3.05, r"$F_d$", fontsize=8.0, color=GREEN_EDGE,
            fontweight="bold", ha="left", va="center")

    # =================================================================
    # ROW 3 — DG-CCA Detail Panel (y = 0.30 to 1.60)
    # =================================================================
    # Panel background
    px, py, pw, ph = 0.30, 0.20, 13.90, 1.60
    p = FancyBboxPatch((px, py), pw, ph,
                       boxstyle="round,pad=0.01,rounding_size=0.05",
                       facecolor=PANEL_FILL, edgecolor=PANEL_EDGE,
                       linewidth=0.9, zorder=0)
    ax.add_patch(p)
    ax.text(px + 0.25, py + ph - 0.20, "DG-CCA Detail & Training Strategy",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=INK)

    # --- Left side: DG-CCA flow ---
    cx = 1.20  # starting X
    cy = py + 0.80  # Y for boxes

    # F_rgb
    bbox(ax, cx, cy, 1.50, 0.50, r"$F_{rgb}$", r"$C\times H\times W$",
         fill=BLUE_FILL, edge=BLUE_EDGE, lw=1.2, fs1=8.5, fs2=7.0)
    arr(ax, cx + 0.75, cy, cx + 1.30, cy, GRAY, 0.8)
    ax.text(cx + 1.90, cy, "GAP", fontsize=7.5, fontweight="bold", color=GRAY,
            ha="center", va="center")
    # GAP box (implicit via text)
    arr(ax, cx + 2.50, cy, cx + 2.50, cy + 0.20, GRAY, 0.8)
    arr(ax, cx + 2.50, cy + 0.05, cx + 2.50, cy - 0.45, GRAY, 0.8)

    # F_d
    cx2 = cx + 5.00
    bbox(ax, cx2, cy, 1.50, 0.50, r"$F_d$", r"$C\times H\times W$",
         fill=GREEN_FILL, edge=GREEN_EDGE, lw=1.2, fs1=8.5, fs2=7.0)
    arr(ax, cx2 - 0.75, cy, cx2 - 1.30, cy, GRAY, 0.8)
    ax.text(cx2 - 1.90, cy, "GAP", fontsize=7.5, fontweight="bold", color=GRAY,
            ha="center", va="center")
    arr(ax, cx2 - 2.50, cy, cx2 - 2.50, cy + 0.20, GRAY, 0.8)
    arr(ax, cx2 - 2.50, cy + 0.05, cx2 - 2.50, cy - 0.45, GRAY, 0.8)

    # z_rgb and z_d boxes (one row below)
    zy = cy - 0.65
    bbox(ax, cx + 2.50, zy, 1.00, 0.40, r"$z_{rgb}$", r"$\mathbb{R}^C$",
         fill="#f0f5ff", edge=BLUE_EDGE, lw=1.0, fs1=7.5, fs2=6.5)
    bbox(ax, cx2 - 2.50, zy, 1.00, 0.40, r"$z_d$", r"$\mathbb{R}^C$",
         fill="#eff8ef", edge=GREEN_EDGE, lw=1.0, fs1=7.5, fs2=6.5)

    # Arrows to concat
    concat_x = cx + 5.50
    arr(ax, cx + 3.00, zy, concat_x - 0.70, zy, GRAY, 0.7)
    arr(ax, cx2 - 3.00, zy, concat_x + 0.70, zy, GRAY, 0.7)

    # Concat box (below z_rgb/z_d)
    cy2 = zy - 0.32
    bbox(ax, concat_x, cy2, 2.20, 0.50, r"Concat $[z_{rgb};\,z_d]$", r"$\mathbb{R}^{2C}$",
         fill=PINK_FILL, edge=PINK_EDGE, lw=1.3, fs1=7.8, fs2=6.5)

    # Down arrow
    arr(ax, concat_x, zy - 0.20, concat_x, cy2 + 0.25, GRAY, 0.7)
    # Small vertical from z_rgb → concat, z_d → concat
    arr(ax, cx + 3.00, zy - 0.20, cx + 3.00, cy2 + 0.12, GRAY, 0.5)
    arr(ax, cx2 - 3.00, zy - 0.20, cx2 - 3.00, cy2 + 0.12, GRAY, 0.5)

    # MLP box
    cy3 = cy2 - 0.40
    bbox(ax, concat_x, cy3, 3.00, 0.53, "Bottleneck MLP",
         r"$W_1\in\mathbb{R}^{C\times 2C}\rightarrow \mathrm{ReLU}\rightarrow W_2\in\mathbb{R}^{C\times C}\rightarrow \sigma$",
         fill=PINK_FILL, edge=PINK_EDGE, lw=1.6, fs1=8.3, fs2=6.3, c2=INK)
    arr(ax, concat_x, cy2 - 0.25, concat_x, cy3 + 0.27, PINK_EDGE, 0.9)

    # Output A
    arr(ax, concat_x + 1.50, cy3 - 0.27, concat_x + 1.50, cy3 - 0.52, PINK_EDGE, 0.8)
    ax.text(concat_x + 1.90, cy3 - 0.55, r"$\mathbf{A}\in\mathbb{R}^C$",
            fontsize=8.0, color=PINK_EDGE, fontweight="bold", ha="left", va="center")

    # F_out formula
    ax.text(concat_x + 4.20, cy3 - 0.55, r"$F_{out} = \mathbf{A} \odot F_{rgb}$",
            fontsize=9.0, color=INK, fontweight="bold", ha="center", va="center")
    arr(ax, concat_x + 2.80, cy3 - 0.55, concat_x + 3.40, cy3 - 0.55, PINK_EDGE, 0.8)

    # --- Right side: Training ---
    tx = 9.00
    ty = py + 1.30
    ax.text(tx, ty, "Two-Stage Training", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(tx, ty - 0.22, "Stage 1 (Epoch 1–100):", fontsize=8.3, fontweight="bold", color=INK)
    ax.text(tx + 0.20, ty - 0.45, r"All params trainable  |  $\mathcal{L}_{rec} = L_1 + 0.05\,\mathcal{L}_{edge}$",
            fontsize=7.8, color=GRAY)
    ax.text(tx, ty - 0.75, "Stage 2 (Epoch 101–120):", fontsize=8.3, fontweight="bold", color=INK)
    ax.text(tx + 0.20, ty - 0.98, r"Backbone frozen  |  Only DG-CCA + Adapter trainable",
            fontsize=7.8, color=GRAY)
    ax.text(tx + 0.20, ty - 1.20, r"$M(x,y)=\mathbf{1}[D\geq 0.1]$  (Hard sky mask, morphological opening)",
            fontsize=7.8, color=INK)
    ax.text(tx + 0.20, ty - 1.42, r"$\mathcal{L}_{total} = \mathcal{L}_{rec} + 0.01\,\mathcal{L}_{TV}$",
            fontsize=8.3, color=PINK_EDGE, fontweight="bold")

    # ── SAVE ──
    for fmt, dpi in [("pdf", None), ("png", 400)]:
        fpath = OUT / f"fig2_v5_final.{fmt}"
        fig.savefig(fpath, bbox_inches="tight", pad_inches=0.06, dpi=dpi)
        print(f"Saved: {fpath}")
    plt.close(fig)


if __name__ == "__main__":
    draw()