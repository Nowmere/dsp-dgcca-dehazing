"""Figure 2: Clean, minimal architecture overview.
One main pipeline (horizontal) + depth branch (below) + DG-CCA detail (right).
No equation clutter, precise alignment, CVPR-quality.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "CVPR_2026_Submission_Template" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── colors ──
INK = "#1a1a2e"
GRAY = "#5a606e"
BLUE_FILL = "#e3ecf7"
BLUE_EDGE = "#2e609a"
GREEN_FILL = "#e2f0e5"
GREEN_EDGE = "#3d7f53"
PINK_FILL = "#fce4ec"
PINK_EDGE = "#b71c50"
PURPLE_FILL = "#ede7f6"
PURPLE_EDGE = "#5e35b1"
ORANGE_FILL = "#fff3e0"
ORANGE_EDGE = "#d84315"
PANEL_FILL = "#f7f9fc"
PANEL_EDGE = "#c4cad4"
WHITE = "#ffffff"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
    "savefig.facecolor": "white",
})


def box(ax, x, y, w, h, text, *, fill=BLUE_FILL, edge=BLUE_EDGE, lw=1.6, dash=False, fs=9.5, fc=INK):
    ls = (0, (4, 3)) if dash else "solid"
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.1,rounding_size=0.08",
                       facecolor=fill, edgecolor=edge, linewidth=lw,
                       linestyle=ls, zorder=4)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=fc, zorder=5)


def box2(ax, x, y, w, h, t1, t2, *, fill=BLUE_FILL, edge=BLUE_EDGE, lw=1.6, dash=False, fs1=9.5, fs2=7.5):
    ls = (0, (4, 3)) if dash else "solid"
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.1,rounding_size=0.08",
                       facecolor=fill, edgecolor=edge, linewidth=lw,
                       linestyle=ls, zorder=4)
    ax.add_patch(p)
    ax.text(x, y + 0.12, t1, ha="center", va="center", fontsize=fs1,
            fontweight="bold", color=INK, zorder=5)
    ax.text(x, y - 0.14, t2, ha="center", va="center", fontsize=fs2,
            color=GRAY, zorder=5)


def arrow(ax, x0, y0, x1, y1, color=BLUE_EDGE, lw=1.3, dash=False, z=3):
    ls = (0, (4, 3)) if dash else "solid"
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=12, linewidth=lw, linestyle=ls,
                        color=color, shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)


def line_path(ax, pts, color=BLUE_EDGE, lw=1.1, z=2):
    for i in range(len(pts) - 1):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=color, linewidth=lw, solid_capstyle="round", zorder=z)
    # arrowhead on last segment
    arrow(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1],
          color=color, lw=lw, z=z)


def panel_box(ax, x, y, w, h, title):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.01,rounding_size=0.06",
                       facecolor=PANEL_FILL, edgecolor=PANEL_EDGE,
                       linewidth=1.0, zorder=0)
    ax.add_patch(p)
    ax.text(x + 0.22, y + h - 0.24, title, ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=INK, zorder=6)


def plus_circle(ax, x, y):
    c = Circle((x, y), 0.17, facecolor=WHITE, edgecolor=BLUE_EDGE,
               linewidth=1.3, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, "+", ha="center", va="center", fontsize=13,
            fontweight="bold", color=INK, zorder=6)


# =====================================================================
def draw():
    W, H = 14.2, 6.0
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Title ──
    ax.text(0.30, 5.80, "Figure 2.  Overall architecture of the proposed geometry-aware physical dehazing framework.",
            fontsize=11.5, fontweight="bold", color=INK, zorder=6)

    # =================================================================
    # UPPER TRACK: RGB pipeline (horizontal) — y = 4.2
    # =================================================================
    Y = 4.2

    # 1) Hazy Input
    box2(ax, 0.80, Y, 1.30, 0.75, "Hazy Input", r"$I$", fill=BLUE_FILL, edge=BLUE_EDGE)

    # 2) RGB Encoder
    box2(ax, 2.90, Y, 1.40, 0.75, "RGB Encoder", r"$3 \rightarrow 64$", fill=BLUE_FILL, edge=BLUE_EDGE)
    arrow(ax, 1.45, Y, 2.20, Y, BLUE_EDGE)

    # 3) DG-CCA
    box(ax, 5.30, Y, 1.60, 0.85, "DG-CCA", fill=PINK_FILL, edge=PINK_EDGE, lw=2.0, fs=10.5)
    arrow(ax, 3.60, Y, 4.50, Y, BLUE_EDGE)

    # 4) FFA Backbone
    box2(ax, 7.80, Y, 2.00, 0.85, "FFA Backbone", "3 Groups, 3 PA Blocks each", fill=PURPLE_FILL, edge=PURPLE_EDGE, lw=1.8, fs1=9.5, fs2=7.3)
    arrow(ax, 6.10, Y, 6.80, Y, PINK_EDGE)

    # 5) Reconstruction
    box2(ax, 10.50, Y, 1.40, 0.75, "Reconstruct", r"$3 \times 3$ Conv", fill=BLUE_FILL, edge=BLUE_EDGE)
    arrow(ax, 8.80, Y, 9.80, Y, PURPLE_EDGE)

    # 6) ⨁ Add
    plus_circle(ax, 11.95, Y)

    # 7) Output
    box2(ax, 13.10, Y, 1.10, 0.75, "Restored", r"$\hat I$", fill=BLUE_FILL, edge=BLUE_EDGE)
    arrow(ax, 12.12, Y, 12.55, Y, BLUE_EDGE)

    # Global residual (drawn ABOVE the RGB track)
    line_path(ax, [
        (0.80 - 0.65, Y + 0.38), (0.80 - 0.65, 5.45),
        (11.95, 5.45), (11.95, Y + 0.70)
    ], color=BLUE_EDGE, lw=1.0, z=1)
    ax.text(6.40, 5.50, "Global Residual Connection", fontsize=7.5,
            color=BLUE_EDGE, ha="center", va="bottom", zorder=5)

    # =================================================================
    # LOWER TRACK: Depth branch — y = 2.3
    # =================================================================
    YB = 2.3

    # Frozen DAV2
    box2(ax, 2.20, YB, 2.10, 0.80, "Depth Anything V2", "Frozen (24.8M params)",
         fill=GREEN_FILL, edge=GREEN_EDGE, dash=True, fs1=9.2, fs2=7.3)

    # Branch from hazy input
    line_path(ax, [
        (0.80, Y - 0.38), (0.80, YB + 0.40), (1.15, YB + 0.40)
    ], color=GREEN_EDGE, lw=1.2, z=2)

    # Depth Prior
    box2(ax, 4.50, YB, 1.60, 0.80, "Depth Prior", "Min-Max + Adapter",
         fill=GREEN_FILL, edge=GREEN_EDGE, fs1=9.0, fs2=7.3)
    arrow(ax, 3.25, YB, 3.70, YB, GREEN_EDGE)

    # Depth → DG-CCA injection (arrow going up)
    line_path(ax, [
        (5.30, YB + 0.40), (5.30, 3.40), (5.30, Y - 0.43)
    ], color=GREEN_EDGE, lw=1.3, z=2)

    ax.text(5.55, 3.40, r"$F_d$", fontsize=8.0, color=GREEN_EDGE,
            fontweight="bold", ha="left", zorder=5)

    # =================================================================
    # RIGHT PANEL: DG-CCA Detail
    # =================================================================
    rx, ry, rw, rh = 5.80, 0.35, 8.10, 1.65
    panel_box(ax, rx, ry, rw, rh, "DG-CCA Detail (right panel)")

    # F_rgb box
    box(ax, rx + 1.20, ry + 1.08, 1.60, 0.45, r"$F_{rgb}$", fill=BLUE_FILL, edge=BLUE_EDGE, lw=1.2, fs=8.8)
    # F_d box
    box(ax, rx + 6.30, ry + 1.08, 1.60, 0.45, r"$F_d$", fill=GREEN_FILL, edge=GREEN_EDGE, lw=1.2, fs=8.8)

    # "GAP" labels & arrows
    ax.text(rx + 2.60, ry + 1.08, "GAP", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=GRAY, zorder=5)
    ax.text(rx + 5.50, ry + 1.08, "GAP", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=GRAY, zorder=5)

    arrow(ax, rx + 2.00, ry + 1.08, rx + 2.40, ry + 1.08, GRAY, 0.8)
    arrow(ax, rx + 5.50, ry + 1.08, rx + 5.90, ry + 1.08, GRAY, 0.8)

    # Arrow down from GAP to z descriptors
    arrow(ax, rx + 2.60, ry + 0.86, rx + 2.60, ry + 0.68, GRAY, 0.8)
    arrow(ax, rx + 5.50, ry + 0.86, rx + 5.50, ry + 0.68, GRAY, 0.8)

    box(ax, rx + 3.85, ry + 0.48, 0.40, 0.40, "", fill=PINK_FILL, edge=PINK_EDGE, lw=0, fs=0)
    ax.text(rx + 3.85, ry + 0.48, "Concat", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=INK, zorder=5)

    # z_rgb and z_d at top of "Concat" box
    ax.text(rx + 3.05, ry + 0.68, r"$z_{rgb}$", ha="center", va="center",
            fontsize=7.8, color=INK, zorder=5)
    ax.text(rx + 4.65, ry + 0.68, r"$z_d$", ha="center", va="center",
            fontsize=7.8, color=INK, zorder=5)

    # Arrows into concat
    arrow(ax, rx + 3.30, ry + 0.68, rx + 3.65, ry + 0.68, GRAY, 0.7)
    arrow(ax, rx + 4.40, ry + 0.68, rx + 4.05, ry + 0.68, GRAY, 0.7)

    # MLP
    box(ax, rx + 3.85, ry + 0.15, 2.80, 0.55, "MLP: FC  ReLU  FC  σ", fill=PINK_FILL, edge=PINK_EDGE, lw=1.5, fs=8.0)
    arrow(ax, rx + 3.85, ry + 0.28, rx + 3.85, ry - 0.13, PINK_EDGE, 0.9)

    # Channel gate output
    ax.text(rx + 6.05, ry + 0.15, r"$\mathbf{A} \in \mathbb{R}^C$", fontsize=8.5,
            color=PINK_EDGE, fontweight="bold", ha="center", zorder=5)
    arrow(ax, rx + 5.25, ry + 0.15, rx + 5.75, ry + 0.15, PINK_EDGE, 0.9)

    ax.text(rx + 7.00, ry + 0.60, r"$F_{out} = \mathbf{A} \odot F_{rgb}$", fontsize=8.8,
            color=INK, fontweight="bold", ha="center", zorder=5)

    # Tiny separator line
    ax.plot([rx + 0.80, rx + 1.60], [ry + 0.15, ry + 0.15], color=PANEL_EDGE, linewidth=0.5, zorder=2)

    # =================================================================
    # LEFT BOTTOM: Training note
    # =================================================================
    lx, ly, lw, lh = 0.30, 0.35, 5.20, 1.65
    panel_box(ax, lx, ly, lw, lh, "Training Strategy")

    ax.text(lx + 0.30, ly + 1.30, "Stage 1 (Epoch 1–100)", fontsize=9.0, fontweight="bold", color=INK, zorder=5)
    ax.text(lx + 0.50, ly + 1.08, "All parameters trainable", fontsize=8.0, color=GRAY, zorder=5)

    ax.plot([lx + 0.30, lx + lw - 0.30], [ly + 0.90, ly + 0.90], color=PANEL_EDGE, linewidth=0.5, linestyle="--", zorder=2)

    ax.text(lx + 0.30, ly + 0.72, "Stage 2 (Epoch 101–120)", fontsize=9.0, fontweight="bold", color=INK, zorder=5)
    ax.text(lx + 0.50, ly + 0.50, "Backbone frozen; only DG-CCA trainable", fontsize=8.0, color=GRAY, zorder=5)

    ax.plot([lx + 0.30, lx + lw - 0.30], [ly + 0.35, ly + 0.35], color=PANEL_EDGE, linewidth=0.5, linestyle="--", zorder=2)

    ax.text(lx + 0.30, ly + 0.18, r"$\mathcal{L}_{total} = \mathcal{L}_{rec} + 0.01\,\mathcal{L}_{TV}$",
            fontsize=8.8, color=ORANGE_EDGE, fontweight="bold", zorder=5)
    ax.text(lx + 0.50, ly - 0.02, r"Sky mask: $M = \mathbf{1}[D \geq 0.1]$",
            fontsize=8.0, color=GRAY, zorder=5)

    # ── SAVE ──
    for fmt, dpi in [("pdf", None), ("png", 400)]:
        path = OUT / f"fig2_v4_clean.{fmt}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.06, dpi=dpi)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    draw()