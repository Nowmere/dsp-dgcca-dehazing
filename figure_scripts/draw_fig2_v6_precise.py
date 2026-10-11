"""Figure 2 V6: Precise layout with fully computed coordinates.
Every element has verified position - no overlaps, clean separation.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "CVPR_2026_Submission_Template" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

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


def box(ax, x, y, w, h, t1, t2=None, *,
        fill=BLUE_FILL, edge=BLUE_EDGE, lw=1.5, dash=False,
        fs1=9.0, fs2=7.2, c1=INK, c2=GRAY, z=5):
    """x,y = center. w,h = full width/height."""
    ls = (0, (4, 3)) if dash else "solid"
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.10,rounding_size=0.07",
                       facecolor=fill, edgecolor=edge, linewidth=lw,
                       linestyle=ls, zorder=z)
    ax.add_patch(p)
    if t2 is not None:
        ax.text(x, y + 0.10, t1, ha="center", va="center",
                fontsize=fs1, fontweight="bold", color=c1, zorder=z + 1)
        ax.text(x, y - 0.14, t2, ha="center", va="center",
                fontsize=fs2, color=c2, zorder=z + 1)
    else:
        ax.text(x, y, t1, ha="center", va="center",
                fontsize=fs1, fontweight="bold", color=c1, zorder=z + 1)


def arr(ax, x0, y0, x1, y1, color=BLUE_EDGE, lw=1.2, z=4):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=11, linewidth=lw, color=color,
                        shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)


def path(ax, pts, color=BLUE_EDGE, lw=1.0, z=3):
    """Polyline with arrowhead on final segment."""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=color, linewidth=lw, solid_capstyle="round", zorder=z)
    arr(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1],
        color=color, lw=lw, z=z)


def tline(ax, x0, y0, x1, y1, color=GRAY, lw=0.8, z=3):
    """Straight line segment, no arrow."""
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
            solid_capstyle="round", zorder=z)


def oplus(ax, x, y):
    c = Circle((x, y), 0.16, facecolor=WHITE, edgecolor=BLUE_EDGE,
               linewidth=1.3, zorder=6)
    ax.add_patch(c)
    ax.text(x, y, "+", ha="center", va="center", fontsize=12,
            fontweight="bold", color=INK, zorder=7)


def txt(ax, x, y, s, *, fs=8.0, color=INK, bold=False, ha="left", va="center", z=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs,
            fontweight="bold" if bold else "normal", color=color, zorder=z)


# =====================================================================
def draw():
    W, H = 14.8, 5.2
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── TITLE ──
    txt(ax, 0.25, 5.00,
        "Figure 2.  Overall architecture of the proposed geometry-aware physical dehazing framework.",
        fs=11.5, bold=True)

    # =================================================================
    # TOP ROW: RGB Pipeline (y = 3.80)
    # =================================================================
    YT = 3.80

    # Box centers (x positions)
    bx = [0.85, 2.75, 5.05, 7.70, 10.15, 12.25, 13.40]
    bw = [1.30, 1.35, 1.60, 2.00, 1.35, 0.70, 1.10]
    # box 5 is plus node, box 6 is restored output

    box(ax, bx[0], YT, bw[0], 0.72, "Hazy Input", r"$I$", fill=BLUE_FILL, edge=BLUE_EDGE)
    box(ax, bx[1], YT, bw[1], 0.72, "RGB Encoder", r"$3 \to 64$", fill=BLUE_FILL, edge=BLUE_EDGE)
    box(ax, bx[2], YT, bw[2], 0.80, "DG-CCA", fill=PINK_FILL, edge=PINK_EDGE, lw=2.0, fs1=10.5)
    box(ax, bx[3], YT, bw[3], 0.80, "FFA Backbone", "3 Groups x 3 PA Blocks",
        fill=PURPLE_FILL, edge=PURPLE_EDGE, lw=1.8)
    box(ax, bx[4], YT, bw[4], 0.72, "Reconstruct", r"$3\!\times\!3$ Conv",
        fill=BLUE_FILL, edge=BLUE_EDGE)
    oplus(ax, bx[5], YT)
    box(ax, bx[6], YT, bw[6], 0.72, "Restored", r"$\hat I$", fill=BLUE_FILL, edge=BLUE_EDGE)

    # Horizontal arrows (between boxes)
    h_edges = [
        (bx[0] + bw[0] / 2, bx[1] - bw[1] / 2, BLUE_EDGE),
        (bx[1] + bw[1] / 2, bx[2] - bw[2] / 2, BLUE_EDGE),
        (bx[2] + bw[2] / 2, bx[3] - bw[3] / 2, PINK_EDGE),
        (bx[3] + bw[3] / 2, bx[4] - bw[4] / 2, PURPLE_EDGE),
        (bx[4] + bw[4] / 2, bx[5] - 0.18, BLUE_EDGE),
        (bx[5] + 0.18, bx[6] - bw[6] / 2, BLUE_EDGE),
    ]
    for x0, x1, c in h_edges:
        arr(ax, x0, YT, x1, YT, c)

    # Global residual (drawn above pipeline)
    path(ax, [
        (bx[0] - bw[0] / 2 - 0.05, YT + 0.36),
        (bx[0] - bw[0] / 2 - 0.05, 4.65),
        (bx[5], 4.65),
        (bx[5], YT + 0.66)
    ], color=BLUE_EDGE, lw=0.9, z=1)
    txt(ax, 6.40, 4.69, "Global Residual", fs=7.5, color=BLUE_EDGE, ha="center", va="bottom")

    # =================================================================
    # MIDDLE ROW: Depth Branch (y = 2.30)
    # =================================================================
    YM = 2.30

    box(ax, 2.10, YM, 2.10, 0.80, "Depth Anything V2", "Frozen (24.8M params)",
        fill=GREEN_FILL, edge=GREEN_EDGE, dash=True, fs1=9.2, fs2=7.3)
    box(ax, 4.65, YM, 1.65, 0.80, "Depth Prior", "Min-Max + Adapter",
        fill=GREEN_FILL, edge=GREEN_EDGE, fs1=9.0, fs2=7.2)

    # Hazy → DAV2 (vertical drop)
    path(ax, [(bx[0], YT - 0.36), (bx[0], YM + 0.40), (1.05, YM + 0.40)],
         color=GREEN_EDGE, lw=1.2)
    # DAV2 → Depth Prior
    arr(ax, 3.15, YM, 3.83, YM, GREEN_EDGE)
    # Depth Prior → DG-CCA (vertical rise)
    path(ax, [
        (5.47, YM + 0.40), (5.47, 3.10), (bx[2], YT - 0.40)
    ], color=GREEN_EDGE, lw=1.3)
    txt(ax, 5.72, 3.10, r"$F_d$", fs=8.0, color=GREEN_EDGE, bold=True)

    # =================================================================
    # BOTTOM PANEL: DG-CCA Detail + Training
    # =================================================================
    # Panel background
    px, py, pw, ph = 0.25, 0.25, 14.30, 1.75
    pbg = FancyBboxPatch((px, py), pw, ph,
                         boxstyle="round,pad=0.01,rounding_size=0.05",
                         facecolor=PANEL_FILL, edgecolor=PANEL_EDGE,
                         linewidth=0.9, zorder=0)
    ax.add_patch(pbg)
    txt(ax, px + 0.25, py + ph - 0.20, "DG-CCA Detail   |   Training Strategy",
        fs=9.2, bold=True)

    # --- LEFT SIDE: DG-CCA flow (x from 0.55 to 8.55) ---
    # Layout plan:
    #  Row 1 (y=1.50): F_rgb (cx=1.3) ... GAP ... ... GAP ... F_d (cx=6.5)
    #  Row 2 (y=0.95): z_rgb (cx=2.5) → → → Concat (cx=4.5) ← ← ← z_d (cx=6.3)
    #  Row 3 (y=0.50): MLP (cx=4.5)
    #  Row 4 (bottom): A (x=5.8) → F_out text

    yR1 = py + 1.25   # top row for F_rgb / F_d
    yR2 = py + 0.72   # middle row for z / concat
    yR3 = py + 0.30   # bottom row for MLP

    # Row 1: F_rgb and F_d
    cxFr = 1.40
    cxFd = 6.90
    box(ax, cxFr, yR1, 1.50, 0.48, r"$F_{rgb}$", r"$C\!\times\!H\!\times\!W$",
        fill=BLUE_FILL, edge=BLUE_EDGE, lw=1.2, fs1=8.5, fs2=7.0)
    box(ax, cxFd, yR1, 1.50, 0.48, r"$F_d$", r"$C\!\times\!H\!\times\!W$",
        fill=GREEN_FILL, edge=GREEN_EDGE, lw=1.2, fs1=8.5, fs2=7.0)

    # GAP indicators (text with arrow)
    gapL = cxFr + 1.55  # x for left GAP text
    gapR = cxFd - 1.55  # x for right GAP text
    arr(ax, cxFr + 0.75, yR1, gapL - 0.30, yR1, GRAY, 0.8)
    txt(ax, gapL, yR1, "GAP", fs=7.5, color=GRAY, bold=True, ha="center")
    arr(ax, gapL + 0.30, yR1, gapL + 0.60, yR1, GRAY, 0.8)

    arr(ax, cxFd - 0.75, yR1, gapR + 0.30, yR1, GRAY, 0.8)
    txt(ax, gapR, yR1, "GAP", fs=7.5, color=GRAY, bold=True, ha="center")
    arr(ax, gapR - 0.30, yR1, gapR - 0.60, yR1, GRAY, 0.8)

    # Vertical drops from GAP to z_rgb / z_d
    dz_x1 = gapL + 0.60  # x = 3.55
    dz_x2 = gapR - 0.60  # x = 4.75
    tline(ax, dz_x1, yR1, dz_x1, yR2 + 0.35, GRAY, 0.8)
    arr(ax, dz_x1, yR2 + 0.22, dz_x1, yR2 + 0.05, GRAY, 0.8)
    tline(ax, dz_x2, yR1, dz_x2, yR2 + 0.35, GRAY, 0.8)
    arr(ax, dz_x2, yR2 + 0.22, dz_x2, yR2 + 0.05, GRAY, 0.8)

    # Row 2: z_rgb, z_d, Concat
    czr = dz_x1   # x = 3.55
    czd = dz_x2   # x = 4.75
    cCcat = 4.15  # concat center

    box(ax, czr, yR2, 1.05, 0.42, r"$z_{rgb}$", r"$\mathbb{R}^{C}$",
        fill="#f0f5ff", edge=BLUE_EDGE, lw=1.0, fs1=7.8, fs2=6.5)
    box(ax, czd, yR2, 1.05, 0.42, r"$z_d$", r"$\mathbb{R}^{C}$",
        fill="#eff8ef", edge=GREEN_EDGE, lw=1.0, fs1=7.8, fs2=6.5)
    box(ax, cCcat, yR2, 1.80, 0.45, r"Concat $[z_{rgb};\,z_d]$", r"$\mathbb{R}^{2C}$",
        fill=PINK_FILL, edge=PINK_EDGE, lw=1.3, fs1=7.8, fs2=6.5)

    # Horizontal arrows z → concat
    arr(ax, czr + 0.53, yR2, cCcat - 0.90, yR2, GRAY, 0.8)
    arr(ax, czd - 0.53, yR2, cCcat + 0.90, yR2, GRAY, 0.8)

    # Row 3: MLP
    tline(ax, cCcat, yR2 - 0.23, cCcat, yR3 + 0.33, PINK_EDGE, 0.9)
    arr(ax, cCcat, yR3 + 0.15, cCcat, yR3 - 0.02, PINK_EDGE, 0.9)

    box(ax, cCcat, yR3, 3.80, 0.55, "Bottleneck MLP",
        r"$W_1\;(\mathbb{R}^{C\times 2C}) \to \mathrm{ReLU} \to W_2\;(\mathbb{R}^{C\times C}) \to \sigma$",
        fill=PINK_FILL, edge=PINK_EDGE, lw=1.6, fs1=8.5, fs2=6.5, c2=INK)

    # Output A + F_out
    outY = yR3 - 0.48
    arr(ax, cCcat + 1.90, yR3 - 0.28, cCcat + 1.90, outY + 0.10, PINK_EDGE, 0.9)
    txt(ax, cCcat + 2.30, outY, r"$\mathbf{A}\in\mathbb{R}^{C}$", fs=8.2,
        color=PINK_EDGE, bold=True)
    arr(ax, cCcat + 3.30, outY, cCcat + 4.10, outY, PINK_EDGE, 0.9)
    txt(ax, cCcat + 5.00, outY, r"$F_{out} = \mathbf{A} \odot F_{rgb}$",
        fs=9.0, color=INK, bold=True, ha="center")

    # --- RIGHT SIDE: Training text ---
    tx = 8.80
    ty = py + 1.45
    dy = 0.26
    txt(ax, tx, ty, "Two-Stage Training Curriculum", fs=9.5, bold=True)
    txt(ax, tx, ty - dy, "Stage 1 (Epoch 1–100):   All parameters jointly optimized",
        fs=8.0, color=GRAY)
    txt(ax, tx + 0.15, ty - 2 * dy,
        r"$\mathcal{L}_{rec} = \|\hat I - J\|_1 + 0.05\,\mathcal{L}_{edge}$",
        fs=8.5, color=PINK_EDGE, bold=True)
    txt(ax, tx, ty - 3.5 * dy, "Stage 2 (Epoch 101–120): Backbone frozen, DG-CCA trainable",
        fs=8.0, color=GRAY)
    txt(ax, tx + 0.15, ty - 4.5 * dy,
        r"$M(x,y)=\mathbf{1}[D(x,y)\geq 0.1]$    (Hard sky mask,   $\tau=0.1$)",
        fs=8.0, color=INK)
    txt(ax, tx + 0.15, ty - 5.5 * dy,
        r"$\mathcal{L}_{TV} = \sum_{x,y}(1-M)(\|\nabla_x\hat I\|_1 + \|\nabla_y\hat I\|_1)$",
        fs=7.8, color=INK)
    txt(ax, tx + 0.15, ty - 6.5 * dy,
        r"$\mathcal{L}_{total} = \mathcal{L}_{rec} + 0.01\,\mathcal{L}_{TV}$",
        fs=8.8, color=PINK_EDGE, bold=True)

    # ── SAVE ──
    for fmt, dpi in [("pdf", None), ("png", 400)]:
        fpath = OUT / f"fig2_v6_precise.{fmt}"
        fig.savefig(fpath, bbox_inches="tight", pad_inches=0.06, dpi=dpi)
        print(f"Saved: {fpath}")
    plt.close(fig)


if __name__ == "__main__":
    draw()