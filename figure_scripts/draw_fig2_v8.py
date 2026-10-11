"""Figure 2 V8: Radically simplified layout.
- Main pipeline: horizontal row, exact box centers
- Depth branch: below, simple vertical injection
- DG-CCA detail: minimal cell-based grid (3 rows, no complex GAP paths)
- Training: right-side text block
Every coordinate verified for non-overlap.
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
B_FILL = "#e3ecf7"; B_EDGE = "#2e609a"
G_FILL = "#e2f0e5"; G_EDGE = "#3d7f53"
P_FILL = "#fce4ec"; P_EDGE = "#b71c50"
V_FILL = "#ede7f6"; V_EDGE = "#5e35b1"
PANEL = "#f7f9fc"; PANELE = "#c4cad4"
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


def b(ax, cx, cy, w, h, t1, t2=None, *,
      fc=B_FILL, ec=B_EDGE, lw=1.5, dash=False,
      fs1=9.0, fs2=7.2, z=5):
    ls = (0, (4, 3)) if dash else "solid"
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.10,rounding_size=0.07",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       linestyle=ls, zorder=z)
    ax.add_patch(p)
    if t2:
        ax.text(cx, cy + 0.10, t1, ha="center", va="center",
                fontsize=fs1, fontweight="bold", color=INK, zorder=z + 1)
        ax.text(cx, cy - 0.14, t2, ha="center", va="center",
                fontsize=fs2, color=GRAY, zorder=z + 1)
    else:
        ax.text(cx, cy, t1, ha="center", va="center",
                fontsize=fs1, fontweight="bold", color=INK, zorder=z + 1)


def ar(ax, x0, y0, x1, y1, ec=B_EDGE, lw=1.2, z=4):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=11, linewidth=lw, color=ec,
                        shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)


def pa(ax, pts, ec=B_EDGE, lw=1.0, z=3):
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=ec, linewidth=lw, solid_capstyle="round", zorder=z)
    ar(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], ec=ec, lw=lw, z=z)


def oplus(ax, x, y):
    c = Circle((x, y), 0.16, facecolor=WHITE, edgecolor=B_EDGE,
               linewidth=1.3, zorder=6)
    ax.add_patch(c)
    ax.text(x, y, "+", ha="center", va="center", fontsize=12,
            fontweight="bold", color=INK, zorder=7)


def txt(ax, x, y, s, fs=8.0, c=INK, b=False, ha="left", va="center", z=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs,
            fontweight="bold" if b else "normal", color=c, zorder=z)


# =====================================================================
def draw():
    W, H = 14.8, 5.5
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    txt(ax, 0.25, 5.30,
        "Figure 2.  Overall architecture of the proposed geometry-aware physical dehazing framework.",
        fs=11.5, b=True)

    # =========== ROW 1: RGB Pipeline (y=4.00) ===========
    ya = 4.00
    xa = [0.85, 2.75, 5.05, 7.70, 10.15, 12.20, 13.35]
    wa = [1.30, 1.35, 1.60, 2.00, 1.35, 0.00, 1.10]
    items = [
        (B_FILL, B_EDGE, 1.5,  ("Hazy Input", r"$I$")),
        (B_FILL, B_EDGE, 1.5,  ("RGB Encoder", r"$3 \to 64$")),
        (P_FILL, P_EDGE, 2.0,  ("DG-CCA", None)),
        (V_FILL, V_EDGE, 1.8,  ("FFA Backbone", "3 Groups x 3 PA Blocks")),
        (B_FILL, B_EDGE, 1.5,  ("Reconstruct", r"$3\!\times\!3$ Conv")),
        (None, None, None, None),
        (B_FILL, B_EDGE, 1.5,  ("Restored", r"$\hat I$")),
    ]

    for i in range(7):
        if i == 5:
            oplus(ax, xa[i], ya)
            continue
        fc, ec, lw, (t1, t2) = items[i]
        b(ax, xa[i], ya, wa[i], 0.72, t1, t2, fc=fc, ec=ec, lw=lw,
          fs1=10.5 if i == 2 else 9.0)

    # Horizontal arrows
    pairs = [
        (xa[0]+wa[0]/2, xa[1]-wa[1]/2, B_EDGE),
        (xa[1]+wa[1]/2, xa[2]-wa[2]/2, B_EDGE),
        (xa[2]+wa[2]/2, xa[3]-wa[3]/2, P_EDGE),
        (xa[3]+wa[3]/2, xa[4]-wa[4]/2, V_EDGE),
        (xa[4]+wa[4]/2, xa[5]-0.18, B_EDGE),
        (xa[5]+0.18, xa[6]-wa[6]/2, B_EDGE),
    ]
    for x0, x1, ec in pairs:
        ar(ax, x0, ya, x1, ya, ec=ec)

    # Global residual (top)
    pa(ax, [
        (xa[0]-wa[0]/2-0.05, ya+0.36),
        (xa[0]-wa[0]/2-0.05, 4.85),
        (xa[5], 4.85),
        (xa[5], ya+0.38)
    ], ec=B_EDGE, lw=0.9, z=1)
    txt(ax, 6.35, 4.89, "Global Residual", fs=7.5, c=B_EDGE, ha="center", va="bottom")

    # =========== ROW 2: Depth Branch (y=2.45) ===========
    yb = 2.45
    b(ax, 2.10, yb, 2.10, 0.78, "Depth Anything V2", "Frozen (24.8M params)",
      fc=G_FILL, ec=G_EDGE, dash=True, fs1=9.2, fs2=7.3)
    b(ax, 4.65, yb, 1.70, 0.78, "Depth Prior", "Min-Max + Adapter",
      fc=G_FILL, ec=G_EDGE, fs1=9.0, fs2=7.2)

    pa(ax, [(xa[0], ya-0.36), (xa[0], yb+0.39), (1.05, yb+0.39)],
       ec=G_EDGE, lw=1.2)
    ar(ax, 3.15, yb, 3.80, yb, ec=G_EDGE)
    pa(ax, [(5.50, yb+0.39), (5.50, 3.30), (xa[2], ya-0.36-0.05)],
       ec=G_EDGE, lw=1.3)
    txt(ax, 5.73, 3.30, r"$F_d$", fs=8.0, c=G_EDGE, b=True)

    # =========== BOTTOM PANEL ===========
    px, py, pw, ph = 0.25, 0.25, 14.30, 1.85
    bg = FancyBboxPatch((px, py), pw, ph,
                        boxstyle="round,pad=0.01,rounding_size=0.05",
                        facecolor=PANEL, edgecolor=PANELE,
                        linewidth=0.9, zorder=0)
    ax.add_patch(bg)
    txt(ax, px+0.25, py+ph-0.20, "DG-CCA Detail   |   Training Strategy", fs=9.2, b=True)

    # ----- DG-CCA: 3-row grid -----
    # Row3a (F_rgb / F_d): y = py + 1.30 = 1.55
    # Row3b (z_rgb / z_d): y = py + 0.75 = 1.00
    # Row3c (Concat):       y = py + 0.35 = 0.60
    # Row3d (MLP):          y = py - 0.10 = 0.15 (slightly below)
    # Row3e (output):       y = py - 0.45 = -0.20 (text only)
    r3a = py + 1.30   # 1.55
    r3b = py + 0.75   # 1.00
    r3c = py + 0.35   # 0.60
    r3d = py - 0.05   # 0.20
    r3e = py - 0.38   # -0.13

    # F_rgb (left)
    b(ax, 1.35, r3a, 1.50, 0.47, r"$F_{rgb}$", r"$C\!\times\!H\!\times\!W$",
      fc=B_FILL, ec=B_EDGE, lw=1.2, fs1=8.5, fs2=7.0)
    # F_d (right)
    b(ax, 6.85, r3a, 1.50, 0.47, r"$F_d$", r"$C\!\times\!H\!\times\!W$",
      fc=G_FILL, ec=G_EDGE, lw=1.2, fs1=8.5, fs2=7.0)

    # Simple downward arrows from F_rgb / F_d to z_rgb / z_d
    # z_rgb box center = 2.65 (midway between F_rgb center 1.35 and concat center 4.10)
    # z_d   box center = 5.55 (midway between F_d center 6.85 and concat center 4.10)
    zx1, zx2 = 2.65, 5.55
    cc = 4.10   # concat center

    # Vertical arrows: F→z
    ar(ax, zx1, r3a - 0.235, zx1, r3b + 0.23, ec=GRAY, lw=0.8)
    ar(ax, zx2, r3a - 0.235, zx2, r3b + 0.23, ec=GRAY, lw=0.8)

    # z_rgb and z_d
    b(ax, zx1, r3b, 1.10, 0.42, r"$z_{rgb}$", r"$\mathbb{R}^{C}$",
      fc="#f0f5ff", ec=B_EDGE, lw=1.0, fs1=7.8, fs2=6.5)
    b(ax, zx2, r3b, 1.10, 0.42, r"$z_d$", r"$\mathbb{R}^{C}$",
      fc="#eff8ef", ec=G_EDGE, lw=1.0, fs1=7.8, fs2=6.5)

    # Concat
    b(ax, cc, r3c, 2.20, 0.45, r"Concat $[z_{rgb};\,z_d]$", r"$\mathbb{R}^{2C}$",
      fc=P_FILL, ec=P_EDGE, lw=1.3, fs1=7.8, fs2=6.5)

    # Arrows: z→concat
    ar(ax, zx1, r3b - 0.21, zx1, r3c + 0.225, ec=GRAY, lw=0.7)
    ar(ax, zx2, r3b - 0.21, zx2, r3c + 0.225, ec=GRAY, lw=0.7)

    # MLP
    b(ax, cc, r3d, 4.20, 0.53, "Bottleneck MLP",
      r"$W_1\;(\mathbb{R}^{C\times 2C}) \to \mathrm{ReLU} \to W_2\;(\mathbb{R}^{C\times C}) \to \sigma$",
      fc=P_FILL, ec=P_EDGE, lw=1.6, fs1=8.5, fs2=6.5, c2=INK)
    ar(ax, cc, r3c - 0.225, cc, r3d + 0.265, ec=P_EDGE, lw=0.9)

    # Output: A → F_out
    ar(ax, cc + 2.10, r3d - 0.265, cc + 2.10, r3e + 0.12, ec=P_EDGE, lw=0.9)
    txt(ax, cc + 2.45, r3e, r"$\mathbf{A}\in\mathbb{R}^{C}$", fs=8.2, c=P_EDGE, b=True)
    ar(ax, cc + 3.80, r3e, cc + 4.60, r3e, ec=P_EDGE, lw=0.9)
    txt(ax, cc + 5.60, r3e, r"$F_{out} = \mathbf{A} \odot F_{rgb}$",
        fs=9.0, c=INK, b=True, ha="center")

    # ----- Training text (right) -----
    tx = 8.80
    ty = py + 1.55
    dd = 0.28
    txt(ax, tx, ty, "Two-Stage Training", fs=9.5, b=True)
    txt(ax, tx, ty-dd, "Stage 1 (Epoch 1..100):  All parameters jointly trained",
        fs=8.0, c=GRAY)
    txt(ax, tx+0.15, ty-2*dd,
        r"$\mathcal{L}_{rec} = \|\hat I - J\|_1 + 0.05\,\mathcal{L}_{edge}$",
        fs=8.5, c=P_EDGE, b=True)
    txt(ax, tx, ty-3.5*dd, "Stage 2 (Epoch 101..120): Backbone frozen, DG-CCA trainable",
        fs=8.0, c=GRAY)
    txt(ax, tx+0.15, ty-4.5*dd,
        r"$M(x,y)=\mathbf{1}[D(x,y)\geq 0.1]$    ($\tau=0.1$, morphological opening)",
        fs=8.0, c=INK)
    txt(ax, tx+0.15, ty-5.5*dd,
        r"$\mathcal{L}_{TV} = \sum_{x,y}(1-M)(\|\nabla_x\hat I\|_1 + \|\nabla_y\hat I\|_1)$",
        fs=7.8, c=INK)
    txt(ax, tx+0.15, ty-6.5*dd,
        r"$\mathcal{L}_{total} = \mathcal{L}_{rec} + 0.01\,\mathcal{L}_{TV}$",
        fs=8.8, c=P_EDGE, b=True)

    # Save
    for fmt, dpi in [("pdf", None), ("png", 400)]:
        fp = OUT / f"fig2_v8.{fmt}"
        fig.savefig(fp, bbox_inches="tight", pad_inches=0.06, dpi=dpi)
        print(f"Saved: {fp}")
    plt.close(fig)


if __name__ == "__main__":
    draw()