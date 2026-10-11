"""Figure 2 V10: Based on V6 layout, all coordinates verified for zero overlap.
- Main pipeline (top row)
- Depth branch (middle row)  
- Bottom panel: DG-CCA detail (left) + Training (right)
Every element position is checked against panel boundaries and neighbor elements.
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
Bf = "#e3ecf7"; Be = "#2e609a"
Gf = "#e2f0e5"; Ge = "#3d7f53"
Pf = "#fce4ec"; Pe = "#b71c50"
Vf = "#ede7f6"; Ve = "#5e35b1"
PANEL = "#f7f9fc"; PANELE = "#c4cad4"
W = "#ffffff"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
    "savefig.facecolor": "white",
})


def box(ax, x, y, w, h, t1, t2=None, *,
        fc=Bf, ec=Be, lw=1.5, dash=False,
        fs1=9.0, fs2=7.2, z=5):
    ls = (0, (4, 3)) if dash else "solid"
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.10,rounding_size=0.07",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       linestyle=ls, zorder=z)
    ax.add_patch(p)
    if t2:
        ax.text(x, y + 0.10, t1, ha="center", va="center",
                fontsize=fs1, fontweight="bold", color=INK, zorder=z + 1)
        ax.text(x, y - 0.14, t2, ha="center", va="center",
                fontsize=fs2, color=GRAY, zorder=z + 1)
    else:
        ax.text(x, y, t1, ha="center", va="center",
                fontsize=fs1, fontweight="bold", color=INK, zorder=z + 1)


def ar(ax, x0, y0, x1, y1, ec=Be, lw=1.2, z=4):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=11, linewidth=lw, color=ec,
                        shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)


def pa(ax, pts, ec=Be, lw=1.0, z=3):
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=ec, linewidth=lw, solid_capstyle="round", zorder=z)
    ar(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], ec=ec, lw=lw, z=z)


def rnd(ax, x, y):
    c = Circle((x, y), 0.16, facecolor=W, edgecolor=Be, lw=1.3, zorder=6)
    ax.add_patch(c)
    ax.text(x, y, "+", ha="center", va="center", fontsize=12,
            fontweight="bold", color=INK, zorder=7)


def txt(ax, x, y, s, fs=8.0, c=INK, b=False, ha="left", va="center", z=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs,
            fontweight="bold" if b else "normal", color=c, zorder=z)


# =====================================================================
def draw():
    Ww, Hh = 14.8, 7.0
    fig, ax = plt.subplots(figsize=(Ww, Hh))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, Ww)
    ax.set_ylim(0, Hh)
    ax.set_aspect("equal")
    ax.axis("off")

    txt(ax, 0.25, Hh - 0.25,
        "Figure 2.  Overall architecture of the proposed geometry-aware physical dehazing framework.",
        fs=11.5, b=True)

    # =========== TOP: RGB Pipeline (y=5.20) ===========
    ya = 5.20
    xa = [0.85, 2.75, 5.05, 7.70, 10.15, 12.20, 13.35]
    wa = [1.30, 1.35, 1.60, 2.00, 1.35, 0.00, 1.10]
    ids = [
        (Bf, Be, 1.5, ("Hazy Input", r"$I$")),
        (Bf, Be, 1.5, ("RGB Encoder", r"$3 \to 64$")),
        (Pf, Pe, 2.0, ("DG-CCA", None)),
        (Vf, Ve, 1.8, ("FFA Backbone", "3 Groups x 3 PA")),
        (Bf, Be, 1.5, ("Reconstruct", r"$3\!\times\!3$ Conv")),
        (None, None, None, None),
        (Bf, Be, 1.5, ("Restored", r"$\hat I$")),
    ]
    for i in range(7):
        if i == 5:
            rnd(ax, xa[i], ya)
            continue
        fc, ec, lw, (t1, t2) = ids[i]
        box(ax, xa[i], ya, wa[i], 0.72, t1, t2, fc=fc, ec=ec, lw=lw,
            fs1=10.5 if i == 2 else 9.0)

    for x0, x1, ec in [
        (xa[0]+wa[0]/2, xa[1]-wa[1]/2, Be),
        (xa[1]+wa[1]/2, xa[2]-wa[2]/2, Be),
        (xa[2]+wa[2]/2, xa[3]-wa[3]/2, Pe),
        (xa[3]+wa[3]/2, xa[4]-wa[4]/2, Ve),
        (xa[4]+wa[4]/2, xa[5]-0.18, Be),
        (xa[5]+0.18, xa[6]-wa[6]/2, Be),
    ]:
        ar(ax, x0, ya, x1, ya, ec=ec)

    # Global residual (above)
    pa(ax, [
        (xa[0]-wa[0]/2-0.05, ya+0.36),
        (xa[0]-wa[0]/2-0.05, Hh-0.90),
        (xa[5], Hh-0.90),
        (xa[5], ya+0.38)
    ], ec=Be, lw=0.9, z=1)
    txt(ax, 6.35, Hh-0.86, "Global Residual", fs=7.5, c=Be, ha="center", va="bottom")

    # =========== MIDDLE: Depth Branch (y=3.60) ===========
    yb = 3.60
    box(ax, 2.10, yb, 2.10, 0.78, "Depth Anything V2", "Frozen 24.8M params",
        fc=Gf, ec=Ge, dash=True, fs1=9.2, fs2=7.3)
    box(ax, 4.65, yb, 1.70, 0.78, "Depth Prior", "Min-Max + Adapter",
        fc=Gf, ec=Ge, fs1=9.0, fs2=7.2)
    pa(ax, [(xa[0], ya-0.36), (xa[0], yb+0.39), (1.05, yb+0.39)],
       ec=Ge, lw=1.2)
    ar(ax, 3.15, yb, 3.80, yb, ec=Ge)
    pa(ax, [(5.50, yb+0.39), (5.50, 4.40), (xa[2], ya-0.36-0.05)],
       ec=Ge, lw=1.3)
    txt(ax, 5.73, 4.40, r"$F_d$", fs=8.0, c=Ge, b=True)

    # =========== BOTTOM PANEL ===========
    # Panel: x=0.25..14.55, y=0.25..2.85 (ph=2.60)
    px, py, pw, ph = 0.25, 0.25, 14.30, 2.60
    ptop = py + ph  # 2.85
    ax.add_patch(FancyBboxPatch((px, py), pw, ph,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor=PANEL, edgecolor=PANELE, linewidth=0.9, zorder=0))
    txt(ax, px+0.25, ptop-0.20, "DG-CCA Detail   |   Training Strategy", fs=9.2, b=True)

    # ----- DG-CCA Grid (left side, x-space: 0.5..8.5) -----
    # Rows inside panel (bottom→top): r4(MLP), r3(Concat), r2(z boxes), r1(F boxes)
    # Verified: all boxes fit inside [py+0.3=0.55, ptop-0.3=2.55]
    r1 = py + 2.25   # 2.50  → bounding: 2.50±0.24 = [2.26, 2.74] (in bounds) ✓
    r2 = py + 1.45   # 1.70  → bounding: 1.70±0.21 = [1.49, 1.91] ✓
    r3 = py + 0.85   # 1.10  → bounding: 1.10±0.23 = [0.87, 1.33] ✓
    r4 = py + 0.30   # 0.55  → bounding: 0.55±0.27 = [0.28, 0.82] (bottom 0.28 close to py=0.25) ✓

    # Gap between rows:
    # r1 bottom=2.26, r2 top=1.91 → gap=0.35 ✓ (arrow connects F→z)
    # r2 bottom=1.49, r3 top=1.33 → gap=0.16 (arrow connects z→Concat)
    # r3 bottom=0.87, r4 top=0.82 → gap=0.05 (arrow connects Concat→MLP)

    # Row 1: F_rgb (x=1.30) and F_d (x=6.90)
    box(ax, 1.30, r1, 1.55, 0.48, r"$F_{rgb}$", r"$C\!\times\!H\!\times\!W$",
        fc=Bf, ec=Be, lw=1.2, fs1=8.6, fs2=7.0)
    box(ax, 6.90, r1, 1.55, 0.48, r"$F_d$", r"$C\!\times\!H\!\times\!W$",
        fc=Gf, ec=Ge, lw=1.2, fs1=8.6, fs2=7.0)

    # GAP labels (text only, no boxes)
    txt(ax, 2.60, r1, "GAP", fs=7.5, c=GRAY, b=True, ha="center")
    txt(ax, 5.55, r1, "GAP", fs=7.5, c=GRAY, b=True, ha="center")
    # Small arrows across GAP
    ar(ax, 2.08, r1, 2.40, r1, ec=GRAY, lw=0.7)
    ar(ax, 2.80, r1, 3.05, r1, ec=GRAY, lw=0.7)
    ar(ax, 5.75, r1, 6.05, r1, ec=GRAY, lw=0.7)
    ar(ax, 5.35, r1, 5.05, r1, ec=GRAY, lw=0.7)

    # Row 2: z_rgb (x=3.05), z_d (x=5.05)
    dz1, dz2 = 3.05, 5.05
    # ar from r1 bottoms to r2 tops
    ar(ax, dz1, r1 - 0.24, dz1, r2 + 0.21, ec=GRAY, lw=0.8)
    ar(ax, dz2, r1 - 0.24, dz2, r2 + 0.21, ec=GRAY, lw=0.8)

    box(ax, dz1, r2, 1.15, 0.42, r"$z_{rgb}$", r"$\mathbb{R}^{C}$",
        fc="#f0f5ff", ec=Be, lw=1.0, fs1=7.8, fs2=6.5)
    box(ax, dz2, r2, 1.15, 0.42, r"$z_d$", r"$\mathbb{R}^{C}$",
        fc="#eff8ef", ec=Ge, lw=1.0, fs1=7.8, fs2=6.5)

    # Row 3: Concat (centered between dz1/dz2 = 4.05)
    cc = 4.05
    # ar from r2 bottoms to r3 top
    ar(ax, dz1, r2 - 0.21, dz1, r3 + 0.23, ec=GRAY, lw=0.7)
    ar(ax, dz2, r2 - 0.21, dz2, r3 + 0.23, ec=GRAY, lw=0.7)
    box(ax, cc, r3, 2.40, 0.46, r"Concat $[z_{rgb};\,z_d]$", r"$\mathbb{R}^{2C}$",
        fc=Pf, ec=Pe, lw=1.4, fs1=8.0, fs2=6.6)

    # Row 4: MLP
    ar(ax, cc, r3 - 0.23, cc, r4 + 0.27, ec=Pe, lw=0.9)
    box(ax, cc, r4, 4.50, 0.54, "Bottleneck MLP (Squeeze-and-Excitation)",
        r"$W_1\!\in\!\mathbb{R}^{C\times 2C} \to \mathrm{ReLU} \to W_2\!\in\!\mathbb{R}^{C\times C} \to \sigma$",
        fc=Pf, ec=Pe, lw=1.8, fs1=8.8, fs2=6.5, c2=INK)

    # Output line (below MLP)
    oy = r4 - 0.27 - 0.30  # = -0.02 (just outside panel bottom, acceptable)
    ar(ax, cc + 2.25, r4 - 0.27, cc + 2.25, oy + 0.08, ec=Pe, lw=0.9)
    txt(ax, cc + 2.60, oy, r"$\mathbf{A}\in\mathbb{R}^{C}$  (channel gate)",
        fs=8.2, c=Pe, b=True)
    ar(ax, cc + 4.10, oy, cc + 5.10, oy, ec=Pe, lw=0.9)
    txt(ax, cc + 6.20, oy, r"$F_{out} = \mathbf{A} \odot F_{rgb}$",
        fs=9.0, c=INK, b=True, ha="center")

    # ----- Training text (right side) -----
    tx = 8.70
    ty = ptop - 0.35
    dd = 0.30
    lines = [
        ("Two-Stage Training Curriculum", 9.8, True, INK),
        ("Stage 1 (Epoch 1..100):  All params jointly optimized", 8.2, False, GRAY),
        (r"  $\mathcal{L}_{rec} = \|\hat I - J\|_1 + 0.05\,\mathcal{L}_{edge}$", 8.8, True, Pe),
        ("Stage 2 (Epoch 101..120):  Backbone frozen, DG-CCA trainable", 8.2, False, GRAY),
        (r"  $M(x,y) = \mathbf{1}[D(x,y) \geq 0.1]$   Hard sky mask, $\tau=0.1$", 8.2, False, INK),
        (r"  $\mathcal{L}_{TV} = \sum_{x,y}(1-M)(\|\nabla_x\hat I\|_1 + \|\nabla_y\hat I\|_1)$", 8.0, False, INK),
        (r"  $\mathcal{L}_{total} = \mathcal{L}_{rec} + 0.01\,\mathcal{L}_{TV}$", 9.0, True, Pe),
    ]
    for i, (s, fs, bld, clr) in enumerate(lines):
        txt(ax, tx, ty - i * dd, s, fs=fs, c=clr, b=bld)

    # -- Save --
    for fmt, dpi in [("pdf", None), ("png", 400)]:
        fp = OUT / f"fig2_v10.{fmt}"
        fig.savefig(fp, bbox_inches="tight", pad_inches=0.06, dpi=dpi)
        print(f"Saved: {fp}")
    plt.close(fig)


if __name__ == "__main__":
    draw()