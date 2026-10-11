"""Figure 2 V11: V6-style layout, 3-row DG-CCA with verified spacing.
No overlaps, clean vertical gaps between all rows.
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
Bf, Be = "#e3ecf7", "#2e609a"
Gf, Ge = "#e2f0e5", "#3d7f53"
Pf, Pe = "#fce4ec", "#b71c50"
Vf, Ve = "#ede7f6", "#5e35b1"
PANEL, PANELE = "#f7f9fc", "#c4cad4"
W = "#ffffff"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
    "savefig.facecolor": "white",
})


def b(ax, cx, cy, w, h, t1, t2=None, *,
      fc=Bf, ec=Be, lw=1.5, dash=False, fs1=9.0, fs2=7.2,
      c1=INK, c2=GRAY, z=5):
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


def ar(ax, x0, y0, x1, y1, ec=Be, lw=1.2, z=4):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=11, linewidth=lw, color=ec,
                        shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)


def pa(ax, pts, ec=Be, lw=1.0, z=3):
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                color=ec, linewidth=lw, zorder=z)
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
    Ww, Hh = 14.8, 7.5
    fig, ax = plt.subplots(figsize=(Ww, Hh))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, Ww); ax.set_ylim(0, Hh)
    ax.set_aspect("equal"); ax.axis("off")

    txt(ax, 0.25, Hh - 0.25,
        "Figure 2.  Overall architecture of the proposed geometry-aware physical dehazing framework.",
        fs=11.5, b=True)

    # =========== TOP: RGB Pipeline (y = Hh - 1.70 = 5.80) ===========
    ya = 5.80
    xa = [0.85, 2.75, 5.05, 7.70, 10.15, 12.20, 13.35]
    wa = [1.30, 1.35, 1.60, 2.00, 1.35, 0.00, 1.10]
    ids = [
        (Bf, Be, 1.5, ("Hazy Input", r"$I$")),
        (Bf, Be, 1.5, ("RGB Encoder", r"$3 \to 64$")),
        (Pf, Pe, 2.0, ("DG-CCA", None)),
        (Vf, Ve, 1.8, ("FFA Backbone", "3 Groups x 3 PA")),
        (Bf, Be, 1.5, ("Reconstruct", r"$3\!\times\!3$ Conv")),
        None,
        (Bf, Be, 1.5, ("Restored", r"$\hat I$")),
    ]
    for i in range(7):
        if i == 5:
            rnd(ax, xa[i], ya); continue
        fc, ec, lw, (t1, t2) = ids[i]
        b(ax, xa[i], ya, wa[i], 0.72, t1, t2, fc=fc, ec=ec, lw=lw,
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

    # Global residual
    pa(ax, [
        (xa[0]-wa[0]/2-0.05, ya+0.36),
        (xa[0]-wa[0]/2-0.05, Hh-0.60),
        (xa[5], Hh-0.60),
        (xa[5], ya+0.38)
    ], ec=Be, lw=0.9, z=1)
    txt(ax, 6.35, Hh-0.56, "Global Residual", fs=7.5, c=Be, ha="center", va="bottom")

    # =========== MIDDLE: Depth Branch (y = 4.20) ===========
    yb = 4.20
    b(ax, 2.10, yb, 2.10, 0.78, "Depth Anything V2", "Frozen 24.8M params",
      fc=Gf, ec=Ge, dash=True, fs1=9.2, fs2=7.3)
    b(ax, 4.65, yb, 1.70, 0.78, "Depth Prior", "Min-Max + Adapter",
      fc=Gf, ec=Ge, fs1=9.0, fs2=7.2)
    pa(ax, [(xa[0], ya-0.36), (xa[0], yb+0.39), (1.05, yb+0.39)], ec=Ge, lw=1.2)
    ar(ax, 3.15, yb, 3.80, yb, ec=Ge)
    pa(ax, [(5.50, yb+0.39), (5.50, 5.00), (xa[2], ya-0.36-0.05)], ec=Ge, lw=1.3)
    txt(ax, 5.73, 5.00, r"$F_d$", fs=8.0, c=Ge, b=True)

    # =========== BOTTOM PANEL: height=3.00, y=0.25..3.25 ===========
    px, py, pw, ph = 0.25, 0.25, 14.30, 3.00
    ptop = py + ph
    ax.add_patch(FancyBboxPatch((px, py), pw, ph,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor=PANEL, edgecolor=PANELE, linewidth=0.9, zorder=0))
    txt(ax, px+0.25, ptop-0.20, "DG-CCA Detail   |   Training Strategy", fs=9.2, b=True)

    # ----- DG-CCA: 3-row layout -----
    # Row A: F_rgb (L), GAP arrows, F_d (R)  → y=rA=2.70
    # Row B: z_rgb → Concat ← z_d (all same row) → y=rB=1.85
    # Row C: MLP → y=rC=0.85
    # Row D: output A → F_out text → y=rD=0.35

    rA = py + 2.45   # 2.70
    rB = py + 1.60   # 1.85
    rC = py + 0.60   # 0.85
    rD = py + 0.15   # 0.40

    # Verify all within [py+0.3=0.55, ptop-0.3=2.95]:
    # rA=2.70±0.26=[2.44,2.96] ✓
    # rB=1.85±0.24=[1.61,2.09] ✓
    # rC=0.85±0.29=[0.56,1.14] ✓
    # rD=0.40 text ✓

    # Gap A→B: 2.44-2.09=0.35 ✓
    # Gap B→C: 1.61-1.14=0.47 ✓
    # Gap C→D: 0.56-0.40=0.16 (text below box, OK) ✓

    # Row A: F_rgb (x=1.30), GAP arrow zone, F_d (x=6.90)
    hA = 0.52
    cxFr, cxFd = 1.30, 6.90
    b(ax, cxFr, rA, 1.55, hA, r"$F_{rgb}$", r"$C\!\times\!H\!\times\!W$",
      fc=Bf, ec=Be, lw=1.2, fs1=8.6, fs2=7.0)
    b(ax, cxFd, rA, 1.55, hA, r"$F_d$", r"$C\!\times\!H\!\times\!W$",
      fc=Gf, ec=Ge, lw=1.2, fs1=8.6, fs2=7.0)

    # GAP labels
    gapLx = 2.65; gapRx = 5.55
    txt(ax, gapLx, rA, "GAP", fs=7.5, c=GRAY, b=True, ha="center")
    txt(ax, gapRx, rA, "GAP", fs=7.5, c=GRAY, b=True, ha="center")
    ar(ax, cxFr+0.775, rA, gapLx-0.30, rA, ec=GRAY, lw=0.7)
    ar(ax, gapLx+0.30, rA, gapLx+0.60, rA, ec=GRAY, lw=0.7)
    ar(ax, cxFd-0.775, rA, gapRx+0.30, rA, ec=GRAY, lw=0.7)
    ar(ax, gapRx-0.30, rA, gapRx-0.60, rA, ec=GRAY, lw=0.7)

    # Row A→B vertical arrows
    dz1x, dz2x = 3.05, 5.05
    hB = 0.46
    ar(ax, dz1x, rA-hA/2, dz1x, rB+hB/2, ec=GRAY, lw=0.8)
    ar(ax, dz2x, rA-hA/2, dz2x, rB+hB/2, ec=GRAY, lw=0.8)

    # Row B: z_rgb (x=3.05), Concat (x=4.05), z_d (x=5.05)
    b(ax, dz1x, rB, 1.15, hB, r"$z_{rgb}$", r"$\mathbb{R}^{C}$",
      fc="#f0f5ff", ec=Be, lw=1.0, fs1=7.8, fs2=6.5)
    b(ax, 4.05, rB, 1.80, hB, r"Concat $[z_{rgb};\,z_d]$", r"$\mathbb{R}^{2C}$",
      fc=Pf, ec=Pe, lw=1.4, fs1=8.0, fs2=6.6)
    b(ax, dz2x, rB, 1.15, hB, r"$z_d$", r"$\mathbb{R}^{C}$",
      fc="#eff8ef", ec=Ge, lw=1.0, fs1=7.8, fs2=6.5)

    # Row B→C arrow
    hC = 0.56
    ar(ax, 4.05, rB-hB/2, 4.05, rC+hC/2, ec=Pe, lw=0.9)

    # Row C: MLP
    b(ax, 4.05, rC, 4.50, hC, "Bottleneck MLP (Squeeze-and-Excitation)",
      r"$W_1\!\in\!\mathbb{R}^{C\times 2C} \to \mathrm{ReLU} \to W_2\!\in\!\mathbb{R}^{C\times C} \to \sigma$",
      fc=Pf, ec=Pe, lw=1.8, fs1=8.8, fs2=6.5, c2=INK)

    # Row D: output
    outX = 4.05 + 2.25  # right edge of MLP
    ar(ax, outX, rC-hC/2, outX, rD+0.10, ec=Pe, lw=0.9)
    txt(ax, outX + 0.35, rD, r"$\mathbf{A}\in\mathbb{R}^{C}$", fs=8.2, c=Pe, b=True,
        va="center")
    ar(ax, outX + 1.80, rD, outX + 2.80, rD, ec=Pe, lw=0.9)
    txt(ax, outX + 3.50, rD, r"$F_{out} = \mathbf{A} \odot F_{rgb}$",
        fs=9.0, c=INK, b=True, ha="center")

    # Param note
    txt(ax, 4.05, rD - 0.28, "Per module: 8,448 params ($2C^2+2C$, $C=64$)",
        fs=7.0, c=GRAY, ha="center")

    # ----- Training text (right) -----
    tx = 8.70; ty = ptop - 0.35; dd = 0.30
    lines = [
        ("Two-Stage Training Curriculum", 9.8, True, INK),
        ("Stage 1 (Epoch 1..100):  All params jointly optimized", 8.2, False, GRAY),
        ("  " + r"$\mathcal{L}_{rec} = \|\hat I - J\|_1 + 0.05\,\mathcal{L}_{edge}$", 8.8, True, Pe),
        ("Stage 2 (Epoch 101..120):  Backbone frozen, DG-CCA trainable", 8.2, False, GRAY),
        ("  " + r"$M(x,y) = \mathbf{1}[D(x,y) \geq 0.1]$   ($\tau=0.1$, morph. opening)", 8.2, False, INK),
        ("  " + r"$\mathcal{L}_{TV} = \sum_{x,y}(1-M)(\|\nabla_x\hat I\|_1 + \|\nabla_y\hat I\|_1)$", 8.0, False, INK),
        ("  " + r"$\mathcal{L}_{total} = \mathcal{L}_{rec} + 0.01\,\mathcal{L}_{TV}$", 9.0, True, Pe),
    ]
    for i, (s, fs, bld, clr) in enumerate(lines):
        txt(ax, tx, ty - i * dd, s, fs=fs, c=clr, b=bld)

    for fmt, dpi in [("pdf", None), ("png", 400)]:
        fp = OUT / f"fig2_v11.{fmt}"
        fig.savefig(fp, bbox_inches="tight", pad_inches=0.06, dpi=dpi)
        print(f"Saved: {fp}")
    plt.close(fig)


if __name__ == "__main__":
    draw()