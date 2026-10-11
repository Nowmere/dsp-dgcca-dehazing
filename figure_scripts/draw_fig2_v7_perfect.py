"""Figure 2 V7: Perfect layout - no overlaps, clean topology.
Uses striped-row approach: each element occupies a unique (cx, cy) slot.
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
C = {
    "bf": "#e3ecf7", "be": "#2e609a",   # blue
    "gf": "#e2f0e5", "ge": "#3d7f53",   # green
    "pf": "#fce4ec", "pe": "#b71c50",   # pink
    "vf": "#ede7f6", "ve": "#5e35b1",   # purple
    "panel": "#f7f9fc", "panel_e": "#c4cad4",
    "white": "#ffffff",
}

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
      fc=C["bf"], ec=C["be"], lw=1.5, dash=False, fs1=9.0, fs2=7.2, z=5):
    """Draw a rounded box centered at (cx, cy)."""
    ls = (0, (4, 3)) if dash else "solid"
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.10,rounding_size=0.07",
                       facecolor=fc, edgecolor=ec,
                       linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    if t2:
        ax.text(cx, cy + 0.10, t1, ha="center", va="center",
                fontsize=fs1, fontweight="bold", color=INK, zorder=z + 1)
        ax.text(cx, cy - 0.14, t2, ha="center", va="center",
                fontsize=fs2, color=GRAY, zorder=z + 1)
    else:
        ax.text(cx, cy, t1, ha="center", va="center",
                fontsize=fs1, fontweight="bold", color=INK, zorder=z + 1)


def ar(ax, x0, y0, x1, y1, ec=C["be"], lw=1.2, z=4):
    """Arrow with head."""
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=11, linewidth=lw, color=ec,
                        shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)


def pa(ax, pts, ec=C["be"], lw=1.0, z=3):
    """Polyline path: segments + arrowhead at end."""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=ec, linewidth=lw, zorder=z)
    ar(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], ec=ec, lw=lw, z=z)


def ln(ax, x0, y0, x1, y1, ec=GRAY, lw=0.7, z=3):
    """Line segment (no arrow)."""
    ax.plot([x0, x1], [y0, y1], color=ec, linewidth=lw,
            solid_capstyle="round", zorder=z)


def txt(ax, x, y, s, fs=8.0, c=INK, b=False, ha="left", va="center", z=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs,
            fontweight="bold" if b else "normal", color=c, zorder=z)


def oplus(ax, x, y):
    c = Circle((x, y), 0.16, facecolor=C["white"], edgecolor=C["be"],
               linewidth=1.3, zorder=6)
    ax.add_patch(c)
    ax.text(x, y, "+", ha="center", va="center", fontsize=12,
            fontweight="bold", color=INK, zorder=7)


# =====================================================================
def draw():
    W, H = 14.8, 5.3
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    txt(ax, 0.25, 5.10, "Figure 2.  Overall architecture of the proposed geometry-aware physical dehazing framework.",
        fs=11.5, b=True)

    # =================================================================
    # ROW A — RGB Pipeline (y = 3.85)
    # =================================================================
    YA = 3.85
    # Box centers: strictly spaced, no overlap
    xA = [0.85, 2.75, 5.05, 7.70, 10.15, 12.20, 13.35]
    wA = [1.30, 1.35, 1.60, 2.00, 1.35, 0.00, 1.10]  # 0 = plus circle
    hA = 0.72
    labels = [
        ("Hazy Input", r"$I$"),
        ("RGB Encoder", r"$3 \to 64$"),
        ("DG-CCA", None),
        ("FFA Backbone", "3 Groups x 3 PA Blocks"),
        ("Reconstruct", r"$3\!\times\!3$ Conv"),
        None,  # plus
        ("Restored", r"$\hat I$"),
    ]
    fills = [C["bf"], C["bf"], C["pf"], C["vf"], C["bf"], None, C["bf"]]
    edges = [C["be"], C["be"], C["pe"], C["ve"], C["be"], None, C["be"]]
    lws  = [1.5, 1.5, 2.0, 1.8, 1.5, None, 1.5]

    for i in range(7):
        if i == 5:
            oplus(ax, xA[i], YA)
            continue
        t1, t2 = labels[i]
        b(ax, xA[i], YA, wA[i], hA, t1, t2,
          fc=fills[i], ec=edges[i], lw=lws[i],
          fs1=10.5 if i == 2 else 9.0)

    # Horizontal arrows
    horz = [
        (xA[0] + wA[0]/2, xA[1] - wA[1]/2, C["be"]),
        (xA[1] + wA[1]/2, xA[2] - wA[2]/2, C["be"]),
        (xA[2] + wA[2]/2, xA[3] - wA[3]/2, C["pe"]),
        (xA[3] + wA[3]/2, xA[4] - wA[4]/2, C["ve"]),
        (xA[4] + wA[4]/2, xA[5] - 0.18, C["be"]),
        (xA[5] + 0.18, xA[6] - wA[6]/2, C["be"]),
    ]
    for x0, x1, ec in horz:
        ar(ax, x0, YA, x1, YA, ec=ec)

    # Global residual (above)
    pa(ax, [
        (xA[0] - wA[0]/2 - 0.05, YA + hA/2),
        (xA[0] - wA[0]/2 - 0.05, 4.70),
        (xA[5], 4.70),
        (xA[5], YA + hA/2 + 0.30)
    ], ec=C["be"], lw=0.9, z=1)
    txt(ax, 6.35, 4.74, "Global Residual", fs=7.5, c=C["be"], ha="center", va="bottom")

    # =================================================================
    # ROW B — Depth Branch (y = 2.35)
    # =================================================================
    YB = 2.35
    b(ax, 2.10, YB, 2.10, 0.80, "Depth Anything V2", "Frozen (24.8M params)",
      fc=C["gf"], ec=C["ge"], dash=True, fs1=9.2, fs2=7.3)
    b(ax, 4.70, YB, 1.70, 0.80, "Depth Prior", "Min-Max + Adapter",
      fc=C["gf"], ec=C["ge"], fs1=9.0, fs2=7.2)

    # Hazy → DAV2 (vertical)
    pa(ax, [(xA[0], YA - hA/2), (xA[0], YB + 0.40), (1.05, YB + 0.40)],
       ec=C["ge"], lw=1.2)
    # DAV2 → Depth Prior
    ar(ax, 3.15, YB, 3.85, YB, ec=C["ge"])
    # Depth Prior → DG-CCA (vertical injection)
    pa(ax, [
        (5.55, YB + 0.40), (5.55, 3.15), (xA[2], YA - hA/2 - 0.05)
    ], ec=C["ge"], lw=1.3)
    txt(ax, 5.78, 3.15, r"$F_d$", fs=8.0, c=C["ge"], b=True)

    # =================================================================
    # ROW C — Bottom Panel: DG-CCA Detail + Training
    # =================================================================
    PX, PY, PW, PH = 0.25, 0.25, 14.30, 1.80
    pbg = FancyBboxPatch((PX, PY), PW, PH,
                         boxstyle="round,pad=0.01,rounding_size=0.05",
                         facecolor=C["panel"], edgecolor=C["panel_e"],
                         linewidth=0.9, zorder=0)
    ax.add_patch(pbg)
    txt(ax, PX + 0.25, PY + PH - 0.20, "DG-CCA Detail   |   Training Strategy",
        fs=9.2, b=True)

    # --- DG-CCA sub-detail (left half, x 0.5..8.5) ---
    # Rows inside panel (y coordinates from bottom panel):
    r1 = PY + 1.25   # F_rgb, F_d
    r2 = PY + 0.78   # z_rgb, z_d
    r3 = PY + 0.38   # Concat
    r4 = PY - 0.05   # MLP (extends downward)
    r5 = PY - 0.40   # output labels (below MLP)

    # F_rgb at x=1.30
    cx_fr = 1.30; w_fr = 1.50; h_fr = 0.48
    b(ax, cx_fr, r1, w_fr, h_fr, r"$F_{rgb}$", r"$C\!\times\!H\!\times\!W$",
      fc=C["bf"], ec=C["be"], lw=1.2, fs1=8.5, fs2=7.0)

    # F_d at x=6.50
    cx_fd = 6.50; w_fd = 1.50; h_fd = 0.48
    b(ax, cx_fd, r1, w_fd, h_fd, r"$F_d$", r"$C\!\times\!H\!\times\!W$",
      fc=C["gf"], ec=C["ge"], lw=1.2, fs1=8.5, fs2=7.0)

    # GAP indicators (text + arrows between F_rgb→GAP, GAP→F_d)
    gapL = cx_fr + 1.35  # 2.65
    gapR = cx_fd - 1.35  # 5.15
    ar(ax, cx_fr + w_fr/2, r1, gapL - 0.35, r1, ec=GRAY, lw=0.8)
    txt(ax, gapL, r1, "GAP", fs=7.5, c=GRAY, b=True, ha="center")
    ar(ax, gapL + 0.35, r1, gapL + 0.65, r1, ec=GRAY, lw=0.8)

    ar(ax, cx_fd - w_fd/2, r1, gapR + 0.35, r1, ec=GRAY, lw=0.8)
    txt(ax, gapR, r1, "GAP", fs=7.5, c=GRAY, b=True, ha="center")
    ar(ax, gapR - 0.35, r1, gapR - 0.65, r1, ec=GRAY, lw=0.8)

    # Drop from GAP to z_rgb / z_d
    dz1_x = gapL + 0.65  # 3.30
    dz2_x = gapR - 0.65  # 4.50
    ln(ax, dz1_x, r1 - h_fr/2, dz1_x, r2 + 0.32, GRAY, 0.7)
    ar(ax, dz1_x, r2 + 0.18, dz1_x, r2 + 0.05, GRAY, 0.8)
    ln(ax, dz2_x, r1 - h_fd/2, dz2_x, r2 + 0.32, GRAY, 0.7)
    ar(ax, dz2_x, r2 + 0.18, dz2_x, r2 + 0.05, GRAY, 0.8)

    # z_rgb at x=3.30, z_d at x=4.50
    w_z = 1.05; h_z = 0.40
    b(ax, dz1_x, r2, w_z, h_z, r"$z_{rgb}$", r"$\mathbb{R}^{C}$",
      fc="#f0f5ff", ec=C["be"], lw=1.0, fs1=7.8, fs2=6.5)
    b(ax, dz2_x, r2, w_z, h_z, r"$z_d$", r"$\mathbb{R}^{C}$",
      fc="#eff8ef", ec=C["ge"], lw=1.0, fs1=7.8, fs2=6.5)

    # Concat at x=3.90 (center between z_rgb and z_d), ONE row below
    cx_cc = 3.90; w_cc = 2.00; h_cc = 0.45
    b(ax, cx_cc, r3, w_cc, h_cc, r"Concat $[z_{rgb};\,z_d]$", r"$\mathbb{R}^{2C}$",
      fc=C["pf"], ec=C["pe"], lw=1.3, fs1=7.8, fs2=6.5)

    # Arrows from z boxes down into concat
    ar(ax, dz1_x, r2 - h_z/2, dz1_x, r3 + h_cc/2, GRAY, 0.7)
    ar(ax, dz2_x, r2 - h_z/2, dz2_x, r3 + h_cc/2, GRAY, 0.7)

    # MLP at x=3.90
    w_mlp = 4.00; h_mlp = 0.55
    b(ax, cx_cc, r4, w_mlp, h_mlp, "Bottleneck MLP",
      r"$W_1\;(\mathbb{R}^{C\times 2C}) \to \mathrm{ReLU} \to W_2\;(\mathbb{R}^{C\times C}) \to \sigma$",
      fc=C["pf"], ec=C["pe"], lw=1.6, fs1=8.5, fs2=6.5, c2=INK)
    ar(ax, cx_cc, r3 - h_cc/2, cx_cc, r4 + h_mlp/2, ec=C["pe"], lw=0.9)

    # Output A
    ar(ax, cx_cc + w_mlp/2, r4 - h_mlp/2, cx_cc + w_mlp/2, r5 + 0.18, ec=C["pe"], lw=0.9)
    txt(ax, cx_cc + w_mlp/2 + 0.35, r5, r"$\mathbf{A}\in\mathbb{R}^{C}$", fs=8.2,
        c=C["pe"], b=True)
    ar(ax, cx_cc + w_mlp/2 + 1.80, r5, cx_cc + w_mlp/2 + 2.50, r5, ec=C["pe"], lw=0.9)
    txt(ax, cx_cc + w_mlp/2 + 3.40, r5, r"$F_{out} = \mathbf{A} \odot F_{rgb}$",
        fs=9.0, c=INK, b=True, ha="center")

    # --- RIGHT SIDE: Training text ---
    tx = 8.70
    ty = PY + 1.50
    dd = 0.265
    txt(ax, tx, ty, "Two-Stage Training Curriculum", fs=9.5, b=True)
    txt(ax, tx, ty - dd, "Stage 1 (Epoch 1..100):   All parameters jointly optimized",
        fs=8.0, c=GRAY)
    txt(ax, tx + 0.15, ty - 2*dd,
        r"$\mathcal{L}_{rec} = \|\hat I - J\|_1 + 0.05\,\mathcal{L}_{edge}$",
        fs=8.5, c=C["pe"], b=True)
    txt(ax, tx, ty - 3.5*dd, "Stage 2 (Epoch 101..120): Backbone frozen, DG-CCA trainable",
        fs=8.0, c=GRAY)
    txt(ax, tx + 0.15, ty - 4.5*dd,
        r"$M(x,y)=\mathbf{1}[D(x,y)\geq 0.1]$    (Hard sky mask,   $\tau=0.1$)",
        fs=8.0, c=INK)
    txt(ax, tx + 0.15, ty - 5.5*dd,
        r"$\mathcal{L}_{TV} = \sum_{x,y}(1-M)(\|\nabla_x\hat I\|_1 + \|\nabla_y\hat I\|_1)$",
        fs=7.8, c=INK)
    txt(ax, tx + 0.15, ty - 6.5*dd,
        r"$\mathcal{L}_{total} = \mathcal{L}_{rec} + 0.01\,\mathcal{L}_{TV}$",
        fs=8.8, c=C["pe"], b=True)

    # ── SAVE ──
    for fmt, dpi in [("pdf", None), ("png", 400)]:
        fpath = OUT / f"fig2_v7_perfect.{fmt}"
        fig.savefig(fpath, bbox_inches="tight", pad_inches=0.06, dpi=dpi)
        print(f"Saved: {fpath}")
    plt.close(fig)


if __name__ == "__main__":
    draw()