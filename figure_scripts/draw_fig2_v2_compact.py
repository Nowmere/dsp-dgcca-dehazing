"""Figure 2 Version 2: 紧凑科学图表型 (Compact Scientific Layout)
Clean, publication-quality figure with multi-scale architecture.
Includes detailed DG-CCA inset and training curriculum.

Author: Architecture Visualization
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper" / "CVPR_2026_Submission_Template" / "figures"
OUTPUT_STEM = OUTPUT_DIR / "fig2_v2_compact_scientific"

# ── Colors ──
C = {
    "ink": "#1a1a2e",
    "muted": "#556677",
    "rgb": "#e3ecf6",
    "rgb_e": "#2a6090",
    "depth": "#e2f0e5",
    "depth_e": "#3a7d4f",
    "fusion": "#fde0e8",
    "fusion_e": "#b71c4c",
    "backbone": "#eee5f7",
    "backbone_e": "#673ab7",
    "loss": "#ffe8cc",
    "loss_e": "#d84315",
    "sky": "#e1f5fe",
    "sky_e": "#0277bd",
    "panel": "#f8f9fc",
    "panel_e": "#c5cad4",
    "white": "#ffffff",
}

def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavuserif",
        "savefig.facecolor": "white",
    })

def bbox(ax, x, y, w, h, txt, sub=None, *, fill=C["rgb"], edge=C["rgb_e"],
         lw=1.5, dashed=False, fs1=8.3, fs2=6.8):
    ls = (0, (4, 2.5)) if dashed else "solid"
    p = FancyBboxPatch((x-w/2, y-h/2), w, h,
                       boxstyle="round,pad=0.12,rounding_size=0.07",
                       facecolor=fill, edgecolor=edge, linewidth=lw,
                       linestyle=ls, zorder=4)
    ax.add_patch(p)
    if sub:
        ax.text(x, y+0.09, txt, ha="center", va="center", fontsize=fs1,
                fontweight="bold", color=C["ink"], zorder=5)
        ax.text(x, y-0.12, sub, ha="center", va="center", fontsize=fs2,
                color=C["muted"], zorder=5)
    else:
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs1,
                fontweight="bold", color=C["ink"], zorder=5)

def arr(ax, x0, y0, x1, y1, color=C["rgb_e"], lw=1.1, z=3, dashes=False):
    ls = (0, (4, 2.5)) if dashes else "solid"
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=11, linewidth=lw, linestyle=ls,
                        color=color, shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)

def poly(ax, pts, color=C["rgb_e"], lw=1.0, z=2):
    for i in range(len(pts)-2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                color=color, linewidth=lw, solid_capstyle="round", zorder=z)
    arr(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color, lw=lw, z=z)

def panel(ax, x, y, w, h, title):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                       facecolor=C["panel"], edgecolor=C["panel_e"],
                       linewidth=0.9, zorder=0)
    ax.add_patch(p)
    ax.text(x+0.22, y+h-0.20, title, ha="left", va="center",
            fontsize=8.8, fontweight="bold", color=C["ink"], zorder=6)

def oplus(ax, x, y):
    c = Circle((x, y), 0.16, facecolor=C["white"], edgecolor=C["rgb_e"],
               linewidth=1.2, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, "+", ha="center", va="center", fontsize=12,
            fontweight="bold", color=C["ink"], zorder=6)

def seg(ax, x0, y0, x1, y1, color=C["muted"], lw=0.8, z=2):
    """Draw a line segment (no arrow)."""
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
            solid_capstyle="round", zorder=z)

# =====================================================================
def draw():
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 7.2)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0.25, 6.92, "Geometry-Aware Dehazing Architecture (DG-CCA + Regional Sky TV Fine-Tuning)",
            fontsize=12.5, fontweight="bold", color=C["ink"])

    # =================================================================
    # TOP HALF: Main pipeline
    # =================================================================
    YT = 5.0
    YB = 2.8

    # ── RGB Track ──
    items = [
        (0.75, 1.15, 0.72, ("Hazy $I$", r"$3\times H\times W$"), C["rgb"], C["rgb_e"], 1.5, 9.0),
        (2.65, 1.25, 0.72, ("Encoder", r"$3\rightarrow 64$"), C["rgb"], C["rgb_e"], 1.5, 8.8),
        (4.90, 1.65, 0.82, ("DG-CCA", ), C["fusion"], C["fusion_e"], 2.0, 10.2),
        (7.80, 2.00, 0.82, ("FFA Backbone", "3 Groups x 3 PA"), C["backbone"], C["backbone_e"], 1.8, 9.0),
        (10.40, 1.25, 0.72, ("Reconstruct", r"$3\times 3$ Conv"), C["rgb"], C["rgb_e"], 1.5, 8.6),
        (12.30, 0.65, 0.60, ("Restored", r"$\hat I$"), C["rgb"], C["rgb_e"], 1.5, 9.2),
    ]

    for i, (cx, w, h, txt, fill, edge, lw, fs1) in enumerate(items):
        if i == 4:  # position for plus node is 12.10
            oplus(ax, 12.10, YT)
        sub = txt[1] if len(txt) > 1 else None
        bbox(ax, cx, YT, w, h, txt[0], sub, fill=fill, edge=edge, lw=lw, fs1=fs1)

    # Arrows along RGB track
    rgb_arrow_pairs = [
        (1.33, 2.02, C["rgb_e"]),
        (3.27, 4.08, C["fusion_e"]),
        (5.72, 6.80, C["backbone_e"]),
        (8.80, 9.78, C["rgb_e"]),
        (10.40, 11.94, C["rgb_e"]),
        (12.26, 12.80, C["rgb_e"]),
    ]
    for x0, x1, color in rgb_arrow_pairs:
        arr(ax, x0, YT, x1, YT, color=color, lw=1.2)

    # Global residual (top)
    poly(ax, [
        (1.20, YT+0.36), (1.20, 6.60), (12.10, 6.60), (12.10, YT+0.52)
    ], color=C["rgb_e"], lw=0.9, z=1)
    ax.text(6.60, 6.66, r"Global Residual $I + R(I)$", fontsize=7.2, color=C["rgb_e"],
            ha="center", zorder=5)

    # ── Depth Track ──
    bbox(ax, 2.10, YB, 2.10, 0.78, "Depth Anything V2",
         "Frozen · 24.8M params", fill=C["depth"], edge=C["depth_e"],
         lw=1.5, dashed=True, fs1=8.8)
    poly(ax, [(0.90, YT-0.36), (0.90, YB+0.39), (1.05, YB+0.39)],
         color=C["depth_e"], lw=1.2)
    ax.text(0.62, 3.95, "share input", fontsize=7.0, color=C["depth_e"],
            ha="center", rotation=90, zorder=5)

    bbox(ax, 4.40, YB, 1.50, 0.78, "Depth Prior", "Min-Max → [0,1]",
         fill=C["depth"], edge=C["depth_e"], fs1=8.5)
    arr(ax, 3.15, YB, 3.65, YB, color=C["depth_e"], lw=1.2)

    bbox(ax, 6.50, YB, 1.40, 0.78, "Depth Encoder", r"$1\times 1$: $1\rightarrow 64$",
         fill=C["depth"], edge=C["depth_e"], fs1=8.3, fs2=6.8)
    arr(ax, 5.15, YB, 5.80, YB, color=C["depth_e"], lw=1.2)

    # Depth injection to DG-CCA
    poly(ax, [
        (7.20, YB+0.39), (7.35, YB+0.39), (7.35, 3.90), (4.90, YT-0.41)
    ], color=C["depth_e"], lw=1.3)
    ax.text(7.55, 3.90, r"$F_d$", fontsize=7.5, color=C["depth_e"],
            fontweight="bold", ha="left", zorder=5)

    # =================================================================
    # BOTTOM HALF
    # =================================================================
    py = 0.30

    # ── Left Panel: DG-CCA Detail ──
    panel(ax, 0.25, py, 8.40, 2.10, "(a) DG-CCA: Channel-wise Cross-modal Attention (Squeeze-and-Excitation style)")

    bx = 0.75
    # F_rgb and F_d on top row
    bbox(ax, bx, py+1.30, 1.50, 0.52, r"$F_{rgb}$", r"$C\times H\times W$",
         fill=C["rgb"], edge=C["rgb_e"], fs1=8.8)
    # F_d at bx+6.00
    bbox(ax, bx+6.00, py+1.30, 1.50, 0.52, r"$F_d$", r"$C\times H\times W$",
         fill=C["depth"], edge=C["depth_e"], fs1=8.8)

    # GAP arrows (F_rgb → right, then down)
    arr(ax, bx+0.75, py+1.30, bx+1.30, py+1.30, C["muted"], 0.8)
    seg(ax, bx+1.30, py+1.30, bx+1.30, py+1.04, C["muted"], 0.8)
    arr(ax, bx+1.30, py+1.04, bx+1.30, py+0.72, C["muted"], 0.8)

    # GAP arrows (F_d ← left, then down)
    # F_d center = bx+6.00, left edge = bx+6.00-0.75 = bx+5.25
    arr(ax, bx+5.25, py+1.30, bx+4.70, py+1.30, C["muted"], 0.8)
    seg(ax, bx+4.70, py+1.30, bx+4.70, py+1.04, C["muted"], 0.8)
    arr(ax, bx+4.70, py+1.04, bx+4.70, py+0.72, C["muted"], 0.8)

    # GAP pools
    bbox(ax, bx+0.70, py+0.52, 1.20, 0.42, r"$\mathbf{z}_{rgb}$", r"$\mathbb{R}^{C}$",
         fill="#f0f5ff", edge=C["rgb_e"], lw=1.0, fs1=7.8, fs2=6.6)
    bbox(ax, bx+5.30, py+0.52, 1.20, 0.42, r"$\mathbf{z}_d$", r"$\mathbb{R}^{C}$",
         fill="#eff8ef", edge=C["depth_e"], lw=1.0, fs1=7.8, fs2=6.6)

    # Arrows: z_rgb → right → down; z_d ← left → down
    arr(ax, bx+1.30, py+0.52, bx+2.20, py+0.52, C["muted"], 0.8)
    arr(ax, bx+4.70, py+0.52, bx+3.80, py+0.52, C["muted"], 0.8)
    seg(ax, bx+2.20, py+0.52, bx+2.20, py+0.31, C["muted"], 0.8)
    seg(ax, bx+3.80, py+0.52, bx+3.80, py+0.31, C["muted"], 0.8)
    arr(ax, bx+2.20, py+0.31, bx+2.20, py+0.02, C["muted"], 0.8)
    arr(ax, bx+3.80, py+0.31, bx+3.80, py+0.02, C["muted"], 0.8)

    # Concat
    bbox(ax, bx+2.75, py-0.28, 2.50, 0.58, r"Concat $[\mathbf{z}_{rgb};\,\mathbf{z}_d]$",
         r"$\mathbb{R}^{2C}$", fill=C["fusion"], edge="#c2185b", lw=1.4, fs1=8.0, fs2=6.6)

    # MLP
    arr(ax, bx+4.00, py-0.57, bx+4.00, py-0.95, C["fusion_e"], 0.9)
    bbox(ax, bx+2.65, py-1.25, 2.70, 0.62, "Bottleneck MLP",
         r"$W_1(\mathbb{R}^{C\times 2C})\rightarrow\mathrm{ReLU}\rightarrow W_2(\mathbb{R}^{C\times C})\rightarrow\sigma$",
         fill="#fff0f5", edge=C["fusion_e"], lw=1.6, fs1=8.8, fs2=6.5)

    # Output gate
    arr(ax, bx+4.00, py-1.56, bx+4.00, py-1.78, C["fusion_e"], 0.9)
    ax.text(bx+4.40, py-1.80, r"$\mathbf{A}\in\mathbb{R}^{C}$", fontsize=8.2,
            color=C["fusion_e"], fontweight="bold", ha="left", zorder=5)
    ax.text(bx+6.45, py-1.80, r"$F_{out}=\mathbf{A}\odot F_{rgb}$",
            fontsize=9.0, color=C["ink"], fontweight="bold", ha="center", zorder=5)
    arr(ax, bx+5.40, py-1.80, bx+5.85, py-1.80, C["fusion_e"], 0.9)

    # Param note
    ax.text(bx+4.00, py-2.07, "Per module: 8,448 params ($2C^2+2C$, $C=64$)",
            fontsize=7.0, color=C["muted"], ha="center", zorder=5)

    # ── Right Panel: Training ──
    sx = 8.95
    sw = 5.30
    panel(ax, sx, py, sw, 2.10, "(b) Curriculum & Regional Decoupled Fine-Tuning")

    ax.text(sx+0.25, py+1.75, "Phase I: Global Training (Epoch 1–100)",
            fontsize=8.8, fontweight="bold", color=C["ink"], zorder=5)
    ax.text(sx+0.35, py+1.45, r"$\mathcal{L}_{rec} = \|\hat I - J\|_1 + 0.05\,\mathcal{L}_{edge}$",
            fontsize=9.0, color=C["loss_e"], zorder=5)
    ax.text(sx+0.35, py+1.22, "— All parameters trainable; DG-CCA fully active",
            fontsize=7.0, color=C["muted"], zorder=5)

    ax.plot([sx+0.30, sx+sw-0.25], [py+1.05, py+1.05], color=C["panel_e"],
            linewidth=0.6, linestyle="--", zorder=2)

    ax.text(sx+0.25, py+0.90, "Phase II: Sky TV Fine-Tuning (Epoch 101–120)",
            fontsize=8.8, fontweight="bold", color=C["ink"], zorder=5)
    ax.text(sx+0.35, py+0.60, r"$\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda\,\mathcal{L}_{TV}$  ($\lambda=0.01$)",
            fontsize=8.5, color=C["ink"], zorder=5)
    ax.text(sx+0.35, py+0.40, "— Backbone frozen; DG-CCA & Adapter still trainable",
            fontsize=7.0, color=C["muted"], zorder=5)

    ax.text(sx+0.25, py+0.15, "Hard Mask & Sky TV Loss:",
            fontsize=7.8, fontweight="bold", color=C["ink"], zorder=5)
    ax.text(sx+0.35, py-0.06, r"$M(x,y)=\mathbf{1}[D(x,y)\geq\tau],\;\tau=0.1$",
            fontsize=8.2, color=C["sky_e"], fontweight="bold", zorder=5)
    ax.text(sx+0.35, py-0.28, r"$\mathcal{L}_{TV}=\sum_{x,y}(1-M)(\|\nabla_x\hat I\|_1+\|\nabla_y\hat I\|_1)$",
            fontsize=7.8, color=C["loss_e"], zorder=5)
    ax.text(sx+0.35, py-0.50, "Morphological opening to remove spurious fragments",
            fontsize=6.8, color=C["muted"], zorder=5)

    # Save
    for fmt, dpi in [("pdf", None), ("png", 360)]:
        path = OUTPUT_STEM.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08, dpi=dpi)
        print(f"Saved: {path}")
    plt.close(fig)

if __name__ == "__main__":
    draw()