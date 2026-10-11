"""Figure 2 Version 3: 垂直堆叠型 (Vertical Stack Architecture)
Top-to-bottom pipeline with central DG-CCA detail and expanded equation display.
Ideal for single-column or portrait layouts.

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
OUTPUT_STEM = OUTPUT_DIR / "fig2_v3_vertical_stack"

# ── Palette ──
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
    ax.text(x+0.22, y+h-0.22, title, ha="left", va="center",
            fontsize=9.0, fontweight="bold", color=C["ink"], zorder=6)

def oplus(ax, x, y):
    c = Circle((x, y), 0.18, facecolor=C["white"], edgecolor=C["rgb_e"],
               linewidth=1.3, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, "+", ha="center", va="center", fontsize=13,
            fontweight="bold", color=C["ink"], zorder=6)

def seg(ax, x0, y0, x1, y1, color=C["muted"], lw=0.8, z=2):
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
            solid_capstyle="round", zorder=z)

# =====================================================================
def draw():
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Taller figure for vertical stack: 10.5" wide, 8.8" tall
    fig, ax = plt.subplots(figsize=(10.5, 8.8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    W, H = 10.5, 8.8
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Title ──
    ax.text(0.20, H - 0.30, "Figure 2. Geometry-Aware Dehazing with 3D Foundation Priors",
            fontsize=12.0, fontweight="bold", color=C["ink"], zorder=6)

    # =================================================================
    # LEFT COLUMN: Main Pipeline (vertical)
    # =================================================================
    cx = 2.80  # center X for the main column

    y_positions = [H - 0.90, H - 1.90, H - 2.85, H - 3.95, H - 4.90, H - 5.90, H - 7.05]
    box_w = 2.60
    box_h = 0.70

    # ── 1. Hazy Input ──
    bbox(ax, cx, y_positions[0], box_w, box_h, "Hazy Input $I$", r"$3\times H\times W$",
         fill=C["rgb"], edge=C["rgb_e"], lw=1.5, fs1=9.5, fs2=7.5)

    # ── 2. DAV2 (side by side with Encoder) ──
    bbox(ax, cx - 1.80, y_positions[1], 1.90, box_h, "Depth Anything V2",
         "Frozen · 24.8M", fill=C["depth"], edge=C["depth_e"], lw=1.5, dashed=True, fs1=8.8, fs2=7.0)
    bbox(ax, cx + 1.10, y_positions[1], 1.60, box_h, "RGB Encoder", r"$3\!\rightarrow\!64$",
         fill=C["rgb"], edge=C["rgb_e"], lw=1.5, fs1=9.0, fs2=7.2)

    # Arrows: Hazy → DAV2 (left), Hazy → Encoder (right)
    poly(ax, [
        (cx-0.40, y_positions[0]-box_h/2), (cx-0.40, y_positions[1]+box_h/2)
    ], color=C["depth_e"], lw=1.2)
    poly(ax, [
        (cx+0.40, y_positions[0]-box_h/2), (cx+0.40, y_positions[1]+box_h/2)
    ], color=C["rgb_e"], lw=1.2)

    # ── 3. Depth Prior & Encoder ──
    bbox(ax, cx - 1.80, y_positions[2], 1.90, box_h, "Depth Prior", "Norm + $1\\times 1$ Conv",
         fill=C["depth"], edge=C["depth_e"], lw=1.5, fs1=8.5, fs2=7.0)
    bbox(ax, cx + 1.10, y_positions[2], 1.60, box_h, "RGB Features", "$F_{rgb}$",
         fill=C["rgb"], edge=C["rgb_e"], lw=1.5, fs1=9.0, fs2=7.2)

    arr(ax, cx-1.80, y_positions[1]-box_h/2, cx-1.80, y_positions[2]+box_h/2, color=C["depth_e"], lw=1.2)
    arr(ax, cx+1.10, y_positions[1]-box_h/2, cx+1.10, y_positions[2]+box_h/2, color=C["rgb_e"], lw=1.2)

    # ── 4. DG-CCA (center, wider) ──
    bbox(ax, cx, y_positions[3], 3.80, 0.82, "DG-CCA: Cross-modal Channel Fusion",
         r"GAP $\rightarrow$ Concat $\rightarrow$ MLP $\rightarrow$ $\sigma$ $\rightarrow$ $\odot$",
         fill=C["fusion"], edge=C["fusion_e"], lw=2.5, fs1=10.5, fs2=7.5)

    # Arrows: RGB features → DG-CCA, Depth → DG-CCA
    poly(ax, [
        (cx+1.10, y_positions[2]-box_h/2), (cx+1.10, y_positions[3]+0.41)
    ], color=C["rgb_e"], lw=1.2)
    poly(ax, [
        (cx-1.80, y_positions[2]-box_h/2), (cx-1.80, y_positions[3]+0.41)
    ], color=C["depth_e"], lw=1.2)

    # ── 5. FFA Backbone ──
    bbox(ax, cx, y_positions[4], 3.40, 0.82, "FFA Backbone", "3 Groups × 3 PA Blocks",
         fill=C["backbone"], edge=C["backbone_e"], lw=2.0, fs1=9.8, fs2=7.5)
    arr(ax, cx, y_positions[3]-0.41, cx, y_positions[4]+0.41, color=C["fusion_e"], lw=1.3)

    # ── 6. Reconstruction ──
    bbox(ax, cx, y_positions[5], 2.60, box_h, "Reconstruct", r"$3\times 3$ Conv $\rightarrow$ Residual $R(I)$",
         fill=C["rgb"], edge=C["rgb_e"], lw=1.5, fs1=9.0, fs2=7.2)
    arr(ax, cx, y_positions[4]-0.41, cx, y_positions[5]+0.35, color=C["backbone_e"], lw=1.3)

    # ── 7. Output ──
    oplus(ax, cx, y_positions[6]+0.50)
    bbox(ax, cx, y_positions[6], 2.00, 0.75, "Restored $\hat I$", r"$= I + R(I)$",
         fill=C["rgb"], edge=C["rgb_e"], lw=1.5, fs1=9.8, fs2=7.5)
    arr(ax, cx, y_positions[5]-0.35, cx, y_positions[6]+0.37, color=C["rgb_e"], lw=1.2)

    # Global residual: from input down to plus node
    poly(ax, [
        (cx+1.40, y_positions[0]), (5.80, y_positions[0]), (5.80, y_positions[6]+0.50),
        (cx+0.18, y_positions[6]+0.50)
    ], color=C["rgb_e"], lw=0.9, z=1)
    ax.text(5.95, y_positions[3], "Global", fontsize=7.5, color=C["rgb_e"],
            ha="left", zorder=5)
    ax.text(5.95, y_positions[3]-0.22, "Residual", fontsize=7.5, color=C["rgb_e"],
            ha="left", zorder=5)

    # =================================================================
    # RIGHT COLUMN: Detail panels
    # =================================================================
    rx = 6.20  # right column start X
    rw = 4.00  # right column width

    # ── Panel 1: DG-CCA equation ──
    py1 = H - 1.25
    ph1 = 3.10
    panel(ax, rx, py1 - ph1, rw, ph1, "DG-CCA Module (Eq. 1–4)")

    eq_x = rx + 0.25
    eq_y = py1 - 0.35
    dy = 0.42

    ax.text(eq_x, eq_y, r"$\mathbf{z}_{rgb} = \frac{1}{HW}\sum_{i,j} F_{rgb}(i,j)$", fontsize=8.5,
            color=C["rgb_e"], fontweight="bold", zorder=5)
    ax.text(eq_x, eq_y - dy, r"$\mathbf{z}_{d} = \frac{1}{HW}\sum_{i,j} F_{d}(i,j)$", fontsize=8.5,
            color=C["depth_e"], fontweight="bold", zorder=5)
    ax.text(eq_x, eq_y - 2*dy, r"$\mathbf{A} = \sigma\left(W_2\,\mathrm{ReLU}(W_1[\mathbf{z}_{rgb};\,\mathbf{z}_d])\right)$",
            fontsize=8.8, color=C["fusion_e"], fontweight="bold", zorder=5)
    ax.text(eq_x, eq_y - 3*dy, r"$F_{out} = \mathbf{A} \odot F_{rgb}$", fontsize=9.0,
            color=C["ink"], fontweight="bold", zorder=5)

    ax.text(eq_x, eq_y - 4*dy, r"$W_1\!:\!C\!\times\!2C,\;W_2\!:\!C\!\times\!C,\;C=64$",
            fontsize=7.5, color=C["muted"], zorder=5)
    ax.text(eq_x, eq_y - 4*dy - 0.22, "8,448 parameters per module",
            fontsize=7.5, color=C["muted"], zorder=5)

    # ── Panel 2: Sky TV ──
    py2 = py1 - ph1 - 0.25
    ph2 = 3.70
    panel(ax, rx, py2 - ph2, rw, ph2, "Regional Decoupled Fine-Tuning")

    ex = rx + 0.20
    ey = py2 - 0.40
    step_h = 0.38

    # Phase I
    ax.text(ex, ey, "Phase I (Epoch 1–100): Global Training",
            fontsize=9.0, fontweight="bold", color=C["ink"], zorder=5)
    ax.text(ex + 0.10, ey - step_h, r"$\mathcal{L}_{rec} = \|\hat I - J\|_1 + 0.05\cdot \mathcal{L}_{edge}$",
            fontsize=9.0, color=C["loss_e"], zorder=5)
    ax.text(ex + 0.10, ey - 2*step_h, "All parameters trainable",
            fontsize=7.5, color=C["muted"], zorder=5)

    # Separator
    ax.plot([ex, rx+rw-0.25], [ey - 2.8*step_h, ey - 2.8*step_h],
            color=C["panel_e"], linewidth=0.6, linestyle="--", zorder=2)

    # Phase II
    p2_start = ey - 3.2*step_h
    ax.text(ex, p2_start, "Phase II (Epoch 101–120): Sky TV Fine-Tuning",
            fontsize=9.0, fontweight="bold", color=C["ink"], zorder=5)
    ax.text(ex + 0.10, p2_start - step_h, r"$\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda\,\mathcal{L}_{TV},\;\lambda=0.01$",
            fontsize=8.8, color=C["ink"], zorder=5)
    ax.text(ex + 0.10, p2_start - 2*step_h, "Backbone frozen · DG-CCA trainable",
            fontsize=7.5, color=C["muted"], zorder=5)

    # Separator
    ax.plot([ex, rx+rw-0.25], [p2_start - 2.8*step_h, p2_start - 2.8*step_h],
            color=C["panel_e"], linewidth=0.6, linestyle="--", zorder=2)

    # Hard Mask
    m_start = p2_start - 3.2*step_h
    ax.text(ex, m_start, "Hard Spatial Sky Mask:",
            fontsize=8.5, fontweight="bold", color=C["sky_e"], zorder=5)
    ax.text(ex + 0.10, m_start - step_h, r"$M(x,y) = \mathbf{1}[D(x,y) \geq \tau],\;\tau=0.1$",
            fontsize=9.0, color=C["sky_e"], fontweight="bold", zorder=5)
    ax.text(ex + 0.10, m_start - 2*step_h, r"Morphological $3\times 3$ opening to clean mask",
            fontsize=7.5, color=C["muted"], zorder=5)

    # Separator
    ax.plot([ex, rx+rw-0.25], [m_start - 2.8*step_h, m_start - 2.8*step_h],
            color=C["panel_e"], linewidth=0.6, linestyle="--", zorder=2)

    # TV Loss
    tv_start = m_start - 3.2*step_h
    ax.text(ex, tv_start, "Targeted Sky TV Loss:",
            fontsize=8.5, fontweight="bold", color=C["loss_e"], zorder=5)
    ax.text(ex + 0.10, tv_start - step_h,
            r"$\mathcal{L}_{TV} = \sum_{x,y}(1-M)(\|\nabla_x\hat I\|_1 + \|\nabla_y\hat I\|_1)$",
            fontsize=8.3, color=C["loss_e"], fontweight="bold", zorder=5)
    ax.text(ex + 0.10, tv_start - 2*step_h,
            "Enforces smooth sky gradients · Active only in sky region",
            fontsize=7.5, color=C["muted"], zorder=5)

    # Save
    for fmt, dpi in [("pdf", None), ("png", 360)]:
        path = OUTPUT_STEM.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08, dpi=dpi)
        print(f"Saved: {path}")
    plt.close(fig)

if __name__ == "__main__":
    draw()