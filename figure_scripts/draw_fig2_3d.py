"""Figure 2 — 3D Neural Network Architecture (Scheme A)
Uses matplotlib 3D axes with Poly3DCollection for tensor blocks.
Based on user-provided 3D drawing code, expanded to full architecture.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "CVPR_2026_Submission_Template" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Colors ──
C_RGB = '#D9E8F5';      E_RGB = '#2B5B84'
C_DEPTH = '#E2F0D9';    E_DEPTH = '#385723'
C_DGCCA = '#FCE4D6';    E_DGCCA = '#C65911'
C_BACKBONE = '#E8E0F0'; E_BACKBONE = '#5E35B1'
C_FROZEN = '#E7E6E6';   E_FROZEN = '#7F7F7F'
C_OUTPUT = '#FFF2CC';   E_OUTPUT = '#BF8F00'


def draw_3d_cube(ax, corner, size, color, alpha=0.6, edge_color='black', linewidth=1.2):
    """Draw a 3D cuboid tensor block."""
    x, y, z = corner
    dx, dy, dz = size
    v = np.array([
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
    ])
    faces = [
        [v[0], v[1], v[5], v[4]], [v[7], v[6], v[2], v[3]],
        [v[0], v[3], v[7], v[4]], [v[1], v[2], v[6], v[5]],
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]]
    ]
    poly = Poly3DCollection(faces, facecolors=color, linewidths=linewidth,
                            edgecolors=edge_color, alpha=alpha)
    ax.add_collection3d(poly)


def dashed_line_3d(ax, p1, p2, color, lw=1.5, dashes=(5, 4)):
    """Draw a dashed 3D line segment."""
    t = np.linspace(0, 1, 50)
    xs = p1[0] + t * (p2[0] - p1[0])
    ys = p1[1] + t * (p2[1] - p1[1])
    zs = p1[2] + t * (p2[2] - p1[2])
    ax.plot(xs, ys, zs, color=color, lw=lw, ls='--', dashes=dashes)


# =====================================================================
fig = plt.figure(figsize=(18, 10), dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([3.5, 1, 1.2])
ax.axis('off')
ax.grid(False)

# ═══════ UPPER STREAM: RGB Pipeline (z=2 plane) ═══════
# Hazy Input
draw_3d_cube(ax, (0.2, 0.5, 2), (0.6, 1.5, 1.2), C_RGB, edge_color=E_RGB)
ax.text(0.5, 1.25, 3.4, "Hazy\n$I$", fontsize=10, ha='center', va='center', weight='bold')

# RGB Encoder
draw_3d_cube(ax, (1.5, 0.5, 2), (0.6, 1.5, 1.2), C_RGB, edge_color=E_RGB)
ax.text(1.8, 1.25, 3.4, "RGB\nEncoder", fontsize=9, ha='center', va='center', weight='bold')

# DG-CCA 1
draw_3d_cube(ax, (2.8, 0.7, 2.2), (0.9, 1.1, 0.8), C_DGCCA, edge_color=E_DGCCA, alpha=0.85)
ax.text(3.25, 1.25, 2.6, "DG-CCA", fontsize=10, ha='center', va='center', weight='bold', color=E_DGCCA)

# FFA Backbone (large block spanning 3 group processing stages)
draw_3d_cube(ax, (4.5, 0.5, 2), (2.2, 1.5, 1.2), C_BACKBONE, edge_color=E_BACKBONE, alpha=0.8)
ax.text(5.6, 1.25, 3.4, "FFA Backbone\n(3 Groups × 3 PA)", fontsize=10,
        ha='center', va='center', weight='bold')

# Reconstruct
draw_3d_cube(ax, (7.4, 0.5, 2), (0.7, 1.5, 1.2), C_RGB, edge_color=E_RGB)
ax.text(7.75, 1.25, 3.4, "Recon\nstruct", fontsize=9, ha='center', va='center', weight='bold')

# ⊕ Add node
ax.scatter([8.8], [1.25], [2.6], color='white', edgecolors=E_RGB, s=120, linewidths=2, zorder=10)
ax.text(8.8, 1.25, 2.6, "+", fontsize=14, ha='center', va='center', weight='bold')

# Restored Output
draw_3d_cube(ax, (9.5, 0.5, 2), (0.6, 1.5, 1.2), C_OUTPUT, edge_color=E_OUTPUT)
ax.text(9.8, 1.25, 3.4, r"$\hat I$", fontsize=11, ha='center', va='center', weight='bold')

# ── Horizontal arrows along RGB stream ──
# Hazy → Encoder
ax.plot([0.8, 1.5], [1.25, 1.25], [2.6, 2.6], color=E_RGB, lw=1.8)
ax.quiver(1.35, 1.25, 2.6, 0.15, 0, 0, color=E_RGB, arrow_length_ratio=0.3, lw=1.5)
# Encoder → DG-CCA
ax.plot([2.1, 2.8], [1.25, 1.25], [2.6, 2.6], color=E_RGB, lw=1.8)
ax.quiver(2.65, 1.25, 2.6, 0.15, 0, 0, color=E_RGB, arrow_length_ratio=0.3, lw=1.5)
# DG-CCA → FFA
ax.plot([3.7, 4.5], [1.25, 1.25], [2.6, 2.6], color=E_DGCCA, lw=1.8)
ax.quiver(4.35, 1.25, 2.6, 0.15, 0, 0, color=E_DGCCA, arrow_length_ratio=0.3, lw=1.5)
# FFA → Reconstruct
ax.plot([6.7, 7.4], [1.25, 1.25], [2.6, 2.6], color=E_BACKBONE, lw=1.8)
ax.quiver(7.25, 1.25, 2.6, 0.15, 0, 0, color=E_BACKBONE, arrow_length_ratio=0.3, lw=1.5)
# Reconstruct → ⊕
ax.plot([8.1, 8.7], [1.25, 1.25], [2.6, 2.6], color=E_RGB, lw=1.8)
ax.quiver(8.55, 1.25, 2.6, 0.15, 0, 0, color=E_RGB, arrow_length_ratio=0.3, lw=1.5)
# ⊕ → Output
ax.plot([8.9, 9.5], [1.25, 1.25], [2.6, 2.6], color=E_RGB, lw=1.8)
ax.quiver(9.35, 1.25, 2.6, 0.15, 0, 0, color=E_RGB, arrow_length_ratio=0.3, lw=1.5)

# ── Global Residual (above) ──
dashed_line_3d(ax, (0.2, 0.5+1.5+0.3, 2+0.6), (0.2, 0.5+1.5+0.3, 2+0.6), color=E_RGB, lw=1.0)
ax.plot([0.2, 8.8], [2.8, 2.8], [2.6, 2.6], color=E_RGB, lw=1.2, ls=':', dashes=(3, 3))
ax.plot([8.8, 8.8], [2.8, 1.35], [2.6, 2.6], color=E_RGB, lw=1.2, ls=':', dashes=(3, 3))
ax.text(4.5, 3.0, 2.6, "Global Residual $I+R(I)$", fontsize=9, ha='center', color=E_RGB, weight='bold')

# ═══════ LOWER STREAM: Depth Branch (z=-2 plane) ═══════
# Frozen DAV2
draw_3d_cube(ax, (0, 0.5, -2), (1.6, 1.5, 1.2), C_FROZEN, edge_color=E_FROZEN, alpha=0.85)
ax.text(0.8, 1.25, -1.4, "Depth Anything V2\n(Frozen · 24.8M)", fontsize=9,
        ha='center', va='center', weight='bold')

# Depth Prior
draw_3d_cube(ax, (2.2, 0.5, -2), (1.2, 1.5, 1.2), C_DEPTH, edge_color=E_DEPTH)
ax.text(2.8, 1.25, -1.4, "Depth\nPrior", fontsize=9, ha='center', va='center', weight='bold')

# Depth baseline line (horizontal through depth stream)
ax.plot([1.6, 4.0], [1.25, 1.25], [-1.4, -1.4], color=E_DEPTH, lw=2.0)

# ── Vertical connection: Hazy → DAV2 (downward) ──
ax.plot([0.5, 0.5], [1.25, 1.25], [2.6, -1.4], color=E_DEPTH, lw=1.5, ls='--', dashes=(5, 4))
ax.text(0.2, 1.25, 0.6, "share input", fontsize=8, ha='center', color=E_DEPTH, rotation=90)

# ── F_d vertical injection: Depth Prior → DG-CCA ──
ax.plot([2.8, 3.25], [1.25, 1.25], [-1.4, 2.2], color=E_DGCCA, lw=2.0, ls=':')
ax.scatter([2.8], [1.25], [-1.4], color=E_DGCCA, s=30)
ax.text(3.0, 1.25, 0.4, r"$F_d$", fontsize=10, ha='center', color=E_DGCCA, weight='bold')

# ═══════ BOTTOM PANELS (drawn on a flat 2D overlay) ═══════
# We use a separate subplot approach: add text at the bottom z=-3 plane
# DG-CCA Detail
ax.text(0.3, 1.25, -3.5,
        "(a) DG-CCA: $F_{rgb},F_d$ → GAP → $z_{rgb},z_d$ → Concat → MLP($W_1$,ReLU,$W_2$,σ) → $A$ → $F_{out}=A⊙F_{rgb}$",
        fontsize=8, ha='left', va='center', color='#333333')
ax.text(0.3, 1.25, -4.0,
        "(b) Training: Stage 1 (Epoch 1–100) $L_{rec}$, Stage 2 (101–120) backbone frozen, $L_{total}=L_{rec}+0.01L_{TV}$",
        fontsize=8, ha='left', va='center', color='#333333')

# ═══════ View & Save ═══════
ax.view_init(elev=22, azim=-55)
ax.set_xlim(-1, 11); ax.set_ylim(0, 3.5); ax.set_zlim(-4.5, 4.5)

plt.title("Figure 2. Overview of Geometry-Aware Physical Dehazing Framework (3D Schema)",
          fontsize=14, pad=25, weight='bold')

for fmt, dpi in [("pdf", 300), ("png", 300)]:
    fp = OUT / f"fig2_3d.{fmt}"
    fig.savefig(fp, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {fp}")

plt.close(fig)
print("Done.")