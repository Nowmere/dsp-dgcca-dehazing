"""Draw a concise, code-aligned architecture figure.

The script generates standalone PDF and PNG assets only. It does not edit
the paper source or replace any existing figure reference.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper" / "CVPR_2026_Submission_Template" / "figures"
OUTPUT_STEM = OUTPUT_DIR / "architecture_v2"

W, H = 14.0, 5.6

COLORS = {
    "ink": "#17212B",
    "muted": "#596674",
    "rgb": "#DCEBFA",
    "rgb_edge": "#2F6FA3",
    "depth": "#DFF1E5",
    "depth_edge": "#3D7E57",
    "fusion": "#FCE8CE",
    "fusion_edge": "#B86B18",
    "backbone": "#E8E2F4",
    "backbone_edge": "#70569A",
    "loss": "#F8E2E3",
    "loss_edge": "#A84D51",
    "panel": "#F7F9FB",
    "panel_edge": "#C9D0D7",
    "white": "#FFFFFF",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "savefig.facecolor": "white",
        }
    )


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    detail=None,
    *,
    fill="rgb",
    edge="rgb_edge",
    dashed=False,
    title_size=9.4,
    detail_size=7.1,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.07",
        facecolor=COLORS[fill],
        edgecolor=COLORS[edge],
        linewidth=1.25,
        linestyle=(0, (4, 2.5)) if dashed else "solid",
        zorder=4,
    )
    ax.add_patch(patch)
    if detail is None:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=COLORS["ink"],
            zorder=5,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2 + 0.12,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=COLORS["ink"],
            zorder=5,
        )
        ax.text(
            x + w / 2,
            y + h / 2 - 0.17,
            detail,
            ha="center",
            va="center",
            fontsize=detail_size,
            color=COLORS["muted"],
            zorder=5,
        )
    return patch


def arrow(ax, start, end, *, color="rgb_edge", width=1.35, dashed=False, zorder=3):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=width,
        linestyle=(0, (4, 2.5)) if dashed else "solid",
        color=COLORS[color],
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def poly_arrow(ax, points, *, color="rgb_edge", width=1.35, dashed=False, zorder=3):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=COLORS[color],
            linewidth=width,
            linestyle=(0, (4, 2.5)) if dashed else "solid",
            solid_capstyle="round",
            zorder=zorder,
        )
    arrow(
        ax,
        points[-2],
        points[-1],
        color=color,
        width=width,
        dashed=dashed,
        zorder=zorder,
    )


def panel(ax, x, y, w, h, label):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=COLORS["panel"],
        edgecolor=COLORS["panel_edge"],
        linewidth=1.0,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.18,
        y + h - 0.25,
        label,
        ha="left",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=6,
    )
    return patch


def plus_node(ax, x, y):
    node = Circle(
        (x, y),
        0.16,
        facecolor=COLORS["white"],
        edgecolor=COLORS["rgb_edge"],
        linewidth=1.2,
        zorder=5,
    )
    ax.add_patch(node)
    ax.text(
        x,
        y,
        "+",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=6,
    )


def draw_framework(ax):
    ax.text(
        0.45,
        5.25,
        "Geometry-aware dehazing network",
        ha="left",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        13.55,
        5.25,
        "Depth guides one early fusion; RGB reconstruction remains lightweight",
        ha="right",
        va="center",
        fontsize=8.2,
        color=COLORS["muted"],
    )

    # Main reconstruction path.
    box(ax, 0.45, 3.45, 1.20, 0.78, "Hazy image", r"$I$", fill="rgb", edge="rgb_edge")
    box(ax, 2.25, 3.45, 1.40, 0.78, "RGB encoder", r"$3\!\to\!64$", fill="rgb", edge="rgb_edge")
    box(ax, 5.15, 3.35, 1.42, 0.98, "DG-CCA", "channel fusion", fill="fusion", edge="fusion_edge", title_size=10.2)
    box(ax, 7.35, 3.35, 1.90, 0.98, "FFA backbone", "3 groups x 3 blocks", fill="backbone", edge="backbone_edge")
    box(ax, 10.05, 3.45, 1.35, 0.78, "Reconstruct", r"$3\!\to\!3$", fill="rgb", edge="rgb_edge")
    plus_node(ax, 12.10, 3.84)
    box(ax, 12.62, 3.45, 0.88, 0.78, "Clear", r"$\hat I$", fill="rgb", edge="rgb_edge")

    arrow(ax, (1.65, 3.84), (2.25, 3.84))
    arrow(ax, (3.65, 3.84), (5.15, 3.84))
    arrow(ax, (6.57, 3.84), (7.35, 3.84), color="fusion_edge")
    arrow(ax, (9.25, 3.84), (10.05, 3.84), color="backbone_edge")
    arrow(ax, (11.40, 3.84), (11.94, 3.84))
    arrow(ax, (12.26, 3.84), (12.62, 3.84))

    # Frozen depth guidance branch.
    box(ax, 2.25, 2.05, 1.40, 0.76, "Depth Anything V2", "frozen", fill="depth", edge="depth_edge", dashed=True, title_size=8.8)
    box(ax, 4.05, 2.05, 1.42, 0.76, "Depth prior", "normalize + gate", fill="depth", edge="depth_edge", title_size=9.0)
    box(ax, 5.87, 2.05, 1.38, 0.76, "Depth encoder", r"$1\!\to\!64$", fill="depth", edge="depth_edge", title_size=9.0)

    poly_arrow(ax, [(1.05, 3.45), (1.05, 2.43), (2.25, 2.43)], color="depth_edge", width=1.2)
    arrow(ax, (3.65, 2.43), (4.05, 2.43), color="depth_edge", width=1.2)
    arrow(ax, (5.47, 2.43), (5.87, 2.43), color="depth_edge", width=1.2)
    poly_arrow(ax, [(7.25, 2.43), (7.48, 2.43), (7.48, 3.03), (6.15, 3.35)], color="depth_edge", width=1.2)

    # Global image residual.
    poly_arrow(
        ax,
        [(1.05, 4.23), (1.05, 4.66), (12.10, 4.66), (12.10, 4.00)],
        color="rgb_edge",
        width=1.05,
        zorder=2,
    )
    ax.text(
        9.55,
        4.72,
        "global residual",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color=COLORS["rgb_edge"],
    )


def draw_compact_details(ax):
    # (a) DG-CCA detail — SE-style channel gating (matching Eq. 1-4 in the paper)
    panel(ax, 0.45, 0.28, 7.60, 1.35, "(a) DG-CCA (Squeeze-and-Excitation style)")

    # Row 1: F_rgb → GAP → z_rgb   ...   z_d ← GAP ← F_d
    ax.text(0.75, 0.88, r"$F_{rgb}$", ha="center", va="center", fontsize=8.8, fontweight="bold", color=COLORS["rgb_edge"])
    ax.text(1.80, 0.88, r"GAP", ha="center", va="center", fontsize=7.8, fontweight="bold", color=COLORS["muted"])
    ax.text(2.85, 0.88, r"$z_{rgb}$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLORS["ink"])

    ax.text(6.70, 0.88, r"$F_d$", ha="center", va="center", fontsize=8.8, fontweight="bold", color=COLORS["depth_edge"])
    ax.text(5.75, 0.88, r"GAP", ha="center", va="center", fontsize=7.8, fontweight="bold", color=COLORS["muted"])
    ax.text(4.70, 0.88, r"$z_d$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLORS["ink"])

    # Arrows: F_rgb→GAP→z_rgb, F_d→GAP→z_d
    arrow(ax, (1.05, 0.88), (1.55, 0.88), width=0.9, color="muted")
    arrow(ax, (2.05, 0.88), (2.60, 0.88), width=0.9, color="muted")
    arrow(ax, (5.15, 0.88), (5.50, 0.88), width=0.9, color="muted")
    arrow(ax, (6.00, 0.88), (6.45, 0.88), width=0.9, color="muted")

    # Row 2: Concat → MLP → σ → A
    ax.text(3.77, 0.50, r"Concat $[z_{rgb};\,z_d]$", ha="center", va="center", fontsize=8.2, fontweight="bold", color=COLORS["fusion_edge"])
    ax.text(3.77, 0.18, r"FC $\to$ ReLU $\to$ FC $\to$ $\sigma$", ha="center", va="center", fontsize=8.0, color=COLORS["ink"])
    arrow(ax, (3.77, 0.65), (3.77, 0.55), width=0.9, color="muted")

    # Arrows: z_rgb / z_d down to Concat
    arrow(ax, (2.85, 0.78), (2.85, 0.65), width=0.7, color="muted")
    arrow(ax, (4.70, 0.78), (4.70, 0.65), width=0.7, color="muted")

    # Output
    ax.text(5.85, 0.18, r"$A\in\mathbb{R}^{C}$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLORS["fusion_edge"])
    ax.text(7.20, 0.18, r"$F_{out}=A\!\odot\!F_{rgb}$", ha="center", va="center", fontsize=8.8, fontweight="bold", color=COLORS["ink"])
    arrow(ax, (4.72, 0.18), (5.40, 0.18), width=0.9, color="fusion_edge")
    arrow(ax, (6.20, 0.18), (6.80, 0.18), width=0.9, color="muted")

    # (b) Training strategy
    panel(ax, 8.35, 0.28, 5.20, 1.35, "(b) Training & Sky TV Loss")
    ax.text(10.95, 0.95, "Stage 1 (Epoch 1..100):", ha="center", va="center", fontsize=9.0, fontweight="bold", color=COLORS["ink"])
    ax.text(10.95, 0.72, r"$\mathcal{L}_{rec}=\|\hat I\!-\!J\|_1+0.05\,\mathcal{L}_{edge}$", ha="center", va="center", fontsize=9.0, color=COLORS["loss_edge"])
    ax.text(10.95, 0.50, "Stage 2 (Epoch 101..120): backbone frozen", ha="center", va="center", fontsize=8.0, color=COLORS["muted"])
    ax.text(10.95, 0.32, r"$\mathcal{L}_{total}=\mathcal{L}_{rec}+0.01\,\mathcal{L}_{TV}$", ha="center", va="center", fontsize=9.0, color=COLORS["loss_edge"])
    ax.text(10.95, 0.12, r"$M=\mathbf{1}[D\!\geq\!0.1]$; $\mathcal{L}_{TV}$ on sky only", ha="center", va="center", fontsize=7.5, color=COLORS["muted"])


def main() -> None:
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.0, 5.6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_framework(ax)
    draw_compact_details(ax)

    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    png_path = OUTPUT_STEM.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=360, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()