"""Figure 2: Professional architecture diagram for CVPR paper.
Clean, academic style with clear visual hierarchy.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.patheffects import withStroke
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "CVPR_2026_Submission_Template" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Color scheme - professional academic colors
COLORS = {
    'rgb': '#4A90D9',      # Blue for RGB features
    'rgb_fill': '#E8F1FC',
    'depth': '#5CB85C',    # Green for depth
    'depth_fill': '#E8F5E8',
    'dgcca': '#E07040',    # Orange for DG-CCA
    'dgcca_fill': '#FCEBE4',
    'backbone': '#7B68EE', # Purple for backbone
    'backbone_fill': '#EDE7F6',
    'text': '#1a1a1a',
    'gray': '#666666',
    'panel': '#F8F9FC',
    'panel_border': '#CCCCCC',
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
})


def draw_box(ax, x, y, w, h, text, subtext=None, color='rgb', fill=True, lw=1.2, radius=0.08, fontsize=9):
    """Draw a rounded rectangle box with text."""
    c = COLORS[color]
    fc = COLORS.get(f'{color}_fill', 'white') if fill else 'white'
    
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad={radius},rounding_size={radius}",
                         facecolor=fc, edgecolor=c, linewidth=lw)
    ax.add_patch(box)
    
    # Draw text
    if subtext:
        ax.text(x, y + 0.08, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=COLORS['text'])
        ax.text(x, y - 0.14, subtext, ha='center', va='center', fontsize=fontsize-2,
                color=COLORS['gray'])
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=COLORS['text'])
    
    return box


def draw_arrow(ax, x0, y0, x1, y1, color='rgb', lw=1.0, style='-|>'):
    """Draw an arrow."""
    arrow = FancyArrowPatch((x0, y0), (x1, y1),
                           arrowstyle=style,
                           mutation_scale=12,
                           linewidth=lw,
                           color=COLORS[color],
                           connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)


def draw_dashed_arrow(ax, x0, y0, x1, y1, color='gray', lw=0.8):
    """Draw a dashed arrow."""
    arrow = FancyArrowPatch((x0, y0), (x1, y1),
                           arrowstyle='-|>',
                           mutation_scale=10,
                           linewidth=lw,
                           color=COLORS[color],
                           linestyle='dashed',
                           connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)


def draw_circle_op(ax, x, y, symbol, color='gray', size=0.18):
    """Draw a circular operation (like addition or concatenation)."""
    circle = Circle((x, y), size, facecolor='white', edgecolor=COLORS[color], linewidth=1.2)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=11,
            fontweight='bold', color=COLORS['text'])


def draw_fig2():
    """Main drawing function."""
    # Figure size: wide format for two-column paper
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # ============ TITLE ============
    ax.text(0.5, 7.6, "Figure 2. Overall architecture of the proposed geometry-aware physical dehazing framework.",
            ha='left', va='center', fontsize=11, fontweight='bold', color=COLORS['text'])
    
    # ============ MAIN ARCHITECTURE (Top section) ============
    main_y = 5.8  # Main flow y position
    
    # 1. Hazy Input
    draw_box(ax, 1.2, main_y, 1.4, 0.7, r"Hazy Input $I$", None, 'rgb', lw=1.5)
    
    # Arrow
    draw_arrow(ax, 1.9, main_y, 2.4, main_y, 'text')
    
    # 2. RGB Encoder
    draw_box(ax, 3.1, main_y, 1.5, 0.7, r"RGB Encoder", r"$3 \rightarrow 64$", 'rgb')
    
    # Arrow with depth injection
    draw_arrow(ax, 3.85, main_y, 4.3, main_y, 'text')
    
    # 3. DG-CCA modules (3 stacked boxes)
    dgcca_x = 5.3
    for i in range(3):
        y_pos = main_y + 0.55 - i * 0.7
        draw_box(ax, dgcca_x, y_pos, 1.4, 0.55, f"DG-CCA #{i+1}", None, 'dgcca', lw=1.8)
        if i < 2:
            # Vertical line connecting DG-CCA
            ax.plot([dgcca_x, dgcca_x], [y_pos - 0.28, y_pos - 0.42], 
                   color=COLORS['dgcca'], linewidth=1.2)
    
    # Arrow out from DG-CCA
    draw_arrow(ax, 6.0, main_y, 6.5, main_y, 'text')
    
    # 4. FFA Backbone
    draw_box(ax, 7.5, main_y, 1.8, 0.7, r"FFA Backbone", r"3 Groups $\times$ 3 PA", 'backbone')
    
    # Arrow
    draw_arrow(ax, 8.4, main_y, 8.9, main_y, 'text')
    
    # 5. Reconstruct
    draw_box(ax, 9.7, main_y, 1.4, 0.7, r"Reconstruct", r"$3\times3$ Conv", 'rgb')
    
    # Arrow
    draw_arrow(ax, 10.4, main_y, 10.9, main_y, 'text')
    
    # 6. Restored Output
    draw_box(ax, 11.7, main_y, 1.4, 0.7, r"Restored $\hat{I}$", None, 'rgb', lw=1.5)
    
    # ============ DEPTH BRANCH ============
    depth_y = 4.0
    
    # DAV2 frozen block
    draw_box(ax, 3.1, depth_y, 1.8, 0.6, r"Depth Anything V2", r"Frozen 24.8M", 'depth', lw=1.5)
    
    # Lock icon (using text)
    ax.text(4.15, depth_y + 0.25, "🔒", fontsize=8, ha='left', va='center')
    
    # Arrow
    draw_arrow(ax, 4.0, depth_y, 4.6, depth_y, 'depth')
    
    # Depth Prior
    draw_box(ax, 5.4, depth_y, 1.4, 0.6, r"Depth Prior", r"Min-Max + Adapter", 'depth')
    
    # Vertical arrow from depth to DG-CCA
    draw_dashed_arrow(ax, 5.4, depth_y + 0.3, 5.4, main_y - 0.35, 'depth', lw=1.0)
    ax.text(5.55, (depth_y + main_y) / 2, r"$F_d$", fontsize=8, color=COLORS['depth'], 
            fontweight='bold', ha='left', va='center')
    
    # ============ GLOBAL RESIDUAL ============
    # Draw residual connection
    residual_x_start = 1.2
    residual_y_top = 6.35
    residual_x_end = 10.4
    residual_y_bottom = main_y + 0.35
    
    # Path
    path_x = [residual_x_start - 0.1, residual_x_start - 0.1, residual_x_end, residual_x_end]
    path_y = [residual_y_top, main_y + 0.35, main_y + 0.35, residual_y_top]
    
    ax.plot([residual_x_start - 0.1, residual_x_start - 0.1],
            [main_y + 0.35, residual_y_top],
            color=COLORS['text'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax.plot([residual_x_start - 0.1, residual_x_end],
            [residual_y_top, residual_y_top],
            color=COLORS['text'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax.plot([residual_x_end, residual_x_end],
            [residual_y_top, main_y + 0.35],
            color=COLORS['text'], linewidth=0.8, linestyle='--', alpha=0.6)
    
    ax.text(5.7, residual_y_top + 0.08, "Global Residual", fontsize=7, 
            color=COLORS['gray'], ha='center', va='bottom', style='italic')
    
    # ============ DG-CCA DETAIL PANEL ============
    panel_x, panel_y = 0.3, 0.4
    panel_w, panel_h = 6.0, 3.0
    
    # Panel background
    panel = FancyBboxPatch((panel_x, panel_y), panel_w, panel_h,
                           boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor=COLORS['panel'], edgecolor=COLORS['panel_border'], 
                           linewidth=0.8)
    ax.add_patch(panel)
    
    # Panel title
    ax.text(panel_x + 0.15, panel_y + panel_h - 0.2, 
            "(a) DG-CCA Module Detail", fontsize=9, fontweight='bold',
            color=COLORS['text'], ha='left', va='top')
    
    # DG-CCA detail flow
    detail_y = panel_y + 1.8
    detail_spacing = 1.1
    
    # F_rgb and F_d
    draw_box(ax, panel_x + 0.8, detail_y + 0.8, 1.0, 0.5, r"$F_{rgb}$", r"$C \times H \times W$", 'rgb', fontsize=8)
    draw_box(ax, panel_x + 2.5, detail_y + 0.8, 1.0, 0.5, r"$F_d$", r"$C \times H \times W$", 'depth', fontsize=8)
    
    # GAP labels
    ax.text(panel_x + 1.55, detail_y + 1.2, "GAP", fontsize=7, color=COLORS['gray'], 
            ha='center', va='bottom', fontweight='bold')
    ax.text(panel_x + 2.2, detail_y + 1.2, "GAP", fontsize=7, color=COLORS['gray'], 
            ha='center', va='bottom', fontweight='bold')
    
    # Arrows down from F to z
    draw_arrow(ax, panel_x + 0.8, detail_y + 0.55, panel_x + 0.8, detail_y + 0.35, 'gray', lw=0.7)
    draw_arrow(ax, panel_x + 2.5, detail_y + 0.55, panel_x + 2.5, detail_y + 0.35, 'depth', lw=0.7)
    
    # z_rgb and z_d
    draw_box(ax, panel_x + 0.8, detail_y + 0.15, 0.9, 0.4, r"$z_{rgb}$", r"$\mathbb{R}^C$", 'rgb', fontsize=7)
    draw_box(ax, panel_x + 2.5, detail_y + 0.15, 0.9, 0.4, r"$z_d$", r"$\mathbb{R}^C$", 'depth', fontsize=7)
    
    # Arrows down to concat
    draw_arrow(ax, panel_x + 0.8, detail_y - 0.05, panel_x + 0.8, detail_y - 0.25, 'gray', lw=0.7)
    draw_arrow(ax, panel_x + 2.5, detail_y - 0.05, panel_x + 2.5, detail_y - 0.25, 'depth', lw=0.7)
    
    # Concat circle
    draw_circle_op(ax, panel_x + 3.5, detail_y - 0.1, r"$\odot$", 'dgcca', size=0.22)
    ax.text(panel_x + 3.5, detail_y - 0.45, "Concat", fontsize=7, color=COLORS['gray'], 
            ha='center', va='top')
    
    # Arrow to MLP
    draw_arrow(ax, panel_x + 3.72, detail_y - 0.1, panel_x + 4.0, detail_y - 0.1, 'dgcca', lw=0.8)
    
    # MLP box
    draw_box(ax, panel_x + 4.5, detail_y - 0.1, 0.9, 0.5, "MLP", r"$2C \rightarrow C$", 'dgcca', fontsize=7)
    
    # Arrow to sigmoid
    draw_arrow(ax, panel_x + 4.95, detail_y - 0.1, panel_x + 5.2, detail_y - 0.1, 'dgcca', lw=0.8)
    
    # Sigmoid
    draw_box(ax, panel_x + 5.5, detail_y - 0.1, 0.55, 0.45, r"$\sigma$", "Sigmoid", 'dgcca', fontsize=8)
    
    # ============ TRAINING STRATEGY PANEL ============
    train_x = 6.5
    train_panel_w = 7.0
    
    # Panel background
    train_panel = FancyBboxPatch((train_x, panel_y), train_panel_w, panel_h,
                                  boxstyle="round,pad=0.02,rounding_size=0.1",
                                  facecolor=COLORS['panel'], edgecolor=COLORS['panel_border'], 
                                  linewidth=0.8)
    ax.add_patch(train_panel)
    
    # Panel title
    ax.text(train_x + 0.15, panel_y + panel_h - 0.2, 
            "(b) Two-Stage Training Curriculum", fontsize=9, fontweight='bold',
            color=COLORS['text'], ha='left', va='top')
    
    # Stage 1
    stage1_y = panel_y + 2.3
    draw_box(ax, train_x + 1.2, stage1_y, 2.0, 0.6, "Stage 1", r"100 epochs", 'backbone', fontsize=8)
    
    stage1_content_y = stage1_y - 0.6
    ax.text(train_x + 1.2, stage1_content_y, "Full network training", fontsize=7.5, 
            ha='center', va='top', color=COLORS['gray'])
    ax.text(train_x + 1.2, stage1_content_y - 0.25, r"$\mathcal{L} = \mathcal{L}_1 + 0.05 \cdot \mathcal{L}_{edge}$", 
            fontsize=7, ha='center', va='top', color=COLORS['text'])
    
    # Arrow between stages
    draw_arrow(ax, train_x + 2.2, stage1_y - 0.3, train_x + 2.2, stage1_y - 0.7, 'text', lw=1.0)
    ax.text(train_x + 2.35, stage1_y - 0.5, "→", fontsize=10, ha='left', va='center', color=COLORS['text'])
    
    # Stage 2
    stage2_y = stage1_y - 1.1
    draw_box(ax, train_x + 1.2, stage2_y, 2.0, 0.6, "Stage 2", r"20 epochs", 'dgcca', fontsize=8)
    
    stage2_content_y = stage2_y - 0.6
    ax.text(train_x + 1.2, stage2_content_y, "Regional decoupled fine-tuning", fontsize=7.5, 
            ha='center', va='top', color=COLORS['gray'])
    ax.text(train_x + 1.2, stage2_content_y - 0.25, r"Backbone frozen, $\lambda = 0.01$", 
            fontsize=7, ha='center', va='top', color=COLORS['text'])
    
    # Sky TV Loss highlight box
    tv_box_x = train_x + 4.0
    tv_box = FancyBboxPatch((tv_box_x, stage2_y - 0.5), 2.3, 1.3,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            facecolor='#FFF8E1', edgecolor='#FFB300', linewidth=1.0)
    ax.add_patch(tv_box)
    
    ax.text(tv_box_x + 1.15, stage2_y + 0.55, "Sky TV Loss", fontsize=8, 
            ha='center', va='top', fontweight='bold', color='#E65100')
    ax.text(tv_box_x + 1.15, stage2_y + 0.25, r"$\mathcal{L}_{TV} = \sum (1-\mathcal{M}) \cdot |\nabla \hat{I}|$", 
            fontsize=6.5, ha='center', va='top', color=COLORS['text'])
    ax.text(tv_box_x + 1.15, stage2_y - 0.05, r"$\tau = 0.1$ (mask sky region)", 
            fontsize=6.5, ha='center', va='top', color=COLORS['gray'])
    ax.text(tv_box_x + 1.15, stage2_y - 0.3, "Targeted Total Variation", 
            fontsize=6, ha='center', va='top', color=COLORS['gray'], style='italic')
    
    # ============ LEGEND ============
    legend_y = 2.5
    legend_x = 0.5
    
    # Legend items
    items = [
        ('rgb', 'RGB Feature'),
        ('depth', 'Depth Feature'),
        ('dgcca', 'DG-CCA Module'),
        ('backbone', 'Network Backbone'),
    ]
    
    for i, (color, label) in enumerate(items):
        x = legend_x + i * 1.8
        # Mini box
        rect = FancyBboxPatch((x, legend_y), 0.3, 0.25,
                              boxstyle="round,pad=0.02,rounding_size=0.05",
                              facecolor=COLORS[f'{color}_fill'], 
                              edgecolor=COLORS[color], linewidth=1.0)
        ax.add_patch(rect)
        # Label
        ax.text(x + 0.4, legend_y + 0.12, label, fontsize=7, 
                ha='left', va='center', color=COLORS['text'])
    
    # Arrow legend
    ax.annotate('', xy=(legend_x + 7.2, legend_y + 0.12), 
                xytext=(legend_x + 7.0, legend_y + 0.12),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['text'], lw=0.8))
    ax.text(legend_x + 7.35, legend_y + 0.12, "Data flow", fontsize=7, 
            ha='left', va='center', color=COLORS['text'])
    
    ax.annotate('', xy=(legend_x + 9.3, legend_y + 0.12), 
                xytext=(legend_x + 9.1, legend_y + 0.12),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['depth'], lw=0.8,
                               linestyle='dashed'))
    ax.text(legend_x + 9.45, legend_y + 0.12, "Depth injection", fontsize=7, 
            ha='left', va='center', color=COLORS['text'])
    
    # ============ SAVE ============
    output_path_png = OUT / "fig2_professional.png"
    output_path_pdf = OUT / "fig2_professional.pdf"
    
    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    fig.savefig(output_path_pdf, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_path_png}")
    print(f"Saved: {output_path_pdf}")
    
    plt.close(fig)


if __name__ == "__main__":
    draw_fig2()