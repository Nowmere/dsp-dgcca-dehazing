"""
Visualize DAV2 depth maps and sky masks for the paper.
Creates side-by-side comparison of depth, mask, and original image.
"""

import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_depth(depth_path):
    """Load depth map from npy file."""
    depth = np.load(str(depth_path))
    return depth

def create_sky_mask(depth, tau=0.1):
    """Create sky mask from depth map."""
    mask = (depth < tau).astype(np.uint8)
    # Morphological opening to remove small fragments
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def visualize_sample(img_path, depth_path, save_path, title=""):
    """Create visualization for a single sample."""
    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load depth
    depth = load_depth(depth_path)
    
    # Create mask
    mask = create_sky_mask(depth, tau=0.1)
    
    # Create figure
    fig = plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.05)
    
    # Original image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img)
    ax1.set_title('Hazy Input', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # DAV2 Depth map
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(depth, cmap='plasma')
    ax2.set_title('DAV2 Depth Map', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Sky mask overlay
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(img)
    mask_colored = np.zeros((*mask.shape, 4))
    mask_colored[mask > 0] = [1, 0, 0, 0.4]  # Red overlay
    ax3.imshow(mask_colored)
    ax3.set_title(f'Sky Mask (τ=0.1)\nCoverage: {mask.sum()/mask.size*100:.1f}%', 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Depth gradient (showing infinity noise)
    ax4 = fig.add_subplot(gs[3])
    grad_x = np.abs(np.gradient(depth, axis=1))
    grad_y = np.abs(np.gradient(depth, axis=0))
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    im4 = ax4.imshow(grad_mag, cmap='hot')
    ax4.set_title('Depth Gradient Magnitude', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved: {save_path}")

def main():
    # Dataset samples (with correct file names)
    datasets = {
        'NH-HAZE': {
            'hazy_dir': './data/NH-HAZE/hazy',
            'depth_dir': './data/NH-HAZE/depth_dav2',
            'samples': ['nh15_hazy.png', 'nh30_hazy.png', 'nh45_hazy.png'],
            'save_dir': './paper/CVPR_2026_Submission_Template/figures/depth_analysis_nhhaze'
        },
        'O-HAZE': {
            'hazy_dir': './data/O-HAZE/hazy',
            'depth_dir': './data/O-HAZE/depth_dav2',
            'samples': ['10_outdoor.jpg', '25_outdoor.jpg', '40_outdoor.jpg'],
            'save_dir': './paper/CVPR_2026_Submission_Template/figures/depth_analysis_ohaze'
        },
        'I-HAZE': {
            'hazy_dir': './data/I-HAZE/hazy',
            'depth_dir': './data/I-HAZE/depth_dav2',
            'samples': ['ih10_hazy.png', 'ih20_hazy.png', 'ih30_hazy.png'],
            'save_dir': './paper/CVPR_2026_Submission_Template/figures/depth_analysis_ihaze'
        },
    }
    
    # Create directories
    for cfg in datasets.values():
        os.makedirs(cfg['save_dir'], exist_ok=True)
    
    # Generate visualizations
    for name, cfg in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing {name}")
        print('='*60)
        
        for i, sample in enumerate(cfg['samples']):
            img_path = os.path.join(cfg['hazy_dir'], sample)
            depth_name = os.path.splitext(sample)[0] + '.npy'
            depth_path = os.path.join(cfg['depth_dir'], depth_name)
            
            if os.path.exists(img_path) and os.path.exists(depth_path):
                save_path = os.path.join(cfg['save_dir'], f"{os.path.splitext(sample)[0]}_analysis.png")
                visualize_sample(img_path, depth_path, save_path, title=f"{name} Sample {i+1}")
            else:
                print(f"  Skipping {sample}: files not found")
    
    # Create combined overview figure
    print(f"\n{'='*60}")
    print("Creating combined overview figure...")
    print('='*60)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    for row, (name, cfg) in enumerate(datasets.items()):
        sample = cfg['samples'][0]
        img_path = os.path.join(cfg['hazy_dir'], sample)
        depth_name = os.path.splitext(sample)[0] + '.npy'
        depth_path = os.path.join(cfg['depth_dir'], depth_name)
        
        if os.path.exists(img_path) and os.path.exists(depth_path):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            depth = load_depth(depth_path)
            mask = create_sky_mask(depth, tau=0.1)
            
            # Original
            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f'{name}: Hazy Input', fontsize=11, fontweight='bold')
            axes[row, 0].axis('off')
            
            # Depth
            im = axes[row, 1].imshow(depth, cmap='plasma')
            axes[row, 1].set_title('DAV2 Depth', fontsize=11, fontweight='bold')
            axes[row, 1].axis('off')
            
            # Mask overlay
            axes[row, 2].imshow(img)
            mask_overlay = np.zeros((*mask.shape, 4))
            mask_overlay[mask > 0] = [1, 0, 0, 0.4]
            axes[row, 2].imshow(mask_overlay)
            axes[row, 2].set_title(f'Sky Mask (τ=0.1)\nCoverage: {mask.sum()/mask.size*100:.1f}%', 
                                   fontsize=11, fontweight='bold')
            axes[row, 2].axis('off')
            
            # Gradient
            grad_x = np.abs(np.gradient(depth, axis=1))
            grad_y = np.abs(np.gradient(depth, axis=0))
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            axes[row, 3].imshow(grad_mag, cmap='hot')
            axes[row, 3].set_title('Depth Gradient', fontsize=11, fontweight='bold')
            axes[row, 3].axis('off')
    
    # Add column labels
    axes[0, 0].set_xlabel('(a) Input', fontsize=10, fontweight='bold')
    axes[0, 1].set_xlabel('(b) Depth', fontsize=10, fontweight='bold')
    axes[0, 2].set_xlabel('(c) Mask', fontsize=10, fontweight='bold')
    axes[0, 3].set_xlabel('(d) Gradient', fontsize=10, fontweight='bold')
    
    plt.suptitle('DAV2 Depth Analysis Across Datasets: Visualizing the Infinity Depth Noise Problem', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    overview_path = './paper/CVPR_2026_Submission_Template/figures/depth_analysis_overview.png'
    plt.savefig(overview_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved: {overview_path}")
    print("\n" + "="*60)
    print("✅ Visualization complete!")
    print("="*60)

if __name__ == '__main__':
    main()