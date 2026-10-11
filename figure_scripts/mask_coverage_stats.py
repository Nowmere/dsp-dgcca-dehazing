"""
Analyze sky mask coverage statistics across datasets.
Computes: mask coverage ratio, false positive rate (non-sky pixels included), 
and overlap with actual sky regions.
"""

import numpy as np
import os
from pathlib import Path
import cv2

def compute_mask_stats(depth_path, gt_sky_mask_path=None, tau=0.1):
    """Compute mask statistics for a single image."""
    # Load depth map (.npy format from DAV2)
    try:
        depth_norm = np.load(str(depth_path))
    except:
        # Fallback: try loading as image
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        if len(depth.shape) == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        depth_norm = depth.astype(np.float32) / 255.0
    
    # DAV2 depth is already normalized to [0, 1] during generation
    
    # Create mask: bottom tau (10%) of depth range
    # depth < tau means pixels closer to camera (bottom of normalized range = sky)
    mask = (depth_norm < tau).astype(np.uint8)
    
    # Apply morphological opening to remove small fragments
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Calculate statistics
    total_pixels = mask.size
    masked_pixels = np.sum(mask)
    coverage_ratio = masked_pixels / total_pixels
    
    stats = {
        'coverage_ratio': coverage_ratio,
        'masked_pixels': masked_pixels,
        'total_pixels': total_pixels
    }
    
    # If ground truth sky mask is available, compute overlap
    if gt_sky_mask_path and os.path.exists(gt_sky_mask_path):
        gt_mask = cv2.imread(str(gt_sky_mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            gt_binary = (gt_mask > 127).astype(np.uint8)
            
            # Intersection over Union
            intersection = np.sum((mask & gt_binary))
            union = np.sum((mask | gt_binary))
            iou = intersection / union if union > 0 else 0
            
            # False positive rate: non-sky pixels incorrectly included in mask
            false_positives = np.sum(mask) - np.sum(mask & gt_binary)
            fp_rate = false_positives / np.sum(1 - gt_binary) if np.sum(1 - gt_binary) > 0 else 0
            
            # False negative rate: sky pixels incorrectly excluded
            false_negatives = np.sum(gt_binary) - np.sum(mask & gt_binary)
            fn_rate = false_negatives / np.sum(gt_binary) if np.sum(gt_binary) > 0 else 0
            
            stats['iou'] = iou
            stats['fp_rate'] = fp_rate
            stats['fn_rate'] = fn_rate
    
    return stats

def analyze_dataset(dataset_path, tau_values=[0.05, 0.1, 0.15, 0.2]):
    """Analyze mask coverage across a dataset with different tau values."""
    depth_files = list(Path(dataset_path).glob('*.npy'))
    
    results = {}
    for tau in tau_values:
        coverages = []
        for f in depth_files:
            stats = compute_mask_stats(f, tau=tau)
            if stats:
                coverages.append(stats['coverage_ratio'])
        
        if coverages:
            results[f'tau_{tau}'] = {
                'mean': np.mean(coverages),
                'std': np.std(coverages),
                'min': np.min(coverages),
                'max': np.max(coverages),
                'n_samples': len(coverages)
            }
    
    return results

def main():
    # Dataset paths (pointing to depth_dav2 directories)
    datasets = {
        'NH-HAZE': 'data/NH-HAZE/depth_dav2',
        'O-HAZE': 'data/O-HAZE/depth_dav2', 
        'I-HAZE': 'data/I-HAZE/depth_dav2',
        'SOTS-Outdoor': 'data/RESIDE/SOTS/outdoor/depth_dav2',
    }
    
    print("=" * 70)
    print("Sky Mask Coverage Statistics (τ = threshold)")
    print("=" * 70)
    
    all_results = {}
    for name, path in datasets.items():
        if os.path.exists(path):
            print(f"\n{name}:")
            results = analyze_dataset(path, tau_values=[0.1])
            for tau_key, stats in results.items():
                print(f"  τ=0.1: coverage = {stats['mean']*100:.2f}% ± {stats['std']*100:.2f}% "
                      f"(range: {stats['min']*100:.1f}% - {stats['max']*100:.1f}%, n={stats['n_samples']})")
                all_results[f"{name}_{tau_key}"] = stats
    
    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Mean Coverage at τ=0.1")
    print("=" * 70)
    print(f"{'Dataset':<20} {'Coverage':<15} {'Std':<10} {'Range':<20}")
    print("-" * 70)
    
    for name, path in datasets.items():
        key = f"{name}_tau_0.1"
        if key in all_results:
            s = all_results[key]
            print(f"{name:<20} {s['mean']*100:>6.2f}%       {s['std']*100:>6.2f}%   "
                  f"[{s['min']*100:>5.1f}%, {s['max']*100:>5.1f}%]")
    
    # Generate LaTeX table
    print("\n" + "=" * 70)
    print("LaTeX Table Code:")
    print("=" * 70)
    latex = """
\\begin{table}[t]
\\centering
\\caption{Sky mask coverage statistics at $\\tau = 0.1$. Values represent mean \\textpm\\ std across dataset images.}
\\label{tab:mask_coverage}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Dataset} & \\textbf{Coverage (\\%)} & \\textbf{Range (\\%)} & \\textbf{Samples} \\\\
\\midrule
"""
    
    for name, path in datasets.items():
        key = f"{name}_tau_0.1"
        if key in all_results:
            s = all_results[key]
            latex += f"{name} & ${s['mean']*100:.1f} \\pm {s['std']*100:.1f}$ & $[{s['min']*100:.1f}, {s['max']*100:.1f}]$ & {s['n_samples']} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    print(latex)
    
    return all_results

if __name__ == '__main__':
    main()