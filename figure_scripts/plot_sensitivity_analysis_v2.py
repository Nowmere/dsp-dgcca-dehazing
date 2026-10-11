"""
绘制敏感性分析图 V2：Δ 相对量 + PSNR/SSIM 双指标
- 4个子图：λ扫描(PSNR)、λ扫描(SSIM)、τ扫描(PSNR)、τ扫描(SSIM)
- Y轴为 Δ(相对于默认参数的变化)，范围 [-0.05, +0.05] 更敏感
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = "e:/Dehazing_Project/paper/CVPR_2026_Submission_Template/figs"
import os
os.makedirs(BASE, exist_ok=True)
OUTPUT = os.path.join(BASE, "sensitivity_analysis_v2.png")

# ========== 1. 字体与样式 ==========
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 1.8
plt.rcParams['lines.markersize'] = 7

# ========== 2. 数据（从 project_memory_core.md 提取）==========
# 默认值：τ=0.10, λ=0.01 → O-HAZE: 15.08/0.6387, NH-HAZE: 12.79/0.4942, I-HAZE: 16.12/0.7577

# λ 扫描 (τ=0.10 固定)
lambda_x = [0.001, 0.005, 0.010, 0.050]
# PSNR
lambda_ohaze_psnr = [15.11, 15.10, 15.08, 15.11]
lambda_nhaze_psnr = [12.81, 12.80, 12.79, 12.80]
lambda_ihaze_psnr = [16.10, 16.10, 16.12, 16.07]
# SSIM
lambda_ohaze_ssim = [0.6393, 0.6392, 0.6387, 0.6382]
lambda_nhaze_ssim = [0.4953, 0.4952, 0.4942, 0.4940]
lambda_ihaze_ssim = [0.7567, 0.7568, 0.7577, 0.7558]

# τ 扫描 (λ=0.01 固定)
tau_x = [0.05, 0.10, 0.15, 0.20]
# PSNR
tau_ohaze_psnr = [15.10, 15.08, 15.11, 15.10]
tau_nhaze_psnr = [12.80, 12.79, 12.80, 12.80]
tau_ihaze_psnr = [16.13, 16.12, 16.08, 16.12]
# SSIM
tau_ohaze_ssim = [0.6390, 0.6387, 0.6388, 0.6387]
tau_nhaze_ssim = [0.4953, 0.4942, 0.4949, 0.4947]
tau_ihaze_ssim = [0.7574, 0.7577, 0.7552, 0.7582]

# 默认参数索引（λ=0.01, τ=0.10）
lambda_base_idx = 2  # λ=0.010
tau_base_idx = 1     # τ=0.10

# ========== 3. 计算 Δ (相对于默认值的变化) ==========
def calc_delta(values, base_idx):
    """计算相对于默认值的差值"""
    base = values[base_idx]
    return [v - base for v in values]

# λ 扫描 Δ
lambda_ohaze_d_psnr = calc_delta(lambda_ohaze_psnr, lambda_base_idx)
lambda_nhaze_d_psnr = calc_delta(lambda_nhaze_psnr, lambda_base_idx)
lambda_ihaze_d_psnr = calc_delta(lambda_ihaze_psnr, lambda_base_idx)
lambda_ohaze_d_ssim = calc_delta(lambda_ohaze_ssim, lambda_base_idx)
lambda_nhaze_d_ssim = calc_delta(lambda_nhaze_ssim, lambda_base_idx)
lambda_ihaze_d_ssim = calc_delta(lambda_ihaze_ssim, lambda_base_idx)

# τ 扫描 Δ
tau_ohaze_d_psnr = calc_delta(tau_ohaze_psnr, tau_base_idx)
tau_nhaze_d_psnr = calc_delta(tau_nhaze_psnr, tau_base_idx)
tau_ihaze_d_psnr = calc_delta(tau_ihaze_psnr, tau_base_idx)
tau_ohaze_d_ssim = calc_delta(tau_ohaze_ssim, tau_base_idx)
tau_nhaze_d_ssim = calc_delta(tau_nhaze_ssim, tau_base_idx)
tau_ihaze_d_ssim = calc_delta(tau_ihaze_ssim, tau_base_idx)

# ========== 4. 绘图 ==========
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.subplots_adjust(hspace=0.35, wspace=0.25)

color_o = '#1f77b4'  # 蓝色 - O-HAZE
color_n = '#ff7f0e'  # 橙色 - NH-HAZE
color_i = '#2ca02c'  # 绿色 - I-HAZE

x_lambda = ['0.001', '0.005', '0.010*', '0.050']
x_tau = ['0.05*', '0.10', '0.15', '0.20']

# 子图 1: λ 扫描 PSNR
ax = axes[0, 0]
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.plot(x_lambda, lambda_ihaze_d_psnr, marker='^', color=color_i, label='I-HAZE')
ax.plot(x_lambda, lambda_ohaze_d_psnr, marker='o', color=color_o, label='O-HAZE')
ax.plot(x_lambda, lambda_nhaze_d_psnr, marker='s', color=color_n, label='NH-HAZE')
ax.set_title(r'(a) $\lambda$ Scan: $\Delta$PSNR (dB)', fontweight='bold')
ax.set_xlabel(r'TV Loss Weight ($\lambda$)')
ax.set_ylabel(r'$\Delta$PSNR (dB)')
ax.set_ylim(-0.05, 0.05)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', ncol=3)
ax.tick_params(axis='x', rotation=15)

# 子图 2: λ 扫描 SSIM
ax = axes[0, 1]
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.plot(x_lambda, lambda_ihaze_d_ssim, marker='^', color=color_i, label='I-HAZE')
ax.plot(x_lambda, lambda_ohaze_d_ssim, marker='o', color=color_o, label='O-HAZE')
ax.plot(x_lambda, lambda_nhaze_d_ssim, marker='s', color=color_n, label='NH-HAZE')
ax.set_title(r'(b) $\lambda$ Scan: $\Delta$SSIM', fontweight='bold')
ax.set_xlabel(r'TV Loss Weight ($\lambda$)')
ax.set_ylabel(r'$\Delta$SSIM')
ax.set_ylim(-0.003, 0.003)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', ncol=3)
ax.tick_params(axis='x', rotation=15)

# 子图 3: τ 扫描 PSNR
ax = axes[1, 0]
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.plot(x_tau, tau_ihaze_d_psnr, marker='^', color=color_i, label='I-HAZE')
ax.plot(x_tau, tau_ohaze_d_psnr, marker='o', color=color_o, label='O-HAZE')
ax.plot(x_tau, tau_nhaze_d_psnr, marker='s', color=color_n, label='NH-HAZE')
ax.set_title(r'(c) $\tau$ Scan: $\Delta$PSNR (dB)', fontweight='bold')
ax.set_xlabel(r'Sky Mask Threshold ($\tau$)')
ax.set_ylabel(r'$\Delta$PSNR (dB)')
ax.set_ylim(-0.05, 0.05)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', ncol=3)
ax.tick_params(axis='x', rotation=15)

# 子图 4: τ 扫描 SSIM
ax = axes[1, 1]
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.plot(x_tau, tau_ihaze_d_ssim, marker='^', color=color_i, label='I-HAZE')
ax.plot(x_tau, tau_ohaze_d_ssim, marker='o', color=color_o, label='O-HAZE')
ax.plot(x_tau, tau_nhaze_d_ssim, marker='s', color=color_n, label='NH-HAZE')
ax.set_title(r'(d) $\tau$ Scan: $\Delta$SSIM', fontweight='bold')
ax.set_xlabel(r'Sky Mask Threshold ($\tau$)')
ax.set_ylabel(r'$\Delta$SSIM')
ax.set_ylim(-0.003, 0.003)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', ncol=3)
ax.tick_params(axis='x', rotation=15)

# 添加说明文字
fig.text(0.5, 0.01, '(* marks the default parameter value; shaded region = ±0.02 dB noise margin)',
         ha='center', fontsize=9, style='italic', color='gray')

plt.savefig(OUTPUT, format='png', bbox_inches='tight', dpi=300)
print(f"Saved: {OUTPUT}")