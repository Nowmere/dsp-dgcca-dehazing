"""
绘制敏感性分析折线图：λ 扫描 + τ 扫描 → sensitivity_analysis_20260721.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "paper", "CVPR_2026_Submission_Template", "figs", "sensitivity_analysis_20260721.png")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# ========== 1. 字体与样式 ==========
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8

# ========== 2. 数据 ==========
# λ 扫描 (τ=0.10)
lambda_x = ['0.001', '0.005', '0.010', '0.050']
lambda_ohaze = [15.11, 15.10, 15.08, 15.11]
lambda_nhaze = [12.81, 12.80, 12.79, 12.80]
lambda_ihaze = [16.10, 16.10, 16.12, 16.07]

# τ 扫描 (λ=0.01)
tau_x = ['0.05', '0.10', '0.15', '0.20']
tau_ohaze = [15.10, 15.08, 15.11, 15.10]
tau_nhaze = [12.80, 12.79, 12.80, 12.80]
tau_ihaze = [16.13, 16.12, 16.08, 16.12]

# ========== 3. 绘图 ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.subplots_adjust(wspace=0.3)

color_o = '#1f77b4'
color_n = '#ff7f0e'
color_i = '#2ca02c'

# --- 子图 1: λ 扫描 ---
ax1.plot(lambda_x, lambda_ihaze, marker='^', color=color_i, label='I-HAZE')
ax1.plot(lambda_x, lambda_ohaze, marker='o', color=color_o, label='O-HAZE')
ax1.plot(lambda_x, lambda_nhaze, marker='s', color=color_n, label='NH-HAZE')
ax1.set_title(r'(a) Sensitivity to TV Loss Weight ($\lambda$)')
ax1.set_xlabel(r'TV Loss Weight ($\lambda$)')
ax1.set_ylabel('PSNR (dB)')
ax1.set_ylim(12.0, 16.5)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower left')

# --- 子图 2: τ 扫描 ---
ax2.plot(tau_x, tau_ihaze, marker='^', color=color_i, label='I-HAZE')
ax2.plot(tau_x, tau_ohaze, marker='o', color=color_o, label='O-HAZE')
ax2.plot(tau_x, tau_nhaze, marker='s', color=color_n, label='NH-HAZE')
ax2.set_title(r'(b) Sensitivity to Sky Mask ($\tau$)')
ax2.set_xlabel(r'Spatial Sky Mask Threshold ($\tau$)')
ax2.set_ylabel('PSNR (dB)')
ax2.set_ylim(12.0, 16.5)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower left')

plt.savefig(OUTPUT, format='png', bbox_inches='tight', dpi=300)
print(f"Saved: {OUTPUT}")