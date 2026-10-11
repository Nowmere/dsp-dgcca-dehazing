import numpy as np
import matplotlib.pyplot as plt

# ================= ⚙️ 配置区 =================
# 使用 r"" 原生字符串，完美解决 Windows 路径反斜杠报错问题
npy_path = r"E:\Dehazing_Project\data\RESIDE\SOTS\outdoor\depth_dav2\0001_0.8_0.2.npy"
# ============================================

def analyze_depth():
    print("======================================================")
    print("🔍 启动 DAV2 深度图数值探伤引擎")
    print("======================================================")

    # 1. 加载深度数据
    try:
        depth = np.load(npy_path)
    except Exception as e:
        print(f"❌ 读取文件失败，请检查路径: {e}")
        return

    # 2. 核心统计指标计算
    h, w = depth.shape
    print(f"📐 深度图分辨率 (H x W): {h} x {w}")
    print(f"📉 全局最小值: {depth.min():.6f}")
    print(f"📈 全局最大值: {depth.max():.6f}")
    print(f"📊 全局平均值: {depth.mean():.6f}")

    # 3. 针对性分析：切片出“天空区域”
    # 在室外图中，画面最顶部的 20% 区域（高度的 0 到 0.2*H）极大概率纯粹是天空
    sky_region_h = int(h * 0.2)
    sky_region = depth[0:sky_region_h, :]
    
    print("\n☁️ --- 天空区域靶向分析 (画面顶部 20%) ---")
    print(f"平均深度值: {sky_region.mean():.6f}")
    print(f"最大深度值: {sky_region.max():.6f} (如果这大于0.1，说明目前的掩码漏掉了天空噪声！)")
    print(f"最小深度值: {sky_region.min():.6f}")

    # 4. 学术级可视化（帮您彻底吃透数据分布）
    print("\n⏳ 正在生成深度分布直方图，请在弹出的窗口中查看...")
    
    plt.figure(figsize=(14, 5))

    # 左侧：深度图的视觉展现
    plt.subplot(1, 2, 1)
    plt.imshow(depth, cmap='magma') # magma 伪彩色，越亮代表值越大
    plt.colorbar(label='Depth Value (0-1)')
    plt.title('DAV2 Depth Map (Visual)')

    # 右侧：像素值分布直方图
    plt.subplot(1, 2, 2)
    # 将二维数组展平，分成 100 个柱子统计频率
    plt.hist(depth.flatten(), bins=100, color='royalblue', alpha=0.7)
    
    # 画一条红色虚线，标出我们当前模型里使用的 0.1 阈值
    plt.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='Current Soft-Mask Threshold (0.1)')
    
    plt.title('Depth Value Histogram (Distribution)')
    plt.xlabel('Depth Value')
    plt.ylabel('Number of Pixels')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_depth()