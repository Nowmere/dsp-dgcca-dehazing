import torch
import torch.nn as nn
from models.fusion import ChannelCrossAttention

# ==========================================
# 基础组件：通道注意力 (CA) 和 像素注意力 (PA)
# ==========================================
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

# ==========================================
# 核心组件：FFA Block
# ==========================================
class FFABlock(nn.Module):
    def __init__(self, channel):
        super(FFABlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.calayer = CALayer(channel)
        self.palayer = PALayer(channel)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

# ==========================================
# 终极形态：融合深度先验的 FFA-Net
# ==========================================
class FFA_DGCCA_Net(nn.Module):
    def __init__(self, gps=3, blocks=3, channels=64):
        """
        gps: FFA Group 的数量
        blocks: 每个 Group 里的 Block 数量
        channels: 内部通道数 (默认64，16G显存绝对安全)
        """
        super(FFA_DGCCA_Net, self).__init__()
        
        # 1. RGB 浅层特征提取
        self.rgb_pre_conv = nn.Conv2d(3, channels, 3, padding=1, bias=True)
        
        # 2. Depth 浅层特征提取 (将 1 通道深度图升维)
        self.depth_pre_conv = nn.Sequential(
            nn.Conv2d(1, channels // 2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, padding=1, bias=True)
        )
        
        # 3. ⭐️ 插入你的核心王牌：跨模态通道注意力
        self.fusion = ChannelCrossAttention(num_channels=channels)
        
        # 4. FFA 深层特征提取网络 (堆叠多个 FFA Block)
        # 为了防爆显存，我们将参数量控制在合理范围
        self.g1 = nn.Sequential(*[FFABlock(channels) for _ in range(blocks)])
        self.g2 = nn.Sequential(*[FFABlock(channels) for _ in range(blocks)])
        self.g3 = nn.Sequential(*[FFABlock(channels) for _ in range(blocks)])
        
        # 5. 后期特征重建
        self.post_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1, bias=True)
        )

    def forward(self, rgb, depth):
        # ==================== 🌟 新增的天空掩码补丁 ====================
        sky_mask = torch.clamp(depth / 0.1, 0.0, 1.0)
        clean_depth = depth * sky_mask
        # =====================================================================

        rgb_feat = self.rgb_pre_conv(rgb)
        depth_feat = self.depth_pre_conv(clean_depth)
        
        # ⭐️ 跨模态融合：用 Depth 的全局轮廓去引导 RGB 特征
        fused_feat = self.fusion(rgb_feat, depth_feat)
        
        # 送入 FFA 级联组进行深层去雾
        res1 = self.g1(fused_feat)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        
        # 重建清晰图像 (加上原始 RGB 的全局残差，防止颜色偏移)
        out = self.post_conv(res3)
        return out + rgb

# ==========================================
# 参照组：纯净版 Baseline (纯 FFA-Net，无深度先验)
# ==========================================
class FFA_Baseline_Net(nn.Module):
    def __init__(self, gps=3, blocks=3, channels=64):
        super(FFA_Baseline_Net, self).__init__()
        
        # 1. 只有 RGB 浅层特征提取 (没有 Depth 分支)
        self.pre_conv = nn.Conv2d(3, channels, 3, padding=1, bias=True)
        
        # 2. FFA 深层特征提取网络 (参数量和完全体保持绝对一致)
        self.g1 = nn.Sequential(*[FFABlock(channels) for _ in range(blocks)])
        self.g2 = nn.Sequential(*[FFABlock(channels) for _ in range(blocks)])
        self.g3 = nn.Sequential(*[FFABlock(channels) for _ in range(blocks)])
        
        # 3. 后期特征重建
        self.post_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1, bias=True)
        )

    def forward(self, rgb):
        # 纯粹的 2D 瞎子瞎算模式：直接吃 RGB，没有深度融合，没有天空掩码
        feat = self.pre_conv(rgb)
        
        res1 = self.g1(feat)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        
        out = self.post_conv(res3)
        return out + rgb

if __name__ == "__main__":
    print("🚀 正在测试 FFA-Net 深度融合版大模型...")
    model = FFA_DGCCA_Net().cuda()
    dummy_rgb = torch.randn(2, 3, 256, 256).cuda()
    dummy_depth = torch.randn(2, 1, 256, 256).cuda()
    
    with torch.cuda.amp.autocast():
        out = model(dummy_rgb, dummy_depth)
        
    print(f"✅ 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"✅ 输出维度: {out.shape}")
    print("🎯 FFA-DGCCA 架构连通性测试通过！")