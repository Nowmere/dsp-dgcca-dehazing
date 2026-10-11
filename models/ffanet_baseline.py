import torch
import torch.nn as nn

# ==========================================
# 基础组件保持不变 (CA, PA, FFABlock)
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
# 消融实验对照组：纯净版 FFA-Net (无 Depth)
# ==========================================
class FFA_Baseline_Net(nn.Module):
    def __init__(self, gps=3, blocks=3, channels=64):
        super(FFA_Baseline_Net, self).__init__()
        
        # 1. 只有 RGB 浅层特征提取
        self.pre_conv = nn.Conv2d(3, channels, 3, padding=1, bias=True)
        
        # ❌ 没有深度特征提取！
        # ❌ 没有 DG-CCA 跨模态融合！
        
        # 2. FFA 深层特征提取网络
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
        # ⚠️ 注意这里：输入只有 rgb，没有 depth
        x = self.pre_conv(rgb)
        
        # 直接送入 FFA 级联组
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        
        out = self.post_conv(res3)
        return out + rgb

if __name__ == "__main__":
    print("🚀 正在测试 FFA-Baseline 残血版模型...")
    model = FFA_Baseline_Net().cuda()
    dummy_rgb = torch.randn(2, 3, 256, 256).cuda()
    # 不需要深度图了
    out = model(dummy_rgb)
    print(f"✅ 输出维度: {out.shape}")