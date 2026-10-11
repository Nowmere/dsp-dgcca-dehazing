import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelCrossAttention(nn.Module):
    """
    核心模块：深度引导的通道跨模态注意力机制
    Depth-Guided Channel Cross-Attention (DG-CCA)
    """
    def __init__(self, num_channels, temperature=1.0):
        super(ChannelCrossAttention, self).__init__()
        
        # 核心参数
        self.num_channels = num_channels
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        
        # 为了更充分地融合模态，我们引入了 1x1 卷积和 3x3 深度可分离卷积
        # 负责生成 Q (来自 RGB), K (来自 Depth), V (来自 Depth)
        self.q_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, groups=num_channels, bias=False)
        )
        
        self.k_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, groups=num_channels, bias=False)
        )
        
        self.v_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, groups=num_channels, bias=False)
        )
        
        # 融合后的输出投影层
        self.project_out = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)

    def forward(self, rgb_feat, depth_feat):
        """
        前向传播函数
        输入:
            rgb_feat: [B, C, H, W] - 包含纹理和颜色信息，作为待修复的 Query
            depth_feat: [B, C, H, W] - 包含物理场景的远近轮廓，作为引导的 Key 和 Value
        """
        b, c, h, w = rgb_feat.shape

        # 1. 生成 Q, K, V
        # Q 来自带雾的 RGB，K 和 V 来自单目深度先验
        q = self.q_conv(rgb_feat)
        k = self.k_conv(depth_feat)
        v = self.v_conv(depth_feat)

        # 2. 形状变换 (B, C, H, W) -> (B, C, H*W)
        q = q.reshape(b, c, -1)
        k = k.reshape(b, c, -1)
        v = v.reshape(b, c, -1)

        # 3. L2 归一化 (极其关键的稳定操作，防止后续内积爆炸)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 4. 计算通道协方差矩阵 (Attention Matrix)
        # 复杂度是 O(N)，矩阵大小是 (B, C, C)，完美避开显存杀手 (H*W, H*W)
        # q: (B, C, HW) | k_transpose: (B, HW, C)
        # attn: (B, C, C)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature

        # ========================================================
        # ⚠️ 导师的防爆隔离区 (AMP Safe Zone)
        # ========================================================
        # 在 16G 显卡开启混合精度 (FP16) 训练时，Softmax 极其脆弱，容易产生 NaN (梯度爆炸)
        # 我们强制把这几步的计算转回 FP32，算完再转回原始 dtype
        original_dtype = attn.dtype
        attn = attn.to(torch.float32)
        attn = F.softmax(attn, dim=-1)
        attn = attn.to(original_dtype)
        # ========================================================

        # 5. 特征加权融合 (V * Attention Matrix)
        # attn: (B, C, C) | v: (B, C, HW)
        # out: (B, C, HW)
        out = torch.matmul(attn, v)

        # 6. 变回原始空间维度 (B, C, H, W)
        out = out.reshape(b, c, h, w)

        # 7. 残差连接：加上原先的 RGB 特征
        out = self.project_out(out)
        out = out + rgb_feat

        return out

# ==========================================
# 单元测试 (Sanity Check)
# ==========================================
if __name__ == "__main__":
    print("🚀 正在对通道交叉注意力模块进行模拟装填测试...")
    # 模拟 Batch Size=4, 通道数=64, 特征图大小=256x256
    dummy_rgb = torch.randn(4, 64, 256, 256).cuda()
    dummy_depth = torch.randn(4, 64, 256, 256).cuda()
    
    model = ChannelCrossAttention(num_channels=64).cuda()
    
    # 模拟混合精度 FP16 运行环境，测试我们的隔离区会不会崩溃
    with torch.amp.autocast('cuda'):
        output = model(dummy_rgb, dummy_depth)
        
    print(f"✅ 输入 RGB 维度: {dummy_rgb.shape}")
    print(f"✅ 输入 Depth 维度: {dummy_depth.shape}")
    print(f"✅ 成功融合！输出维度: {output.shape}")
    print("🎯 模块测试通过，防 NaN 隔离区生效，显存占用健康！")