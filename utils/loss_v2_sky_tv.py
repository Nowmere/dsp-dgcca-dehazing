import torch
import torch.nn as nn

# ==========================================
# 基础组件：边缘损失 (保护前景物体的物理轮廓)
# ==========================================
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, x, y):
        # 计算预测图与清晰图在水平和垂直方向的梯度差异
        x_diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        x_diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        y_diff_h = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
        y_diff_w = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
        # 梯度 L1 损失
        return torch.mean(torch.abs(x_diff_h - y_diff_h)) + torch.mean(torch.abs(x_diff_w - y_diff_w))

# ==========================================
# 🚀 终极武器：融合天空平滑机制的复合损失
# ==========================================
class CompositeLoss_SkyTV(nn.Module):
    def __init__(self, edge_weight=0.05, tv_weight=0.01, sky_threshold=0.1):
        """
        edge_weight: 边缘损失权重 (维持原有设定的 0.05)
        tv_weight: 靶向天空 TV 损失权重 (设置为 0.01，轻微约束即可，防止画面模糊)
        sky_threshold: 天空掩码阈值 (DAV2 下设定为 0.1)
        """
        super(CompositeLoss_SkyTV, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.edge_loss = EdgeLoss()
        self.edge_weight = edge_weight
        self.tv_weight = tv_weight
        self.sky_threshold = sky_threshold

    def forward(self, preds, clear, depth):
        # 1. 基础重建损失 (全局色彩映射)
        loss_l1 = self.l1_loss(preds, clear)

        # 2. 边缘保护损失 (防止去雾后物体边缘发虚)
        loss_edge = self.edge_loss(preds, clear)

        # 3. 🎯 靶向天空平滑损失 (Targeted Sky TV Loss)
        # 生成反向掩码：找出天空区域（深度值小于 0.1 的地方设为 1，其余设为 0）
        sky_mask_inverted = (depth < self.sky_threshold).float()
        
        # 将预测图与反向掩码相乘，把除了天空以外的像素全部抹零
        pred_sky = preds * sky_mask_inverted
        
        # 计算天空区域的全变分损失 (Total Variation)
        # 强制相邻像素的值尽可能接近，消除由于深度切断导致的“天空高频噪点”
        tv_h = torch.mean(torch.abs(pred_sky[:, :, 1:, :] - pred_sky[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(pred_sky[:, :, :, 1:] - pred_sky[:, :, :, :-1]))
        loss_tv = tv_h + tv_w

        # 4. 加权总和
        total_loss = loss_l1 + (self.edge_weight * loss_edge) + (self.tv_weight * loss_tv)
        
        return total_loss