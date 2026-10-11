import torch
import torch.nn as nn

# ==========================================
# 基础组件：边缘损失 (与 v2 一致)
# ==========================================
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, x, y):
        x_diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        x_diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        y_diff_h = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
        y_diff_w = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
        return torch.mean(torch.abs(x_diff_h - y_diff_h)) + torch.mean(torch.abs(x_diff_w - y_diff_w))


# ==========================================
# V3: 自适应天空 TV Loss（带深度置信度门控）
# ==========================================
class CompositeLoss_SkyTV_Adaptive(nn.Module):
    def __init__(self, edge_weight=0.05, tv_weight=0.01, sky_threshold=0.1, depth_std_threshold=0.08):
        """
        edge_weight: 边缘损失权重
        tv_weight: 靶向天空 TV 损失权重
        sky_threshold: 天空掩码阈值 (depth < 0.1 → 天空)
        depth_std_threshold: 深度置信度阈值。
            - 如果某张图的深度图标准差 < 此阈值 → 说明深度图几乎均匀（DAV2 在浓雾中失效）
              → 关闭该图的 TV Loss（tv_weight=0），防止反向伤害
            - 正常室外场景深度 std 通常在 0.15-0.35
        """
        super(CompositeLoss_SkyTV_Adaptive, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.edge_loss = EdgeLoss()
        self.edge_weight = edge_weight
        self.tv_weight = tv_weight
        self.sky_threshold = sky_threshold
        self.depth_std_threshold = depth_std_threshold

    def forward(self, preds, clear, depth):
        # 1. 基础重建损失 (全局色彩映射)
        loss_l1 = self.l1_loss(preds, clear)

        # 2. 边缘保护损失
        loss_edge = self.edge_loss(preds, clear)

        # 3. 自适应天空 TV Loss (Per-sample depth confidence gating)
        # depth: [B, 1, H, W]
        B = preds.size(0)
        loss_tv_total = 0.0

        for b in range(B):
            # 当前样本的深度图
            d_single = depth[b:b+1]  # [1, 1, H, W]
            p_single = preds[b:b+1]  # [1, 3, H, W]

            # 计算深度图的空间标准差 (排除 batch 维度后的所有空间像素)
            d_std = d_single.std()

            if d_std < self.depth_std_threshold:
                # 深度图几乎均匀 → DAV2 在该图上失效 → 跳过 TV Loss
                continue

            # 生成天空掩码 (depth < 0.1 的位置)
            sky_mask_inverted = (d_single < self.sky_threshold).float()

            # 检查掩码覆盖率：如果没有天空区域，也跳过
            sky_ratio = sky_mask_inverted.mean()
            if sky_ratio < 0.01:  # 天空占比 < 1% → 没有天空
                continue

            # 将预测图中非天空部分抹零
            pred_sky = p_single * sky_mask_inverted

            # TV Loss
            tv_h = torch.mean(torch.abs(pred_sky[:, :, 1:, :] - pred_sky[:, :, :-1, :]))
            tv_w = torch.mean(torch.abs(pred_sky[:, :, :, 1:] - pred_sky[:, :, :, :-1]))
            loss_tv_total += (tv_h + tv_w)

        # 平均（防止 batch size 不同时 loss 尺度变化）
        loss_tv = loss_tv_total / max(B, 1)

        # 4. 加权总和
        total_loss = loss_l1 + (self.edge_weight * loss_edge) + (self.tv_weight * loss_tv)

        return total_loss