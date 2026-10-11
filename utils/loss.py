import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    """
    边缘损失函数 (Edge Loss)
    使用 Sobel 算子提取图像的高频边缘信息，约束模型恢复清晰的物理轮廓。
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # 定义 Sobel 卷积核 (水平和垂直方向)
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        
        # 边缘提取的滤波器权重
        self.loss = nn.L1Loss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        # 简单的拉普拉斯高频提取
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4
        filtered_up = self.conv_gauss(new_filter)
        return current - filtered_up

    def forward(self, x, y):
        # 计算预测图和清晰图在边缘高频信息上的 L1 差异
        x_edge = self.laplacian_kernel(x)
        y_edge = self.laplacian_kernel(y)
        return self.loss(x_edge, y_edge)

class CompositeLoss(nn.Module):
    """
    终极复合损失函数 (Composite Loss)
    包含全局色彩重建 (L1) 和 局部边缘重建 (Edge Loss)
    """
    def __init__(self, edge_weight=0.05):
        super(CompositeLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.edge_loss = EdgeLoss()
        self.edge_weight = edge_weight

    def forward(self, pred, target):
        # 1. 计算基础像素差异
        loss_l1 = self.l1_loss(pred, target)
        
        # 2. 计算边缘轮廓差异
        loss_edge = self.edge_loss(pred, target)
        
        # 3. 按照权重组合 (边缘损失数值通常较大，所以赋予较小的权重平滑梯度)
        total_loss = loss_l1 + (self.edge_weight * loss_edge)
        
        return total_loss

if __name__ == "__main__":
    print("🚀 正在测试复合损失函数...")
    dummy_pred = torch.randn(4, 3, 256, 256).cuda()
    dummy_gt = torch.randn(4, 3, 256, 256).cuda()
    
    criterion = CompositeLoss(edge_weight=0.05).cuda()
    loss = criterion(dummy_pred, dummy_gt)
    
    print(f"✅ Loss 计算成功: {loss.item():.4f}")