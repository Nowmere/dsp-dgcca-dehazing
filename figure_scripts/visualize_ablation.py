import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 导入两个模型
from models.ffanet import FFA_DGCCA_Net
from models.ffanet_baseline import FFA_Baseline_Net

def visualize_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 实例化并加载两个模型的权重
    print("🚀 加载完全体模型...")
    model_ours = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(device)
    model_ours.load_state_dict(torch.load(os.path.join(base_dir, 'checkpoints_ots', 'ffa_dgcca_ots_epoch_20.pth'), map_location=device, weights_only=True))
    model_ours.eval()

    print("🚀 加载 Baseline 模型...")
    model_base = FFA_Baseline_Net(gps=3, blocks=3, channels=64).to(device)
    model_base.load_state_dict(torch.load(os.path.join(base_dir, 'checkpoints_baseline', 'ffa_baseline_epoch_20.pth'), map_location=device, weights_only=True))
    model_base.eval()

    # 2. 挑选一张典型的室外测试图 (你可以换成你想看的图的名字)
    # 我们随便挑一张，比如 '0001_0.8_0.2.jpg'
    hazy_dir = os.path.join(base_dir, 'data', 'RESIDE', 'SOTS', 'outdoor', 'hazy')
    clear_dir = os.path.join(base_dir, 'data', 'RESIDE', 'SOTS', 'outdoor', 'clear')
    depth_dir = os.path.join(base_dir, 'data', 'RESIDE', 'SOTS', 'outdoor', 'depth_hazy')
    
    # 自动找第一张图
    # 🎯 导师的精确狙击指令：锁定第一名！
    test_img_name = "0108_1_0.2.jpg"  # 你也可以换成第二名 "0108_1_0.2.jpg" 试试
    clear_id = test_img_name.split('_')[0]
    
    hazy_path = os.path.join(hazy_dir, test_img_name)
    clear_path = os.path.join(clear_dir, f"{clear_id}.png")
    if not os.path.exists(clear_path):
        clear_path = os.path.join(clear_dir, f"{clear_id}.jpg")
    depth_path = os.path.join(depth_dir, f"{test_img_name.split('.')[0]}.png")

    hazy_pil = Image.open(hazy_path).convert('RGB')
    clear_pil = Image.open(clear_path).convert('RGB')
    depth_pil = Image.open(depth_path).convert('L')

    hazy_tensor = TF.to_tensor(hazy_pil).unsqueeze(0).to(device)
    depth_tensor = TF.to_tensor(depth_pil).unsqueeze(0).to(device)

    # 3. 双模型推理
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            pred_ours = model_ours(hazy_tensor, depth_tensor)
            pred_base = model_base(hazy_tensor)
            
    pred_ours = torch.clamp(pred_ours, 0, 1).squeeze().cpu().numpy().transpose(1, 2, 0)
    pred_base = torch.clamp(pred_base, 0, 1).squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # 4. 画图对比
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(hazy_pil); axes[0].set_title("Input (Hazy)"); axes[0].axis('off')
    axes[1].imshow(pred_base); axes[1].set_title("Baseline (FFA-Net)"); axes[1].axis('off')
    axes[2].imshow(pred_ours); axes[2].set_title("Ours (DG-CCA)"); axes[2].axis('off')
    axes[3].imshow(clear_pil); axes[3].set_title("Ground Truth"); axes[3].axis('off')

    out_path = os.path.join(base_dir, 'ablation_visual.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"\n✅ 视觉对比图已保存至: {out_path}")

if __name__ == "__main__":
    visualize_comparison()