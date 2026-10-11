import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.ffanet import FFA_DGCCA_Net
from models.ffanet_baseline import FFA_Baseline_Net

import matplotlib.cm as cm

def plot_paper_ready_visuals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 加载模型
    model_ours = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(device)
    model_ours.load_state_dict(torch.load(os.path.join(base_dir, 'checkpoints_ots', 'ffa_dgcca_ots_epoch_20.pth'), map_location=device, weights_only=True))
    model_ours.eval()

    model_base = FFA_Baseline_Net(gps=3, blocks=3, channels=64).to(device)
    model_base.load_state_dict(torch.load(os.path.join(base_dir, 'checkpoints_baseline', 'ffa_baseline_epoch_20.pth'), map_location=device, weights_only=True))
    model_base.eval()

    # 2. 依然找刚才那张图
    hazy_dir = os.path.join(base_dir, 'data', 'RESIDE', 'SOTS', 'outdoor', 'hazy')
    clear_dir = os.path.join(base_dir, 'data', 'RESIDE', 'SOTS', 'outdoor', 'clear')
    depth_dir = os.path.join(base_dir, 'data', 'RESIDE', 'SOTS', 'outdoor', 'depth_hazy')
    
    test_img_name = [f for f in os.listdir(hazy_dir) if f.endswith(('.jpg', '.png'))][0]
    clear_id = test_img_name.split('_')[0]
    
    hazy_path = os.path.join(hazy_dir, test_img_name)
    clear_path = os.path.join(clear_dir, f"{clear_id}.png")
    if not os.path.exists(clear_path): clear_path = os.path.join(clear_dir, f"{clear_id}.jpg")
    depth_path = os.path.join(depth_dir, f"{test_img_name.split('.')[0]}.png")

    hazy_pil = Image.open(hazy_path).convert('RGB')
    clear_pil = Image.open(clear_path).convert('RGB')
    depth_pil = Image.open(depth_path).convert('L')

    hazy_tensor = TF.to_tensor(hazy_pil).unsqueeze(0).to(device)
    depth_tensor = TF.to_tensor(depth_pil).unsqueeze(0).to(device)

    # 3. 推理
    with torch.no_grad(), torch.amp.autocast('cuda'):
        pred_ours = model_ours(hazy_tensor, depth_tensor)
        pred_base = model_base(hazy_tensor)
            
    img_hazy = hazy_pil
    img_base = TF.to_pil_image(torch.clamp(pred_base.squeeze(), 0, 1))
    img_ours = TF.to_pil_image(torch.clamp(pred_ours.squeeze(), 0, 1))
    img_gt = clear_pil

    # 🌟 4. 定义我们要“拿放大镜看”的区域 (x, y, 宽, 高)
    # 这个坐标你需要根据实际图片的尺寸慢慢微调，找一个最能体现你优势的地方
    # 如果你不知道坐标，取消下面两行的注释，运行代码会弹出一个大图。
    # 你的鼠标放在哪里，窗口右下角就会实时显示 (x, y) 坐标！
    # 记下你满意的坐标后，关掉窗口，把坐标填到下面的变量里。
    
    # plt.imshow(img_gt)
    # plt.show() 
    
    # 在上面找到心仪的坐标后，填在这里：
    crop_x, crop_y, crop_w, crop_h = 100, 350, 100, 100  # w 和 h 保持 100 通常是放大的最佳比例
    # =================================================== 
    
    def get_crop(img):
        return img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

    crop_hazy = get_crop(img_hazy)
    crop_base = get_crop(img_base)
    crop_ours = get_crop(img_ours)
    crop_gt = get_crop(img_gt)

    # 5. 画出带有红框的排版图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    images = [img_hazy, img_base, img_ours, img_gt]
    titles = ["Hazy", "Baseline", "Ours", "GT"]
    crops = [crop_hazy, crop_base, crop_ours, crop_gt]

    for i in range(4):
        # 第一排：原图并画上红框
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(titles[i])
        axes[0, i].axis('off')
        rect = patches.Rectangle((crop_x, crop_y), crop_w, crop_h, linewidth=2, edgecolor='r', facecolor='none')
        axes[0, i].add_patch(rect)
        
        # 第二排：放大的局部切片
        axes[1, i].imshow(crops[i])
        axes[1, i].axis('off')

    plt.tight_layout()
    out_path = os.path.join(base_dir, 'paper_ready_visual.png')
    plt.savefig(out_path, dpi=300)
    print(f"✅ 论文级局部放大对比图已保存至: {out_path}")

if __name__ == "__main__":
    plot_paper_ready_visuals()