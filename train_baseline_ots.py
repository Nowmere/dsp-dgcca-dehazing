import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils.dataset_ots import OTSDataset
from utils.loss import CompositeLoss
# 🌟 引入你的 Baseline 模型 (请确保你的 ffanet.py 里有这个类，或者用原来的纯净 FFA-Net)
from models.ffanet import FFA_Baseline_Net 
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # 🚀 引擎外挂开启

def train_baseline_ots():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, 'checkpoints_ots')
    os.makedirs(checkpoint_dir, exist_ok=True)

    hazy_dir = os.path.join(base_dir, 'data', 'RESIDE', 'OTS', 'hazy')
    clear_dir = os.path.join(base_dir, 'data', 'RESIDE', 'OTS', 'clear')
    depth_dir = os.path.join(base_dir, 'data', 'RESIDE', 'OTS', 'depth_hazy')

    ## 🌟 提速版参数配置
    batch_size = 16         # 从 4 提升到 8 (甚至可以大胆试一下 16)
    accumulation_steps = 1  # 从 4 降到 2 (保持 8 * 2 = 16 的等效 Batch)
    num_epochs = 100         
    learning_rate = 1e-4

    dataset = OTSDataset(hazy_dir, clear_dir, depth_dir, crop_size=256, is_train=True)
    # 🌟 增加 num_workers，并开启 pin_memory
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4,       # 如果你的 CPU 是 8 核以上，直接开到 8
                            pin_memory=True,     # 开启显存直通车
                            drop_last=True)
                            
    # 🌟 实例化纯净版 Baseline (没有 DG-CCA)
    model = FFA_Baseline_Net(gps=3, blocks=3, channels=64).to(device)
    
    # ==================== 🌟 断点续传核心代码 ====================
    resume_epoch = 40  # 告诉程序从哪里爬起来
    resume_path = os.path.join(checkpoint_dir, f"ffa_baseline_ots_epoch_{resume_epoch}.pth")
    
    if os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path))
        print(f"♻️ 满血复活！成功加载第 {resume_epoch} 轮存档！")
    else:
        print(f"❌ 找不到存档 {resume_path}，请检查路径！")
        return # 找不到就直接退出，别从头跑
    # =============================================================
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = CompositeLoss(edge_weight=0.05).to(device)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    print(f"======================================================")
    print(f"⚖️ 开启绝对公平对决：Baseline (纯 FFA-Net) 100 轮长跑")
    print(f"======================================================")

    for epoch in range(resume_epoch,num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
        for i, (hazy, clear, _) in enumerate(pbar): # 🌟 忽略 depth
            hazy, clear = hazy.to(device), clear.to(device)

            with torch.amp.autocast('cuda'):
                # 🌟 Baseline 只吃 RGB 雾图！
                preds = model(hazy) 
                loss = criterion(preds, clear) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            current_loss = loss.item() * accumulation_steps
            epoch_loss += current_loss
            
            if i % 10 == 0:
                pbar.set_postfix({'Loss': f"{current_loss:.4f}"})

        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            # 🌟 名字别和完全体弄混了！
            save_path = os.path.join(checkpoint_dir, f"ffa_baseline_ots_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train_baseline_ots()