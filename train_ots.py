import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils.dataset_ots import OTSDataset
from utils.loss import CompositeLoss
from models.ffanet import FFA_DGCCA_Net

def train_ots():
    # 1. 环境与路径配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    checkpoint_dir = os.path.join(base_dir, 'checkpoints_ots')
    hazy_dir = os.path.join(base_dir, 'data', 'RESIDE', 'OTS', 'hazy')
    clear_dir = os.path.join(base_dir, 'data', 'RESIDE', 'OTS', 'clear')
    depth_dir = os.path.join(base_dir, 'data', 'RESIDE', 'OTS', 'depth_dav2') 

    # ================= 🌟 断点续传配置区 🌟 =================
    RESUME_TRAINING = True  # 是否开启断点续传
    RESUME_EPOCH = 82       # ⚠️ 填入你中断的 Epoch 数量
    # ⚠️ 请去 checkpoints_ots 文件夹核对一下名字是不是这个！
    # 如果你之前存的名字没有 _dav2，请把 _dav2 删掉
    RESUME_PATH = os.path.join(checkpoint_dir, f"ffa_dgcca_ots_dav2_epoch_{RESUME_EPOCH}.pth") 
    # =========================================================

    # 2. 核心训练参数
    batch_size = 4           
    accumulation_steps = 4    
    num_epochs = 100          
    learning_rate = 1e-4

    # 3. 载入数据集
    print(f"⏳ 正在初始化数据集加载器...")
    dataset = OTSDataset(hazy_dir, clear_dir, depth_dir, crop_size=256, is_train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # 4. 模型、损失函数、优化器
    model = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = CompositeLoss(edge_weight=0.05).to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # 🌟 5. 执行断点续传抢救手术 🌟
    start_epoch = 0
    if RESUME_TRAINING and os.path.exists(RESUME_PATH):
        print(f"🔄 检测到存档文件，正在从 Epoch {RESUME_EPOCH} 恢复模型...")
        # 加载权重 (加入 weights_only=True 消除警告，如果报错可删掉这个参数)
        checkpoint = torch.load(RESUME_PATH, map_location=device, weights_only=False)
        
        # 兼容老版本的纯权重存档和新版本的全状态存档
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✅ 完美恢复：发现全状态存档！优化器与学习率已无损接管。")
        else:
            model.load_state_dict(checkpoint)
            print("⚠️ 兼容恢复：发现纯权重存档。")
            print(f"⏩ 正在快进学习率调度器至第 {RESUME_EPOCH} 轮...")
            for _ in range(RESUME_EPOCH):
                scheduler.step()
                
        start_epoch = RESUME_EPOCH
    elif RESUME_TRAINING:
        print(f"❌ 警告：找不到存档文件 {RESUME_PATH}！将从头开始训练。")

    print(f"======================================================")
    print(f"🔥 OTS 室外霸主计划 | 续传启动")
    print(f"🚀 当前进度: 从 Epoch {start_epoch + 1} 冲刺至 {num_epochs}")
    print(f"======================================================")

    # 🌟 6. 训练主循环 (注意 range 改成了从 start_epoch 开始)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        optimizer.zero_grad() 
        
        for i, (hazy, clear, depth) in enumerate(pbar):
            hazy, clear, depth = hazy.to(device), clear.to(device), depth.to(device)

            with torch.amp.autocast('cuda'):
                preds = model(hazy, depth)
                loss = criterion(preds, clear)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            current_loss = loss.item() * accumulation_steps
            epoch_loss += current_loss
            
            if i % 10 == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({'Loss': f"{current_loss:.4f}", 'LR': f"{curr_lr:.2e}"})

        # 每个 Epoch 结束后的逻辑
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"✅ Epoch [{epoch+1}/{num_epochs}] 完成 | 平均 Loss: {avg_loss:.4f}")
        
        # 🌟 7. 升级版史诗级存档：保存所有状态！
        if (epoch + 1) % 2 == 0 or (epoch + 1) == num_epochs:
            save_path = os.path.join(checkpoint_dir, f"ffa_dgcca_ots_dav2_epoch_{epoch+1}.pth")
            
            # 打包所有状态
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss
            }
            torch.save(checkpoint_dict, save_path)
            print(f"💾 [全状态存档] 进度已安全保存: {save_path}")

if __name__ == "__main__":
    try:
        train_ots()
    except KeyboardInterrupt:
        print("\n🛑 训练被手动停止。不用怕，我们的全状态存档随时可以恢复！")