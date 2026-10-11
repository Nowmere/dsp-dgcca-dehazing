import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 导入您的模型
from models.ffanet import FFA_DGCCA_Net

def test_model():
    # ================= 配置区 =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ⚠️ 1. 填入您刚刚跑完的 100 轮的权重路径
    CHECKPOINT_PATH = "./checkpoints_ots/ffa_dgcca_ots_dav2_epoch_100.pth"
    
    # ⚠️ 2. 填入您的测试集路径 (例如 SOTS Outdoor)
    TEST_HAZY_DIR = "./data/RESIDE/SOTS/outdoor/hazy"
    TEST_CLEAR_DIR = "./data/RESIDE/SOTS/outdoor/clear"
    TEST_DEPTH_DIR = "./data/RESIDE/SOTS/outdoor/depth_dav2"
    
    # ⚠️ 3. 结果保存路径（用于写论文时的可视化对比图）
    RESULTS_DIR = "./results/SOTS_outdoor_epoch100"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # =========================================

    print("======================================================")
    print("🚀 启动终极测试：多模态去雾网络开盲盒")
    print("======================================================")

    # 1. 加载模型与权重
    print("⏳ 正在加载 Epoch 100 的神级权重...")
    model = FFA_DGCCA_Net(gps=3, blocks=3, channels=64).to(device)
    
    # 兼容我们之前写的“全状态存档”
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval() # 极其重要：关闭 Dropout 和 BatchNorm 的训练模式

    # 2. 获取测试集文件列表
    hazy_files = [f for f in os.listdir(TEST_HAZY_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"📦 共找到 {len(hazy_files)} 张测试图片。开始评估 PSNR / SSIM...")

    total_psnr = 0.0
    total_ssim = 0.0
    valid_images = 0

    with torch.no_grad(): # 测试时绝不需要计算梯度，省显存提速
        for hazy_name in tqdm(hazy_files, desc="Testing"):
            name_no_ext = os.path.splitext(hazy_name)[0]
            
            # SOTS 测试集的命名通常是 "xxxx_0.8_0.2.jpg"，对应 GT 是 "xxxx.jpg" 或 "xxxx.png"
            prefix = hazy_name.split('_')[0]
            clear_path = os.path.join(TEST_CLEAR_DIR, f"{prefix}.png")
            if not os.path.exists(clear_path):
                clear_path = os.path.join(TEST_CLEAR_DIR, f"{prefix}.jpg")

            hazy_path = os.path.join(TEST_HAZY_DIR, hazy_name)
            depth_path = os.path.join(TEST_DEPTH_DIR, f"{name_no_ext}.npy")

            # 异常拦截：如果找不到对应的 GT 或 深度图，跳过
            if not os.path.exists(clear_path) or not os.path.exists(depth_path):
                continue

            try:
                # 读取数据 (直接保持原分辨率测试！)
                hazy_img = Image.open(hazy_path).convert('RGB')
                clear_img = Image.open(clear_path).convert('RGB')
                clear_np = np.array(clear_img) # 用于算指标
                
                depth_np = np.load(depth_path).astype(np.float32)
                
                # 转 Tensor 并送入 GPU (Batch Size = 1)
                hazy_tensor = TF.to_tensor(hazy_img).unsqueeze(0).to(device)
                depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(device)
                
                # 🔥 网络推理 🔥
                pred_tensor = model(hazy_tensor, depth_tensor)
                
                # 将输出限制在 0-1 之间，并转回 Numpy 格式用于计算指标
                pred_tensor = torch.clamp(pred_tensor, 0, 1)
                pred_np = pred_tensor.squeeze(0).cpu().numpy()
                pred_np = np.transpose(pred_np, (1, 2, 0)) # CHW -> HWC
                pred_np = (pred_np * 255.0).round().astype(np.uint8)
                
                # 计算指标 (使用 skimage 的标准算法)
                # channel_axis=2 表示图像通道在最后一个维度
                cur_psnr = psnr(clear_np, pred_np)
                cur_ssim = ssim(clear_np, pred_np, channel_axis=2)
                
                total_psnr += cur_psnr
                total_ssim += cur_ssim
                valid_images += 1
                
                # 保存去雾后的图片
                save_img = Image.fromarray(pred_np)
                save_img.save(os.path.join(RESULTS_DIR, f"{name_no_ext}_dehazed.png"))

            except Exception as e:
                print(f"\n⚠️ 处理 {hazy_name} 时发生错误: {e}")

    # 3. 统计最终成绩
    if valid_images > 0:
        avg_psnr = total_psnr / valid_images
        avg_ssim = total_ssim / valid_images
        print("======================================================")
        print("🏆 论文级核心指标出炉 🏆")
        print(f"📊 测试集总数: {valid_images} 张")
        print(f"🌟 平均 PSNR: {avg_psnr:.4f} dB")
        print(f"🌟 平均 SSIM: {avg_ssim:.4f}")
        print(f"💾 去雾结果图已全量保存至: {RESULTS_DIR}")
        print("======================================================")
    else:
        print("❌ 未能成功处理任何图片，请检查路径！")

if __name__ == "__main__":
    test_model()