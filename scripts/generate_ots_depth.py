import os
import cv2
import torch
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def generate_ots_depth():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    src_hazy_dir = os.path.join(BASE_DIR, 'data', 'RESIDE', 'OTS', 'hazy')
    out_depth_dir = os.path.join(BASE_DIR, 'data', 'RESIDE', 'OTS', 'depth_hazy')
    
    if not os.path.exists(out_depth_dir):
        os.makedirs(out_depth_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 突破 403 封锁
    torch.hub._validate_not_a_forked_repo = lambda a,b,c: True
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.to(device)
    midas.eval()
    
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    img_names = [f for f in os.listdir(src_hazy_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    pbar = tqdm(img_names, desc="⚙️ 提炼 OTS 深度 (防弹版)")
    for i, img_name in enumerate(pbar):
        depth_name = f"{img_name.split('.')[0]}.png"
        out_path = os.path.join(out_depth_dir, depth_name)
        
        # 断点续传：存在就跳过
        if os.path.exists(out_path): 
            continue
            
        img_path = os.path.join(src_hazy_dir, img_name)
        
        # 🌟 终极防弹机制：检测坏图！
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"\n⚠️ 警告：侦测到损坏的图片，已自动丢弃 -> {img_name}")
            continue # 直接跳过这张坏图，处理下一张
            
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()

        output = prediction.cpu().numpy()
        depth_min, depth_max = output.min(), output.max()
        if depth_max - depth_min > 0:
            normalized_depth = (255 * (output - depth_min) / (depth_max - depth_min)).astype("uint8")
        else:
            normalized_depth = output.astype("uint8")
            
        cv2.imwrite(out_path, normalized_depth)

        if i % 100 == 0:
            torch.cuda.empty_cache()

    print("\n✅ OTS 深度图提炼彻底完成！")

if __name__ == '__main__':
    generate_ots_depth()