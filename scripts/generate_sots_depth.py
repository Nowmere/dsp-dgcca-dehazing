import os
import cv2
import torch
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def generate_sots_depth():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 你的 SOTS 文件夹路径 (请确保你把 SOTS 解压成了这种结构)
    src_hazy_dir = os.path.join(BASE_DIR, 'data', 'RESIDE', 'SOTS', 'indoor', 'hazy')
    out_depth_dir = os.path.join(BASE_DIR, 'data', 'RESIDE', 'SOTS', 'indoor', 'depth_hazy')
    
    if not os.path.exists(out_depth_dir):
        os.makedirs(out_depth_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 启动 SOTS 弹药提炼厂 | 计算节点: {device}")

    # 突破 GitHub API 限制
    torch.hub._validate_not_a_forked_repo = lambda a,b,c: True
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.to(device)
    midas.eval()
    
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    img_names = [f for f in os.listdir(src_hazy_dir) if f.endswith(('.png', '.jpg'))]
    print(f"🎯 侦测到 {len(img_names)} 张 SOTS 测试图，开始提炼深度...")
    
    pbar = tqdm(img_names, desc="⚙️ 提炼 SOTS 深度")
    for img_name in pbar:
        # SOTS 的名字往往是 1400_1.png，为了匹配代码我们直接存原名
        out_path = os.path.join(out_depth_dir, img_name)
        if os.path.exists(out_path): continue
            
        img_path = os.path.join(src_hazy_dir, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
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

    print("\n✅ SOTS 深度图提炼完成！")

if __name__ == '__main__':
    generate_sots_depth()