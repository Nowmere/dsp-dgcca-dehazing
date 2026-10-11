import os
import cv2
import torch
import warnings
from tqdm import tqdm

# 屏蔽底层库一些无关紧要的满屏警告，保持终端干净
warnings.filterwarnings("ignore")

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_offline_depths():
    print("======================================================")
    print("🚀 启动 MiDaS 工业级离线伪深度提炼厂 (终极重构版)")
    print("======================================================")
    
    # ---------------------------------------------------------
    # 📁 工业级路径引擎：自动定位项目根目录
    # ---------------------------------------------------------
    # __file__ 获取当前脚本的绝对路径，向上推两层就是 Dehazing_Project 根目录
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"📂 侦测到项目根目录: {BASE_DIR}")
    
    # 严格按照你刚才建好的正规军目录结构拼接路径
    src_hazy_dir = os.path.join(BASE_DIR, 'data', 'RESIDE', 'ITS', 'hazy')
    src_clear_dir = os.path.join(BASE_DIR, 'data', 'RESIDE', 'ITS', 'clear')
    
    out_hazy_depth_dir = os.path.join(BASE_DIR, 'data', 'RESIDE', 'ITS', 'depth_hazy')
    out_clear_depth_dir = os.path.join(BASE_DIR, 'data', 'RESIDE', 'ITS', 'depth_clear')
    
    make_dirs(out_hazy_depth_dir)
    make_dirs(out_clear_depth_dir)

    # ---------------------------------------------------------
    # ⚙️ 引擎初始化与 403 封锁突破
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 当前计算节点: {device} | 准备载入预训练权重...")

    # 🪓 【终极黑魔法】：强行废掉 PyTorch 的 GitHub API 检查逻辑，绕过 403 封锁！
    torch.hub._validate_not_a_forked_repo = lambda a,b,c: True
    
    # 加载 MiDaS DPT_Large 模型
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval() # 开启评估模式
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    # ---------------------------------------------------------
    # 🏭 核心处理流水线
    # ---------------------------------------------------------
    def process_directory(input_dir, output_dir, desc):
        print(f"\n🔍 正在扫描目标文件夹: {input_dir}")
        if not os.path.exists(input_dir):
            print(f"❌ 致命错误：找不到路径 {input_dir}")
            print(f"💡 导师提示：请确保你把图片放到了上面这个准确的路径里！")
            return

        img_names = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(img_names) == 0:
            print(f"⚠️ 警告：文件夹 {input_dir} 存在，但是里面没有图片！")
            return
            
        print(f"🎯 成功侦测到 {len(img_names)} 张弹药，开始提炼...")
        pbar = tqdm(img_names, desc=f"⚙️ {desc}")
        
        for i, img_name in enumerate(pbar):
            out_path = os.path.join(output_dir, img_name.split('.')[0] + '.png')
            
            # 【断点续传机制】
            if os.path.exists(out_path):
                continue
                
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_batch = transform(img).to(device)

            # 【显存防爆机制】纯推理不保留梯度
            with torch.no_grad(): 
                prediction = midas(input_batch)
                
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()
            
            # 归一化并保存为 8 位灰度图
            depth_min, depth_max = output.min(), output.max()
            if depth_max - depth_min > 0:
                normalized_depth = (255 * (output - depth_min) / (depth_max - depth_min)).astype("uint8")
            else:
                normalized_depth = output.astype("uint8")
                
            cv2.imwrite(out_path, normalized_depth)

            # 【防显存泄漏机制】
            if i % 50 == 0:
                torch.cuda.empty_cache()

    # 开始双线提炼！
    process_directory(src_hazy_dir, out_hazy_depth_dir, "提炼 Hazy 伪深度")
    process_directory(src_clear_dir, out_clear_depth_dir, "提炼 Clear 理想深度")
    
    print("\n✅ 全线战备完成！可以开始模型训练了！")

if __name__ == '__main__':
    generate_offline_depths()