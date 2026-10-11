import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def get_professional_font(size):
    """ 尝试获取系统内置的高级字体 """
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",      # Windows
        "C:/Windows/Fonts/times.ttf",      # Windows Serif
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", # Linux
        "/System/Library/Fonts/Helvetica.ttc" # MacOS
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def create_comparison_row(img_list, label_list, metrics_list, row_width=1600):
    """
    将一组图片拼接成一行，并带有专业标注
    """
    n = len(img_list)
    single_w = row_width // n
    # 根据第一张图比例计算高度
    orig_w, orig_h = img_list[0].size
    single_h = int((single_w / orig_w) * orig_h)
    
    # 为底部文字留出空间 (增加 15% 的高度)
    label_area_h = int(single_h * 0.18)
    combined_img = Image.new('RGB', (row_width, single_h + label_area_h), (255, 255, 255))
    draw = ImageDraw.Draw(combined_img)
    
    font_title = get_professional_font(int(label_area_h * 0.35))
    font_metrics = get_professional_font(int(label_area_h * 0.25))

    for i in range(n):
        # 1. 调整尺寸并粘贴
        img = img_list[i].resize((single_w, single_h), Image.LANCZOS)
        combined_img.paste(img, (i * single_w, 0))
        
        # 2. 绘制下方文字区
        text_x = i * single_w + 10
        title_y = single_h + int(label_area_h * 0.15)
        metrics_y = single_h + int(label_area_h * 0.55)
        
        # 绘制标题 (如 Hazy, Baseline 等)
        draw.text((text_x, title_y), label_list[i], fill=(0, 0, 0), font=font_title)
        
        # 绘制指标 (PSNR/SSIM)
        if metrics_list[i]:
            m_text = f"PSNR: {metrics_list[i][0]:.2f} / SSIM: {metrics_list[i][1]:.4f}"
            # 如果是 Ours，加个加粗提醒或特殊标记
            color = (200, 0, 0) if "Ours" in label_list[i] else (80, 80, 80)
            draw.text((text_x, metrics_y), m_text, fill=color, font=font_metrics)

    return combined_img

def main():
    # ================= ⚙️ 配置区 (请根据实际情况修改) =================
    # 挑选 3-4 张最有代表性的图片名字 (不需要后缀)
    # 建议从 SOTS-Indoor 或 O-HAZE 中挑选
    IMAGE_NAMES = ["01_outdoor", "05_outdoor", "11_outdoor"] 
    
    # 文件夹路径
    HAZY_DIR = "./data/RESIDE/O-HAZE/hazy"
    GT_DIR = "./data/RESIDE/O-HAZE/GT"
    BASELINE_DIR = "./results/SOTS_outdoor_Baseline_epoch120"
    OURS_DIR = "./results/sots_e2e_epoch120"
    
    SAVE_DIR = "./paper_figures"
    os.makedirs(SAVE_DIR, exist_ok=True)
    # ===============================================================

    print("🚀 正在启动排版引擎，准备生成视觉对比图...")

    for name in IMAGE_NAMES:
        try:
            # 1. 读取所有相关的图
            h_path = os.path.join(HAZY_DIR, f"{name}.jpg")
            if not os.path.exists(h_path): h_path = h_path.replace(".jpg", ".png")
            
            g_path = os.path.join(GT_DIR, f"{name}_GT.jpg") # 适配 O-HAZE 命名
            if not os.path.exists(g_path): g_path = os.path.join(GT_DIR, f"{name.split('_')[0]}.png")
            
            b_path = os.path.join(BASELINE_DIR, f"{name}_dehazed.png")
            o_path = os.path.join(OURS_DIR, f"{name}_dehazed.png")

            img_h = Image.open(h_path).convert('RGB')
            img_g = Image.open(g_path).convert('RGB')
            img_b = Image.open(b_path).convert('RGB')
            img_o = Image.open(o_path).convert('RGB')

            # 2. 计算指标 (现场算，保证绝对准确)
            w, h = img_g.size
            # 对齐尺寸
            np_g = np.array(img_g)
            np_b = np.array(img_b.resize((w, h), Image.LANCZOS))
            np_o = np.array(img_o.resize((w, h), Image.LANCZOS))

            m_b = (psnr(np_g, np_b), ssim(np_g, np_b, channel_axis=2))
            m_o = (psnr(np_g, np_o), ssim(np_g, np_o, channel_axis=2))

            # 3. 拼接
            imgs = [img_h, img_b, img_o, img_g]
            labels = ["(a) Hazy Input", "(b) Baseline (FFA)", "(c) Ours (DAV2+DG-CCA)", "(d) Ground Truth"]
            metrics = [None, m_b, m_o, None]

            final_row = create_comparison_row(imgs, labels, metrics)
            
            save_path = os.path.join(SAVE_DIR, f"compare_{name}.png")
            final_row.save(save_path)
            print(f"✅ 已生成对比图: {save_path}")

        except Exception as e:
            print(f"⚠️ 处理 {name} 时出错: {e}")

    print(f"\n🎉 所有对比图已存入 {SAVE_DIR}。快去看看视觉震撼效果吧！")

if __name__ == "__main__":
    main()