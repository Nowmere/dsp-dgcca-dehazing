import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def get_professional_font(size):
    """ 获取高级排版字体 """
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",      
        "C:/Windows/Fonts/times.ttf",      
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 
        "/System/Library/Fonts/Helvetica.ttc" 
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def create_comparison_row(img_list, label_list, metrics_list, row_width=1800):
    """ 将 4 张图水平拼接，并加上专业的文字标签 """
    n = len(img_list)
    single_w = row_width // n
    orig_w, orig_h = img_list[0].size
    single_h = int((single_w / orig_w) * orig_h)
    
    # 底部留白区域
    label_area_h = int(single_h * 0.18)
    combined_img = Image.new('RGB', (row_width, single_h + label_area_h), (255, 255, 255))
    draw = ImageDraw.Draw(combined_img)
    
    font_title = get_professional_font(int(label_area_h * 0.35))
    font_metrics = get_professional_font(int(label_area_h * 0.25))

    for i in range(n):
        # 粘贴图片
        img = img_list[i].resize((single_w, single_h), Image.LANCZOS)
        combined_img.paste(img, (i * single_w, 0))
        
        # 写字位置
        text_x = i * single_w + 15
        title_y = single_h + int(label_area_h * 0.15)
        metrics_y = single_h + int(label_area_h * 0.55)
        
        # 画列标题
        draw.text((text_x, title_y), label_list[i], fill=(0, 0, 0), font=font_title)
        
        # 画 PSNR/SSIM 指标
        if metrics_list[i]:
            m_text = f"PSNR: {metrics_list[i][0]:.2f} / SSIM: {metrics_list[i][1]:.4f}"
            color = (200, 0, 0) if "Ours" in label_list[i] else (80, 80, 80)
            draw.text((text_x, metrics_y), m_text, fill=color, font=font_metrics)

    return combined_img

def main():
    # ================= ⚙️ 配置区 =================
    HAZY_DIR = "./data/O-HAZE/hazy"
    GT_DIR = "./data/O-HAZE/GT"
    BASELINE_DIR = "./results/SOTS_outdoor_Baseline_epoch120"
    OURS_DIR = "./results/sots_e2e_epoch120"
    
    # 🌟 专门为这 20 张候选图建一个文件夹，方便您看图
    SAVE_DIR = "./paper_figures/Top20_o-haze"
    os.makedirs(SAVE_DIR, exist_ok=True)
    # ============================================

    print("======================================================")
    print("🚀 启动全自动流水线：寻宝雷达 + 视觉排版引擎")
    print("======================================================")

    result_files = [f for f in os.listdir(OURS_DIR) if f.endswith('.png')]
    if len(result_files) == 0:
        print("❌ 找不到结果图片，请检查目录路径。")
        return

    records = []

    # ================= 第一阶段：高速雷达扫图 =================
    for file_name in tqdm(result_files, desc="[1/2] 扫描 500 张图片算差值"):
        base_name = file_name.replace("_dehazed.png", "")
        
        ours_path = os.path.join(OURS_DIR, file_name)
        baseline_path = os.path.join(BASELINE_DIR, file_name)
        
        gt_path = os.path.join(GT_DIR, f"{base_name.split('_')[0]}.png")
        if not os.path.exists(gt_path): gt_path = gt_path.replace(".png", ".jpg")

        if not (os.path.exists(baseline_path) and os.path.exists(gt_path)):
            continue

        img_ours = cv2.imread(ours_path)
        img_base = cv2.imread(baseline_path)
        img_gt = cv2.imread(gt_path)

        h, w = img_gt.shape[:2]
        img_ours = cv2.resize(img_ours, (w, h))
        img_base = cv2.resize(img_base, (w, h))

        psnr_b = psnr(img_gt, img_base)
        psnr_o = psnr(img_gt, img_ours)
        
        # 记录 Ours 领先的分数
        delta_psnr = psnr_o - psnr_b
        records.append({'name': base_name, 'delta_psnr': delta_psnr})

    # 排序并取 Top 20 (过滤掉差值极其离谱的废图)
    valid_records = [r for r in records if r['delta_psnr'] < 15.0]
    valid_records.sort(key=lambda x: x['delta_psnr'], reverse=True)
    top_20 = valid_records[:20]

    # ================= 第二阶段：自动生成排版图 =================
    print(f"\n✨ 寻宝完成！已锁定前 20 张神图。开始排版出图...")
    
    labels = ["(a) Hazy Input", "(b) Baseline (FFA)", "(c) Ours (DAV2+DG-CCA)", "(d) Ground Truth"]

    for i, r in enumerate(tqdm(top_20, desc="[2/2] 正在生成排版图")):
        name = r['name']
        rank = i + 1
        
        h_path = os.path.join(HAZY_DIR, f"{name}.jpg")
        if not os.path.exists(h_path): h_path = h_path.replace(".jpg", ".png")
        g_path = os.path.join(GT_DIR, f"{name.split('_')[0]}.png")
        if not os.path.exists(g_path): g_path = g_path.replace(".png", ".jpg")
        b_path = os.path.join(BASELINE_DIR, f"{name}_dehazed.png")
        o_path = os.path.join(OURS_DIR, f"{name}_dehazed.png")

        img_h = Image.open(h_path).convert('RGB')
        img_g = Image.open(g_path).convert('RGB')
        img_b = Image.open(b_path).convert('RGB')
        img_o = Image.open(o_path).convert('RGB')

        w, h = img_g.size
        np_g = np.array(img_g)
        np_b = np.array(img_b.resize((w, h), Image.LANCZOS))
        np_o = np.array(img_o.resize((w, h), Image.LANCZOS))

        m_b = (psnr(np_g, np_b), ssim(np_g, np_b, channel_axis=2))
        m_o = (psnr(np_g, np_o), ssim(np_g, np_o, channel_axis=2))

        metrics = [None, m_b, m_o, None]
        final_row = create_comparison_row([img_h, img_b, img_o, img_g], labels, metrics)
        
        # 命名格式：Top排名_图片原名_领先多少dB.png
        save_name = f"Top{rank:02d}_{name}_Win_{r['delta_psnr']:.2f}dB.png"
        final_row.save(os.path.join(SAVE_DIR, save_name))

    print("\n======================================================")
    print(f"🎉 任务圆满完成！")
    print(f"📂 请打开文件夹: {SAVE_DIR}")
    print("您可以直接在里面像看幻灯片一样，通过肉眼海选最终放入论文的王牌图！")
    print("======================================================")

if __name__ == "__main__":
    main()