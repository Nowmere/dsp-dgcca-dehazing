import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def get_professional_font(size):
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
    n = len(img_list)
    single_w = row_width // n
    orig_w, orig_h = img_list[0].size
    single_h = int((single_w / orig_w) * orig_h)
    
    label_area_h = int(single_h * 0.18)
    combined_img = Image.new('RGB', (row_width, single_h + label_area_h), (255, 255, 255))
    draw = ImageDraw.Draw(combined_img)
    
    font_title = get_professional_font(int(label_area_h * 0.35))
    font_metrics = get_professional_font(int(label_area_h * 0.25))

    for i in range(n):
        img = img_list[i].resize((single_w, single_h), Image.LANCZOS)
        combined_img.paste(img, (i * single_w, 0))
        
        text_x = i * single_w + 15
        title_y = single_h + int(label_area_h * 0.15)
        metrics_y = single_h + int(label_area_h * 0.55)
        
        draw.text((text_x, title_y), label_list[i], fill=(0, 0, 0), font=font_title)
        
        if metrics_list[i]:
            m_text = f"PSNR: {metrics_list[i][0]:.2f} / SSIM: {metrics_list[i][1]:.4f}"
            color = (200, 0, 0) if "Ours" in label_list[i] else (80, 80, 80)
            draw.text((text_x, metrics_y), m_text, fill=color, font=font_metrics)

    return combined_img

def find_actual_path(directory, prefix):
    if not os.path.exists(directory):
        return None
    for f in os.listdir(directory):
        # 无视大小写，只要是以 01_ 或 01. 开头统统算数！
        if f.lower().startswith(f"{prefix.lower()}_") or f.lower().startswith(f"{prefix.lower()}."):
            return os.path.join(directory, f)
    return None

def main():
    # ================= ⚙️ 配置区 =================
    # ⚠️ 主人，请重点盯防这两个路径！
    HAZY_DIR = r"E:\Dehazing_Project\data\O-HAZE\hazy"
    GT_DIR = r"E:\Dehazing_Project\data\O-HAZE\GT"
    
    BASELINE_DIR = r"E:\Dehazing_Project\results\O-HAZE_Baseline_epoch120"
    OURS_DIR = r"E:\Dehazing_Project\results\O-HAZE_e2e_epoch120"
    
    SAVE_DIR = "./paper_figures/Top20_ohaze"
    os.makedirs(SAVE_DIR, exist_ok=True)
    # ============================================

    print("======================================================")
    print("🚀 启动全自动流水线：附带神级自我诊断系统")
    print("======================================================")

    # 🌟 强制视野共享：检查 Python 到底看到了什么
    dirs_to_check = {
        "HAZY_DIR (雾图)": HAZY_DIR,
        "GT_DIR (清晰图)": GT_DIR,
        "BASELINE_DIR (Baseline结果)": BASELINE_DIR,
        "OURS_DIR (Ours结果)": OURS_DIR
    }

    print("🔍 正在进行盘前检查...")
    for name, path in dirs_to_check.items():
        if not os.path.exists(path):
            print(f"\n❌ 致命错误：Python 说它找不到文件夹 【{name}】！")
            print(f"   它尝试访问的路径是: {path}")
            print("   👉 建议：请您去文件资源管理器里，右键那个文件夹 -> 属性 -> 复制真正的绝对路径贴到代码里！")
            return
        files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(files) == 0:
            print(f"\n❌ 致命错误：【{name}】 这个文件夹是空的，或者里面没有图片！")
            print(f"   路径: {path}")
            return
        print(f"✅ {name} 正常！看到了 {len(files)} 张图，前两张是: {files[:2]}")
    
    print("-" * 54)

    # 如果走到这里，说明路径绝对没问题！
    result_files = [f for f in os.listdir(OURS_DIR) if f.endswith('.png')]
    records = []

    for file_name in tqdm(result_files, desc="[1/2] 算差值"):
        base_name = file_name.replace("_dehazed.png", "") 
        prefix = base_name.split('_')[0] 
        
        ours_path = os.path.join(OURS_DIR, file_name)
        baseline_path = os.path.join(BASELINE_DIR, file_name)
        
        gt_path = find_actual_path(GT_DIR, prefix)
        hazy_path = find_actual_path(HAZY_DIR, prefix)

        if not (os.path.exists(baseline_path) and gt_path and hazy_path):
            continue

        img_ours = cv2.imread(ours_path)
        img_base = cv2.imread(baseline_path)
        img_gt = cv2.imread(gt_path)

        h, w = img_gt.shape[:2]
        img_ours = cv2.resize(img_ours, (w, h))
        img_base = cv2.resize(img_base, (w, h))

        psnr_b = psnr(img_gt, img_base)
        psnr_o = psnr(img_gt, img_ours)
        
        records.append({
            'name': base_name, 
            'delta_psnr': psnr_o - psnr_b,
            'paths': (hazy_path, gt_path, baseline_path, ours_path)
        })

    if not records:
        print("\n❌ 匹配失败！这不可能！如果盘前检查都过了，这里不该失败的。")
        return

    valid_records = [r for r in records if r['delta_psnr'] < 15.0]
    valid_records.sort(key=lambda x: x['delta_psnr'], reverse=True)
    top_20 = valid_records[:20]

    print(f"\n✨ 寻宝完成！锁定前 {len(top_20)} 张神图。开始排版出图...")
    labels = ["(a) Hazy Input", "(b) Baseline (FFA)", "(c) Ours (DAV2+DG-CCA)", "(d) Ground Truth"]

    for i, r in enumerate(tqdm(top_20, desc="[2/2] 正在生成排版图")):
        name = r['name']
        h_path, g_path, b_path, o_path = r['paths']

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

        final_row = create_comparison_row([img_h, img_b, img_o, img_g], labels, [None, m_b, m_o, None])
        final_row.save(os.path.join(SAVE_DIR, f"Top{i+1:02d}_{name}_Win_{r['delta_psnr']:.2f}dB.png"))

    print("\n======================================================")
    print(f"🎉 任务圆满完成！王牌图片已存入: {SAVE_DIR}")
    print("======================================================")

if __name__ == "__main__":
    main()