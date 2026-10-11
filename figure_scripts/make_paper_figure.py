import os
import cv2
import numpy as np

def find_img_by_keyword(folder_path, keyword):
    """在文件夹中根据关键词（如 15_outdoor）智能寻找图片"""
    if not os.path.exists(folder_path):
        return None
    for f in os.listdir(folder_path):
        if keyword in f and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            return os.path.join(folder_path, f)
    return None

def create_comparison_row(img_paths_or_folders, labels, keyword, crop_box, zoom_factor=3, text_color=(255, 255, 255)):
    canvas_list = []
    base_h, base_w = None, None
    resolved_paths = []

    # 第一遍循环：智能解析所有路径
    for path in img_paths_or_folders:
        if os.path.isdir(path):
            real_path = find_img_by_keyword(path, keyword)
        else:
            real_path = path if os.path.exists(path) else None
        resolved_paths.append(real_path)
        
        if real_path and base_h is None:
            img_temp = cv2.imread(real_path)
            base_h, base_w = img_temp.shape[0], img_temp.shape[1]

    if base_h is None:
        base_h, base_w = 1024, 1024
        
    # 第二遍循环：高质量重组与局部放大
    for i, path in enumerate(resolved_paths):
        if not path:
            print(f"⚠️ 提示: 未能匹配到 {labels[i]} 的图片，已自动生成无缝黑块。")
            img = np.zeros((base_h, base_w, 3), dtype=np.uint8)
        else:
            img = cv2.imread(path)
            if img.shape[0] != base_h or img.shape[1] != base_w:
                img = cv2.resize(img, (base_w, base_h), interpolation=cv2.INTER_CUBIC)
            
        h, w, _ = img.shape
        ymin, xmin, box_h, box_w = int(crop_box[0]*h), int(crop_box[1]*w), int(crop_box[2]*h), int(crop_box[3]*w)
        
        crop_region = img[ymin:ymin+box_h, xmin:xmin+box_w].copy()
        zoom_h, zoom_w = int(box_h * zoom_factor), int(box_w * zoom_factor)
        crop_zoomed = cv2.resize(crop_region, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)
        
        crop_zoomed = cv2.copyMakeBorder(crop_zoomed, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        zh, zw, _ = crop_zoomed.shape
        
        img_with_box = img.copy()
        cv2.rectangle(img_with_box, (xmin, ymin), (xmin+box_w, ymin+box_h), (0, 0, 255), 4) # 稍微加粗红框
        
        # 优雅地嵌入右下角
        img_with_box[h-zh-15:h-15, w-zw-15:w-15] = crop_zoomed
        # 优雅打上论文标签
        cv2.putText(img_with_box, labels[i], (30, int(h*0.06)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3, cv2.LINE_AA)
        
        canvas_list.append(img_with_box)
        
    return np.hstack(canvas_list)

def main():
    # 🎯 核心搜索关键词（不需要写死小尾巴后缀了！）
    KEYWORD = "15_outdoor" 
    
    # 🚨 精准指引各大模型的存放目录或具体文件
    PATHS = [
        f"./CPLIR/Datasets/OHAZE_Test/input",                              # 1. Hazy 输入文件夹
        f"E:\Dehazing_Project\results\O-HAZE_Baseline_epoch120",           # 2. FFA-Net结果文件夹 (若没有可先不管)
        f"./DehazeFormer/DehazeFormer/ohaze_results",                      # 3. DehazeFormer 结果文件夹
        f"./CPLIR/results_cpl_ohaze",                                      # 4. CPLIR 结果文件夹
        r"E:\Dehazing_Project\results\O-HAZE_e2e_epoch120",                # 5. 您的模型结果文件夹 (已完美对齐！👈)
        f"./CPLIR/Datasets/OHAZE_Test/target"                              # 6. Ground Truth 文件夹
    ]
    
    LABELS = ["(a) Hazy Input", "(b) FFA-Net", "(c) DehazeFormer", "(d) CPLIR (SOTA)", "(e) Ours (DAV2)", "(f) Ground Truth"]
    
    # 针对 15_outdoor.png 完美挑选的局部裁剪比例
    CROP_BOX = (0.35, 0.45, 0.15, 0.15) 
    
    print(f"🎬 正在针对图片包含 '{KEYWORD}' 的场景提取局部细节并执行高清重组...")
    result_figure = create_comparison_row(PATHS, LABELS, KEYWORD, CROP_BOX, zoom_factor=2.5)
    
    output_name = f"FINAL_paper_comparison_{KEYWORD}.png"
    cv2.imwrite(output_name, result_figure)
    print(f"🎉 恭喜主人！顶刊级全景视觉对比图已正式诞生：{output_name}")

if __name__ == "__main__":
    main()