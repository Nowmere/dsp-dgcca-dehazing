import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2  # 🌟 祭出工业界老大哥 OpenCV
from PIL import Image
import torchvision.transforms.functional as TF

class OTSDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, depth_dir, crop_size=256, is_train=True):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.depth_dir = depth_dir
        self.crop_size = crop_size
        self.is_train = is_train
        self.hazy_files = [f for f in os.listdir(hazy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.hazy_files)

    def safe_read_image(self, path):
        # 🌟 暴力美学：用 OpenCV 强行读取，无论截断还是损坏
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise ValueError(f"OpenCV 彻底无法读取该文件: {path}")
        # OpenCV 默认是 BGR，转换为 PyTorch/PIL 习惯的 RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def __getitem__(self, idx):
        try:
            hazy_name = self.hazy_files[idx]
            name_no_ext = os.path.splitext(hazy_name)[0]
            hazy_path = os.path.join(self.hazy_dir, hazy_name)
            
            prefix = hazy_name.split('_')[0]
            clear_path_jpg = os.path.join(self.clear_dir, f"{prefix}.jpg")
            clear_path_png = os.path.join(self.clear_dir, f"{prefix}.png")
            
            if os.path.exists(clear_path_jpg):
                clear_path = clear_path_jpg
            elif os.path.exists(clear_path_png):
                clear_path = clear_path_png
            else:
                raise FileNotFoundError("找不到GT图")

            # 💣 使用我们的暴力引擎读取图片
            hazy_img = self.safe_read_image(hazy_path)
            clear_img = self.safe_read_image(clear_path)
            
            # ================= 🌟 Slaver 的防递归深度加载逻辑 🌟 =================
            if self.depth_dir is not None:
                # Ours 模式：正常读取深度图
                depth_path_pt = os.path.join(self.depth_dir, f"{name_no_ext}.pt")
                depth_path_npy = os.path.join(self.depth_dir, f"{name_no_ext}.npy")
                if os.path.exists(depth_path_pt):
                    depth_data = torch.load(depth_path_pt, map_location='cpu', weights_only=True)
                    if isinstance(depth_data, torch.Tensor):
                        depth_tensor = depth_data
                        if depth_tensor.dim() == 2:
                            depth_tensor = depth_tensor.unsqueeze(0)
                    else:
                        depth_np = np.array(depth_data, dtype=np.float32)
                        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
                elif os.path.exists(depth_path_npy):
                    depth_np = np.load(depth_path_npy).astype(np.float32)
                    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
                else:
                    # 深度文件缺失，创建假 depth
                    w, h = clear_img.size
                    depth_tensor = torch.zeros((1, h, w))
                if depth_tensor.numel() == 0:
                    raise ValueError("深度图为空")
            else:
                # Baseline 模式：因为 depth_dir 是 None，不去拼路径，直接造一个假 Tensor 占位！
                # 取出清晰图的宽和高，造一个尺寸完全一样、值全为 0 的深度图
                w, h = clear_img.size
                depth_tensor = torch.zeros((1, h, w))
            # =====================================================================
            
            # 转换为 Tensor
            hazy_tensor = TF.to_tensor(hazy_img)
            clear_tensor = TF.to_tensor(clear_img)

            # 数据增强逻辑
            if self.is_train:
                _, h, w = hazy_tensor.shape
                if h > self.crop_size and w > self.crop_size:
                    top = random.randint(0, h - self.crop_size)
                    left = random.randint(0, w - self.crop_size)
                    hazy_tensor = TF.crop(hazy_tensor, top, left, self.crop_size, self.crop_size)
                    clear_tensor = TF.crop(clear_tensor, top, left, self.crop_size, self.crop_size)
                    depth_tensor = TF.crop(depth_tensor, top, left, self.crop_size, self.crop_size)
                else:
                    hazy_tensor = TF.resize(hazy_tensor, (self.crop_size, self.crop_size), antialias=True)
                    clear_tensor = TF.resize(clear_tensor, (self.crop_size, self.crop_size), antialias=True)
                    depth_tensor = TF.resize(depth_tensor, (self.crop_size, self.crop_size), antialias=True)

                if random.random() > 0.5:
                    hazy_tensor = TF.hflip(hazy_tensor)
                    clear_tensor = TF.hflip(clear_tensor)
                    depth_tensor = TF.hflip(depth_tensor)

            return hazy_tensor, clear_tensor, depth_tensor

        except Exception as e:
            # 🛡️ 防止无限递归：限制重试 10 次
            if not hasattr(self, '_retry_count'):
                self._retry_count = 0
            self._retry_count += 1
            if self._retry_count > 10:
                self._retry_count = 0
                raise RuntimeError(f"Dataset: too many consecutive errors, last: {e}") from e
            new_idx = random.randint(0, len(self.hazy_files) - 1)
            return self.__getitem__(new_idx)
