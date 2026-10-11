import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torchvision.transforms.functional as TF

# 全局免疫一针
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

    def __getitem__(self, idx):
        # 🛡️ 绝对领域：从这里开始，哪怕天塌下来，也会被 try 兜住！
        try:
            # 局部免疫再补一针
            ImageFile.LOAD_TRUNCATED_IMAGES = True 

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

            depth_path = os.path.join(self.depth_dir, f"{name_no_ext}.npy")

            # 💣 危险区域：现在它们已经安全地躺在 try 的肚子里了
            hazy_img = Image.open(hazy_path).convert('RGB')
            clear_img = Image.open(clear_path).convert('RGB')
            
            depth_np = np.load(depth_path).astype(np.float32)
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
            
            hazy_tensor = TF.to_tensor(hazy_img)
            clear_tensor = TF.to_tensor(clear_img)

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
            # 🛡️ 无论报什么错，全部在这里消化掉，然后换下一张！
            print(f"\n🛡️ [坚如磐石] 拦截损坏数据: {self.hazy_files[idx]} | 错误信息: {e}")
            new_idx = random.randint(0, len(self.hazy_files) - 1)
            return self.__getitem__(new_idx)