import os, cv2, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from .self_blend import self_blend
from .augment import random_jpeg


class FrameDataset(Dataset):
    def __init__(self, csv_path, root_dir, img_size, tf, train=True, sbi_cfg: dict=None, jpeg_cfg: dict=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.img_size = img_size
        self.tf = tf
        self.train = train
        self.sbi = sbi_cfg or {"enabled": False}
        self.jpeg_cfg = jpeg_cfg or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        rec = self.df.iloc[i]
        name = str(rec['img_name']); label = int(rec['target'])
        rgb = self._read(name)

        if self.train and self.sbi.get("enabled", False):
            # 建议只对真样本做 SBI，以去身份偏置
            if label == 0 and np.random.rand() < self.sbi.get("prob", 0.3):
                rgb = self_blend(rgb, self.sbi)
                label = 1  # 自混后视为“假”

        # 真实流水线：先伪造/融合 -> 后压缩
        if self.train and self.jpeg_cfg.get("jpeg_prob", 0) > 0:
            rgb = random_jpeg(rgb, self.jpeg_cfg["jpeg_prob"],
                              self.jpeg_cfg.get("jpeg_qmin",40),
                              self.jpeg_cfg.get("jpeg_qmax",90))

        x = self.tf(rgb)
        return x, torch.tensor(label), name

    def _read(self, name):
        path = os.path.join(self.root_dir, name)
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (self.img_size, self.img_size))