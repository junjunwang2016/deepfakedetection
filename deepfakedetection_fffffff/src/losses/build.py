import torch, numpy as np
from .focal import FocalLoss

def class_balanced_alpha(counts, beta=0.9999):
    counts = np.asarray(counts, dtype=np.float64) + 1e-12
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / eff
    w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)

def build_loss(cfg, train_df=None, device='cuda'):
    name = cfg["name"]
    alpha = None
    if cfg.get("class_balanced_alpha", False) and train_df is not None:
        counts = [train_df['target'].value_counts().get(i,1) for i in range(2)]
        alpha = class_balanced_alpha(counts, beta=cfg.get("cb_beta",0.9999)).to(device)
    if name == "focal":
        return FocalLoss(gamma=cfg.get("gamma",2.0), alpha=alpha)
    elif name == "bce":
        # 二分类 BCE（logits 输入）
        import torch.nn as nn
        return nn.BCEWithLogitsLoss()  # 需要把标签转成 one-hot 再喂
    else:
        raise ValueError(name)
