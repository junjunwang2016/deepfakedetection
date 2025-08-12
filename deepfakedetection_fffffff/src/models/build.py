# src/models/build.py （在文件末尾追加）
import torch.nn as nn
from .backbones import create_backbone
from .heads import LNLinearHead
from .freq.f3net import F3Branch
import torch

class RGBF3Classifier(nn.Module):
    """
    空间（RGB） + 频域（F3Net 分支）双流
    """
    def __init__(self, cfg_model, num_classes):
        super().__init__()
        # 空间分支
        self.rgb_backbone = create_backbone(cfg_model["backbone"], cfg_model.get("pretrained", True), cfg_model.get("pretrained_path", None))
        self.rgb_dim = getattr(self.rgb_backbone, 'num_features', 1024)

        # 频域分支
        assert cfg_model.get("freq", {}).get("enabled", True), "freq.enabled must be True for RGBF3Classifier"
        self.f3 = F3Branch(cfg_model["freq"])
        self.freq_dim = self.f3.out_dim

        # 融合头
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.rgb_dim + self.freq_dim),
            nn.Linear(self.rgb_dim + self.freq_dim, num_classes)
        )

    def forward(self, x):
        f_rgb = self.rgb_backbone(x)     # (B,Cr)
        f_f3  = self.f3(x)               # (B,Cf)
        feat  = torch.cat([f_rgb, f_f3], dim=1)
        return self.fuse(feat)
    # === 关键补丁：给优化器用的“别名” ===
    @property
    def backbone(self):
        # 返回一个临时 ModuleList，把 RGB backbone + 频域分支 一起当“骨干”
        return nn.ModuleList([self.rgb_backbone, self.f3])

    @property
    def head(self):
        # 让优化器还能通过 .head 拿到分类头参数
        return self.fuse

class RGBClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = create_backbone(cfg["backbone"], cfg["pretrained"], cfg.get("pretrained_path"))
        feat_dim = getattr(self.backbone, 'num_features', 1024)
        if cfg.get("head","ln_linear") == "ln_linear":
            self.head = LNLinearHead(feat_dim, cfg["num_classes"])
        else:
            raise ValueError("Unknown head")
    def forward(self, x):
        f = self.backbone(x)
        return self.head(f)


def build_model_from_cfg(cfg, num_classes):
    mtype = cfg["model"].get("type", "rgb")
    if mtype == "rgb":
        return RGBClassifier({
            "backbone": cfg["model"]["backbone"],
            "pretrained": cfg["model"]["pretrained"],
            "pretrained_path": cfg["model"]["pretrained_path"],
            "num_classes": num_classes,
            "head": cfg["model"]["head"],
        })
    elif mtype == "rgb_f3":
        return RGBF3Classifier(cfg["model"], num_classes)
    else:
        raise ValueError(mtype)
