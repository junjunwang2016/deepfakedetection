# train.py
import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ---- utils / core ----
from src.utils.ddp import init_distributed, is_main
from src.utils.seed import set_global_seed
from src.data.augment import build_transforms
from src.data.datasets import FrameDataset
from src.models.build import build_model_from_cfg
from src.losses.build import build_loss
from src.optim.build import build_optimizer, build_scheduler
from src.train.engine import train_loop
from src.train.ema import maybe_build_ema


# ---------------------------
# CLI & Config helpers
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--overrides', nargs='*', default=[], help="key=value 形式覆盖配置，支持 a.b.c=val")
    parser.add_argument('--local_rank', type=int, default=-1)  # torchrun 注入
    return parser.parse_args()


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _autotype(v: str):
    vl = v.lower()
    if vl in ('true', 'false'):
        return vl == 'true'
    try:
        if v.strip().startswith('[') and v.strip().endswith(']'):
            # 简易列表解析: 例如 [0.65,0.95]
            body = v.strip()[1:-1]
            parts = [p.strip() for p in body.split(',') if p.strip()!='']
            return [_autotype(p) for p in parts]
        return int(v)
    except:
        try:
            return float(v)
        except:
            return v


def apply_overrides(cfg, overrides):
    for kv in overrides:
        if '=' not in kv: continue
        k, v = kv.split('=', 1)
        node = cfg
        keys = k.split('.')
        for p in keys[:-1]:
            node = node.setdefault(p, {})
        node[keys[-1]] = _autotype(v)
    return cfg


# ---------------------------
# A tiny proxy to mix DDP forward with base-attr access
# ---------------------------
class TrainProxy(torch.nn.Module):
    """
    让引擎在 'model' 上：
      - forward / train() / eval() 走 DDP wrapper（以便梯度同步）
      - 访问属性（如 .backbone）与 state_dict() 走 base model（便于解冻与 EMA）
    """
    def __init__(self, ddp_model: torch.nn.Module, base_model: torch.nn.Module):
        super().__init__()
        self._ddp = ddp_model
        self._base = base_model

    # forward 转给 DDP 包裹的模型
    def forward(self, *args, **kwargs):
        return self._ddp(*args, **kwargs)

    # 训练/评估模式切换给 DDP
    def train(self, mode: bool = True):
        self._ddp.train(mode)
        return self

    def eval(self):
        self._ddp.eval()
        return self

    # EMA/保存等用 base 的 state_dict
    def state_dict(self, *args, **kwargs):
        return self._base.state_dict(*args, **kwargs)

    # 让引擎能访问 .backbone 等属性（落到 base）
    def __getattr__(self, name: str):
        # 优先 Module 自身/父类
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # 其次 ddp 的属性（如某些 forward 需要）
        if hasattr(self._ddp, name):
            return getattr(self._ddp, name)
        # 最后 base 的属性（如 .backbone）
        return getattr(self._base, name)


# ---------------------------
# main
# ---------------------------
def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    # DDP init
    init_distributed(args)

    os.makedirs(cfg["train"]["out_dir"], exist_ok=True)

    # seed（按 rank 偏移）
    rank_offset = args.local_rank if args.local_rank is not None and args.local_rank >= 0 else 0
    set_global_seed(42 + rank_offset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------- data & transforms ----------------
    t_train = build_transforms(True,  cfg["data"]["img_size"], cfg["aug"])
    t_val   = build_transforms(False, cfg["data"]["img_size"], cfg["aug"])

    # Self-Blend & JPEG 配置（训练集用，验证禁用）
    sbi_cfg = {
        "enabled":       cfg["aug"].get("sbi_enabled", False),
        "prob":          cfg["aug"].get("sbi_prob", 0.3),
        "poisson":       cfg["aug"].get("sbi_poisson", False),
        "edge_sigma":    cfg["aug"].get("sbi_edge_sigma", 5),
        "alpha_range":   cfg["aug"].get("sbi_alpha_range", [0.65, 0.95]),
        "color_jitter":  cfg["aug"].get("sbi_color_jitter", 0.08),
        "blur_prob":     cfg["aug"].get("sbi_blur_prob", 0.2),
        "warp": {
            "max_rot":   cfg["aug"].get("sbi_warp_max_rot", 8),
            "max_scale": cfg["aug"].get("sbi_warp_max_scale", 0.10),
            "max_shift": cfg["aug"].get("sbi_warp_max_shift", 0.05),
        },
        "mask_shrink":   cfg["aug"].get("sbi_mask_shrink", 0.85),
    }
    jpeg_cfg = {
        "jpeg_prob": cfg["aug"].get("jpeg_prob", 0.0),
        "jpeg_qmin": cfg["aug"].get("jpeg_qmin", 40),
        "jpeg_qmax": cfg["aug"].get("jpeg_qmax", 90),
    }

    ds_train = FrameDataset(
        cfg["data"]["train_csv"], cfg["data"]["train_root"], cfg["data"]["img_size"], t_train,
        train=True, sbi_cfg=sbi_cfg, jpeg_cfg=jpeg_cfg
    )
    ds_val   = FrameDataset(
        cfg["data"]["val_csv"],   cfg["data"]["val_root"],   cfg["data"]["img_size"], t_val,
        train=False, sbi_cfg=None, jpeg_cfg=None
    )

    # Distributed samplers
    if args.local_rank != -1:
        sampler_train = DistributedSampler(ds_train, shuffle=True,  drop_last=False)
        sampler_val   = DistributedSampler(ds_val,   shuffle=False, drop_last=False)
    else:
        sampler_train = None
        sampler_val   = None

    # DataLoaders
    worker_args = dict(prefetch_factor=4, persistent_workers=True) if cfg["data"]["workers"] > 0 else {}
    dl_train = DataLoader(
        ds_train,
        batch_size=cfg["data"]["batch_size"],
        shuffle=(sampler_train is None),
        num_workers=cfg["data"]["workers"],
        pin_memory=not cfg["train"]["debug"],
        sampler=sampler_train, drop_last=False, **worker_args
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["workers"],
        pin_memory=not cfg["train"]["debug"],
        sampler=sampler_val, drop_last=False, **worker_args
    )

    # ---------------- model ----------------
    # 支持两种：model.type=rgb / rgb_f3（双流，含 F3 频域分支）
    model_base = build_model_from_cfg(cfg, num_classes=cfg["data"]["num_classes"]).to(device)

    # SyncBN（如果用 BN 模型且确实需要；ConvNeXt 主体是 LN，通常不需）
    if args.local_rank != -1 and cfg.get("ddp", {}).get("sync_bn", False):
        model_base = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_base)

    model_ddp = model_base
    if args.local_rank != -1:
        model_ddp = torch.nn.parallel.DistributedDataParallel(
            model_base,
            device_ids=[args.local_rank] if device.type == 'cuda' else None,
            output_device=args.local_rank if device.type == 'cuda' else None,
            broadcast_buffers=False, find_unused_parameters=True
        )

    # 使用 Proxy：forward/模式切换走 DDP，属性/EMA/保存走 base
    model = TrainProxy(model_ddp, model_base)

    # ---------------- loss / optim / sched / ema ----------------
    # 为了 class-balanced alpha，传入训练集 df
    train_df = ds_train.df if hasattr(ds_train, "df") else None
    criterion = build_loss(cfg["loss"], train_df=train_df, device=device)

    # 重要：优化器一开始就把「骨干 + 头」都放进去（Warmup 期间骨干 requires_grad=False，不影响）
    optimizer = build_optimizer(model_base, cfg["optim"], train_backbone=True)
    scheduler = build_scheduler(optimizer, cfg["optim"]["scheduler"])
    ema = maybe_build_ema(model_base, cfg["train"]["ema"]["enabled"], cfg["train"]["ema"]["decay"])

    # ---------------- train ----------------
    loop_cfg = {
        "epochs": cfg["train"]["epochs"],
        "warmup_epochs": cfg["train"]["warmup_epochs"],
        "mixup": cfg["train"]["mixup"],              # dict: enabled/alpha/prob_start/prob_end
        "out_dir": cfg["train"]["out_dir"],
        "max_train_batches": cfg["train"]["max_train_batches"],
    }

    train_loop(
        model, (dl_train, dl_val),
        criterion, optimizer, scheduler,
        loop_cfg, ema=ema, device=device
    )

    # ---------------- done ----------------
    if is_main():
        print("Training finished.")


if __name__ == "__main__":
    main()
