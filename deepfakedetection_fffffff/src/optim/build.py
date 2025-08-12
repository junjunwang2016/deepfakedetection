import torch.optim as optim

def build_optimizer(model, cfg, train_backbone: bool):
    params = []
    if train_backbone:
        params.append({"params": model.backbone.parameters(), "lr": cfg["lr_backbone"], "weight_decay": cfg["weight_decay"]})
    params.append({"params": model.head.parameters(), "lr": cfg["lr_head"], "weight_decay": cfg["weight_decay"]})
    return optim.AdamW(params)

def build_scheduler(optimizer, name="reduce_on_plateau"):
    if name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=False)
    else:
        raise ValueError(name)
