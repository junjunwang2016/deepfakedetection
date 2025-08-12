import torch, numpy as np, torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from tqdm.auto import tqdm
from ..utils.ddp import is_main, all_reduce_sum, broadcast_scalar, all_gather_list
from .thresholds import find_best_threshold

def one_hot(y, num_classes): return F.one_hot(y, num_classes=num_classes).float()

def mixup_batch(x, y, alpha, num_classes):
    if alpha <= 0: return x, one_hot(y, num_classes), 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = lam*x + (1.0-lam)*x[idx]
    y1 = one_hot(y, num_classes); y2 = y1[idx]
    y_mix = lam*y1 + (1.0-lam)*y2
    return x2, y_mix, lam

@torch.no_grad()
def validate_ddp(eval_model, loader, criterion, amp, device):
    eval_model.eval()
    y_true_local, y_prob_local = [], []
    loss_sum_local, correct_local, total_local = 0.0, 0, 0
    pbar = tqdm(loader, desc="[Valid]", leave=False, disable=not is_main())
    for xb, yb, _ in pbar:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type=='cuda'):
            logits = eval_model(xb)
            loss = criterion(logits, yb)
        prob = F.softmax(logits, dim=1)[:,1]
        pred = logits.argmax(dim=1)
        bs = yb.size(0)
        loss_sum_local += loss.item()*bs; total_local += bs; correct_local += (pred==yb).sum().item()
        y_true_local.extend(yb.detach().cpu().tolist()); y_prob_local.extend(prob.detach().cpu().tolist())
        if is_main():
            pbar.set_postfix({"loss": f"{loss_sum_local/max(1,total_local):.4f}", "acc": f"{100.0*correct_local/max(1,total_local):.2f}%"})

    loss_sum = torch.tensor([loss_sum_local], device=device)
    correct  = torch.tensor([correct_local],  device=device)
    total    = torch.tensor([total_local],    device=device)
    all_reduce_sum(loss_sum); all_reduce_sum(correct); all_reduce_sum(total)
    val_loss = float(loss_sum.item())/max(1,int(total.item()))
    val_acc  = 100.0*float(correct.item())/max(1,int(total.item()))

    # gather predictions
    y_true_all, y_prob_all = [], []
    for lst in all_gather_list(y_true_local): y_true_all += lst
    for lst in all_gather_list(y_prob_local): y_prob_all += lst

    auc, cm050, rep050, best_f1, best_ba = float('nan'), None, "", (0.5,0.), (0.5,0.)
    if is_main():
        try: auc = roc_auc_score(y_true_all, y_prob_all) if len(set(y_true_all))>1 else float('nan')
        except Exception: auc = float('nan')
        best_f1 = find_best_threshold(y_true_all, y_prob_all, 'f1')
        best_ba = find_best_threshold(y_true_all, y_prob_all, 'ba')
        th = 0.50
        y_hat = (np.array(y_prob_all)>=th).astype(int)
        cm050 = confusion_matrix(y_true_all, y_hat, labels=[0,1])
        rep050 = classification_report(y_true_all, y_hat, digits=4, zero_division=0)

    auc = broadcast_scalar(auc, device)
    t_f1 = broadcast_scalar(best_f1[0], device); s_f1 = broadcast_scalar(best_f1[1], device)
    t_ba = broadcast_scalar(best_ba[0], device); s_ba = broadcast_scalar(best_ba[1], device)

    return val_loss, val_acc, auc, (t_f1,s_f1), (t_ba,s_ba), cm050, rep050, (np.array(y_true_all) if is_main() else None), (np.array(y_prob_all) if is_main() else None)

def train_loop(model, loaders, criterion, optimizer, scheduler, cfg, ema=None, device=None):
    train_loader, val_loader = loaders
    amp = (device.type=='cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_auc = -1.0

    def set_backbone_trainable(flag):
        for p in model.backbone.parameters():
            p.requires_grad = flag

    # Warmup 冻结骨干
    set_backbone_trainable(False)

    for ep in range(cfg["epochs"]):
        # 解冻点
        if ep == cfg["warmup_epochs"]:
            set_backbone_trainable(True)

        # DDP sampler 设置 epoch
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(ep)

        # 动态 MixUp 概率
        t = ep / max(1, cfg["epochs"]-1)
        mix_prob = cfg["mixup"]["prob_start"]*(1-t) + cfg["mixup"]["prob_end"]*t if cfg["mixup"]["enabled"] else 0.0

        model.train()
        loss_sum_l, corr_l, tot_l = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"[Train] Ep{ep+1}", leave=False, disable=not is_main())
        for it,(xb,yb,_) in enumerate(pbar):
            if cfg.get("max_train_batches") and it>=cfg["max_train_batches"]: break
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            use_mix = (np.random.rand() < mix_prob)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                if use_mix:
                    xmix, ysoft, _ = mixup_batch(xb, yb, cfg["mixup"]["alpha"], num_classes=2)
                    logits = model(xmix)
                    loss = -(ysoft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None: ema.update(model)

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                bs = yb.size(0)
                loss_sum_l += loss.item()*bs; tot_l += bs; corr_l += (pred==yb).sum().item()
            if is_main():
                pbar.set_postfix({"loss": f"{loss_sum_l/max(1,tot_l):.4f}",
                                  "acc": f"{100.0*corr_l/max(1,tot_l):.2f}%",
                                  "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        # 验证（EMA 优先）
        eval_model = ema.module if ema is not None else model
        val_loss, val_acc, auc, best_f1, best_ba, cm050, rep050, y_true, y_prob = \
            validate_ddp(eval_model, val_loader, criterion, amp, device)

        scheduler.step(auc)

        if is_main():
            print(f"\n[Ep {ep+1}/{cfg['epochs']}] "
                  f"train_loss={loss_sum_l/max(1,tot_l):.4f}, train_acc={100.0*corr_l/max(1,tot_l):.2f}% | "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, AUC={auc:.4f} | "
                  f"bestF1@{best_f1[0]:.2f}={best_f1[1]:.4f}, bestBA@{best_ba[0]:.2f}={best_ba[1]:.4f}")
            if cm050 is not None:
                print("CM@0.50:\n", cm050)
                print("Report@0.50:\n", rep050)

        # 保存（仅主进程）
        if is_main() and auc > best_auc:
            best_auc = auc
            to_save = ema.module if ema is not None else model
            torch.save(to_save.state_dict(), f"{cfg['out_dir']}/rgb_best.pth")
            if y_true is not None:
                np.savez_compressed(f"{cfg['out_dir']}/oof_rgb.npz", y=y_true, prob=y_prob)
            with open(f"{cfg['out_dir']}/rgb_best_threshold.txt","w") as f:
                f.write(f"{best_ba[0]:.6f}")
            print(f">> Saved best: AUC={best_auc:.4f}")
