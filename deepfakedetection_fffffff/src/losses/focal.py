import torch, torch.nn.functional as F, torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma, self.reduction = gamma, reduction
        if alpha is not None: self.register_buffer('alpha', alpha)
        else: self.alpha = None
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce   = F.nll_loss(logp, target, reduction='none')
        pt   = logp.exp()[torch.arange(logp.size(0), device=logp.device), target].clamp_min(1e-8)
        loss = (1-pt)**self.gamma * ce
        if self.alpha is not None:
            at = self.alpha.gather(0, target); loss = at*loss
        return loss.mean() if self.reduction=='mean' else loss.sum()
