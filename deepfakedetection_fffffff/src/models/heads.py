import torch.nn as nn

class LNLinearHead(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, num_classes))
    def forward(self, x): return self.net(x)
