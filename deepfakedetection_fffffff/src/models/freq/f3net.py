# src/models/freq/f3net.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# --------- 小工具 ---------
def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    # x: (B,3,H,W) -> (B,1,H,W)
    if x.size(1) == 1: return x
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    return 0.2989*r + 0.5870*g + 0.1140*b

def radial_bins(h, w, device, bands: int):
    # 生成归一化半径 [0,1] 的网格，用于圆环分带
    yy, xx = torch.meshgrid(torch.linspace(-1,1,h,device=device), torch.linspace(-1,1,w,device=device), indexing='ij')
    rr = torch.sqrt(xx*xx + yy*yy)    # (H,W), 0 at center
    rr = rr / rr.max().clamp_min(1e-6)
    # 等分频带的边界（bands 段 -> bands+1 个边界）
    edges = torch.linspace(0, 1, bands+1, device=device)
    return rr, edges

# --------- FAD: Frequency-aware Decomposition ---------
class FAD(nn.Module):
    """
    近似论文 FAD：把灰度图做 FFT，按半径等分成 M 个频带，逐带掩蔽后 iFFT 重构，
    得到 (B, M, H, W) 的“频带图”。默认用 FFT；如需 DCT，可替换到 DCT/iDCT。
    """
    def __init__(self, bands: int = 6, transform: str = "fft", learnable_edges: bool = False):
        super().__init__()
        self.bands = bands
        self.transform = transform.lower()
        self.learnable_edges = learnable_edges
        if learnable_edges:
            # 学习频带分割：保证单调递增 -> softplus 累加再归一
            self.delta = nn.Parameter(torch.ones(bands+1))  # 边界数 = bands+1
        else:
            self.register_parameter('delta', None)

    def get_edges(self, device):
        if self.learnable_edges:
            d = F.softplus(self.delta)
            edges = torch.cumsum(d, dim=0)
            edges = edges / edges[-1].clamp_min(1e-6)
            edges = edges.clamp(0,1)
        else:
            edges = None  # 用等分
        return edges

    def forward(self, x: torch.Tensor):
        # x: (B,3,H,W) or (B,1,H,W)
        xg = rgb_to_gray(x)
        B, _, H, W = xg.shape
        device = x.device

        # 2D FFT
        if self.transform == "fft":
            X = torch.fft.fft2(xg, norm='ortho')  # (B,1,H,W), complex
            X = torch.fft.fftshift(X, dim=(-2,-1))
        else:
            raise NotImplementedError("DCT path not wired here; set transform=fft or plug your DCT.")

        rr, edges_eq = radial_bins(H, W, device, self.bands)
        edges = self.get_edges(device) or edges_eq  # (bands+1,)

        comps = []
        for i in range(self.bands):
            lo, hi = edges[i], edges[i+1]
            mask = (rr >= lo) & (rr < hi)  # (H,W)
            mask = mask[None, None, ...]   # (1,1,H,W)
            Xi = X * mask
            # iFFT
            Xi = torch.fft.ifftshift(Xi, dim=(-2,-1))
            xi = torch.fft.ifft2(Xi, norm='ortho').real  # (B,1,H,W)
            comps.append(xi)
        out = torch.cat(comps, dim=1)  # (B,M,H,W)
        # 简单标准化
        out = (out - out.mean(dim=(2,3), keepdim=True)) / (out.std(dim=(2,3), keepdim=True)+1e-6)
        return out

# --------- LFS: Local Frequency Statistics ---------
class LFS(nn.Module):
    """
    近似论文 LFS：灰度图上滑窗（window, stride），每个窗做 FFT，按半径把能量分带求均值，做 log10，
    重新拼成 (B, M, H', W') 的“局部频率统计图”。默认 FFT。
    """
    def __init__(self, bands: int = 6, window: int = 10, stride: int = 2, transform: str = "fft", use_log10: bool = True):
        super().__init__()
        self.bands = bands
        self.window = window
        self.stride = stride
        self.transform = transform.lower()
        self.use_log10 = use_log10

        # 预生成一次窗内的半径与 edges（随设备/HW 变更时再移动）
        rr, edges = radial_bins(window, window, device=torch.device('cpu'), bands=bands)
        self.register_buffer('rr_patch', rr)
        self.register_buffer('edges_patch', edges)

    def forward(self, x: torch.Tensor):
        xg = rgb_to_gray(x)  # (B,1,H,W)
        B, _, H, W = xg.shape

        # unfold 成 (B, 1*win*win, L)；L 是窗口数
        patches = F.unfold(xg, kernel_size=self.window, stride=self.stride)  # (B, win*win, L)
        Lnum = patches.size(-1)
        patches = patches.transpose(1,2).contiguous().view(B*Lnum, 1, self.window, self.window)  # (B*L,1,win,win)

        if self.transform == "fft":
            Freq = torch.fft.fft2(patches, norm='ortho')             # (B*L,1,win,win)
            Freq = torch.fft.fftshift(Freq, dim=(-2,-1))
            amp = Freq.abs()                                         # 幅度
        else:
            raise NotImplementedError("DCT path not wired here; set transform=fft or plug your DCT.")

        rr = self.rr_patch.to(amp.device)
        edges = self.edges_patch.to(amp.device)

        feats = []
        for i in range(self.bands):
            lo, hi = edges[i], edges[i+1]
            mask = ((rr >= lo) & (rr < hi)).float()  # (win,win)
            mask = mask[None, None, ...]            # (1,1,win,win)
            num = mask.sum().clamp_min(1.0)
            stat = (amp * mask).sum(dim=(-2,-1), keepdim=False) / num   # (B*L,1)
            feats.append(stat)
        q = torch.cat(feats, dim=1)  # (B*L, M)

        if self.use_log10:
            q = torch.log10(q + 1e-6)

        # 还原为 (B, M, H', W')
        Hout = (H - self.window)//self.stride + 1
        Wout = (W - self.window)//self.stride + 1
        q = q.view(B, Lnum, self.bands).transpose(1,2).contiguous()      # (B,M,L)
        q = q.view(B, self.bands, Hout, Wout)
        return q

# --------- 轻量 MixBlock：Cross-SE（向量级跨门控） ---------
class CrossSE(nn.Module):
    """
    用向量级 Cross Squeeze-Excitation 做 FAD/LFS 跨门控（论文的 MixBlock 是跨注意力，我们用轻量替代）。
    """
    def __init__(self, dim_fad, dim_lfs, hidden=256):
        super().__init__()
        self.f2l = nn.Sequential(nn.Linear(dim_fad, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dim_lfs), nn.Sigmoid())
        self.l2f = nn.Sequential(nn.Linear(dim_lfs, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dim_fad), nn.Sigmoid())
    def forward(self, f_fad, f_lfs):
        # 输入都是 (B,C)
        g_fad = self.l2f(f_lfs)
        g_lfs = self.f2l(f_fad)
        return f_fad * g_fad, f_lfs * g_lfs

# --------- F3 分支封装：FAD/LFS -> CNN -> CrossSE -> 拼接特征 ---------
class F3Branch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        bands = cfg["bands"]
        tfm = cfg.get("transform", "fft")
        self.mode = cfg.get("mode", "Both")

        # modules
        self.fad = FAD(bands=bands, transform=tfm, learnable_edges=cfg.get("fad",{}).get("learnable_edges", False))
        self.lfs = LFS(bands=bands,
                       window=cfg.get("lfs",{}).get("window",10),
                       stride=cfg.get("lfs",{}).get("stride",2),
                       transform=tfm,
                       use_log10=cfg.get("lfs",{}).get("log10", True))

        # 两个 CNN 子干（注意 in_chans = bands）
        bb_name = cfg.get("backbone", "resnet18")
        pretrained = cfg.get("pretrained", True)
        self.bb_fad = timm.create_model(bb_name, pretrained=pretrained, num_classes=0, global_pool='avg', in_chans=bands)
        self.bb_lfs = timm.create_model(bb_name, pretrained=pretrained, num_classes=0, global_pool='avg', in_chans=bands)

        dim_fad = getattr(self.bb_fad, 'num_features', 512)
        dim_lfs = getattr(self.bb_lfs, 'num_features', 512)
        self.cross = CrossSE(dim_fad, dim_lfs, hidden=min(512, (dim_fad+dim_lfs)//2))

        self.out_dim = dim_fad + dim_lfs

    def forward(self, x):
        outs = []
        if self.mode in ["FAD","Both"]:
            fad_map = self.fad(x)                    # (B,M,H,W)
            f_fad   = self.bb_fad(fad_map)           # (B,Cf)
            outs.append(('fad', f_fad))
        if self.mode in ["LFS","Both"]:
            lfs_map = self.lfs(x)                    # (B,M,h,w)
            # 保守起见，给 CNN 足够分辨率；如 h,w 太小可上采样
            if lfs_map.size(-1) < 64 or lfs_map.size(-2) < 64:
                lfs_map = F.interpolate(lfs_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
            f_lfs   = self.bb_lfs(lfs_map)           # (B,Cl)
            outs.append(('lfs', f_lfs))

        if len(outs) == 2:
            f_fad = outs[0][1]; f_lfs = outs[1][1]
            f_fad_g, f_lfs_g = self.cross(f_fad, f_lfs)
            feat = torch.cat([f_fad_g, f_lfs_g], dim=1)
        else:
            feat = outs[0][1]
        return feat
