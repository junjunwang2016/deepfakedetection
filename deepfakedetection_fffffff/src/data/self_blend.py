import numpy as np, cv2, math, random

def _rand_range(a,b): return a + (b-a)*random.random()

def _ellipse_mask(h, w, shrink=0.85):
    cy, cx = h*0.5, w*0.5
    ry = h*shrink*_rand_range(0.65, 0.95) * 0.5
    rx = w*shrink*_rand_range(0.65, 0.95) * 0.5
    ang = _rand_range(-20, 20)
    Y, X = np.ogrid[:h, :w]
    cosv, sinv = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    xp =  (X-cx)*cosv + (Y-cy)*sinv
    yp = -(X-cx)*sinv + (Y-cy)*cosv
    mask = ((xp*xp)/(rx*rx + 1e-6) + (yp*yp)/(ry*ry + 1e-6)) <= 1.0
    return mask.astype(np.float32)

def _feather(mask, sigma):
    k = int(max(1, sigma*3)|1)
    return cv2.GaussianBlur(mask, (k,k), sigmaX=sigma, sigmaY=sigma)

def _rand_affine(h, w, max_rot=8, max_scale=0.12, max_shift=0.06):
    ang = _rand_range(-max_rot, max_rot)
    sc  = 1.0 + _rand_range(-max_scale, max_scale)
    tx  = _rand_range(-max_shift, max_shift) * w
    ty  = _rand_range(-max_shift, max_shift) * h
    M = cv2.getRotationMatrix2D((w/2,h/2), ang, sc)
    M[:,2] += (tx, ty)
    return M

def self_blend(rgb_uint8: np.ndarray, cfg: dict) -> np.ndarray:
    """
    rgb_uint8: HxWx3, uint8, RGB
    cfg keys:
      prob, poisson(bool), alpha_range(tuple), edge_sigma(int),
      color_jitter(float), blur_prob(float), warp(dict)
    """
    if random.random() > cfg.get("prob", 0.3):
        return rgb_uint8

    h, w = rgb_uint8.shape[:2]
    src = rgb_uint8.astype(np.float32) / 255.0
    tgt = src.copy()

    # 颜色/模糊扰动（让“源脸”有统计差异）
    cj = cfg.get("color_jitter", 0.08)
    gain = 1.0 + np.random.uniform(-cj, cj, size=(1,1,3))
    bias = np.random.uniform(-cj, cj, size=(1,1,3))
    src = np.clip(src * gain + bias, 0, 1)
    if random.random() < cfg.get("blur_prob", 0.2):
        k = random.choice([3,5])
        src = cv2.GaussianBlur(src, (k,k), 0)

    # 几何小变换
    M = _rand_affine(h, w,
                     max_rot=cfg.get("warp",{}).get("max_rot", 8),
                     max_scale=cfg.get("warp",{}).get("max_scale", 0.10),
                     max_shift=cfg.get("warp",{}).get("max_shift", 0.05))
    src = cv2.warpAffine(src, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 随机椭圆 mask + 羽化
    hard = _ellipse_mask(h, w, shrink=cfg.get("mask_shrink", 0.85))
    mask = _feather(hard, sigma=cfg.get("edge_sigma", 5))  # [0,1]

    if cfg.get("poisson", False):
        # Poisson 融合（较慢）
        center = (w//2, h//2)
        # 需要 uint8 & BGR
        src_bgr = (src[:,:,::-1]*255).astype(np.uint8)
        tgt_bgr = (tgt[:,:,::-1]*255).astype(np.uint8)
        mask_u8 = (np.clip(mask,0,1)*255).astype(np.uint8)
        blended = cv2.seamlessClone(src_bgr, tgt_bgr, mask_u8, center, cv2.MIXED_CLONE)
        out = blended[:,:,::-1].astype(np.float32)/255.0
    else:
        # Alpha 羽化融合（快）
        a = mask[...,None]
        out = a*src + (1.0 - a)*tgt

    # 随机调节整体 alpha（让融合强度有变化）
    al = cfg.get("alpha_range", (0.65, 0.95))
    g = _rand_range(al[0], al[1])
    out = g*out + (1.0-g)*tgt

    out = (np.clip(out,0,1)*255.0).astype(np.uint8)
    return out
