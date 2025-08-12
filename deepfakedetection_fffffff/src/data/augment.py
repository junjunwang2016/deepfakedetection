import numpy as np, cv2, torch
from torchvision import transforms

def random_jpeg(rgb, prob, qmin, qmax):
    if np.random.rand() > prob: return rgb
    q = np.random.randint(qmin, qmax+1)
    ok, enc = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok: return rgb
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB) if dec is not None else rgb

def build_transforms(train: bool, img_size: int, aug_cfg):
    tfs = [transforms.ToPILImage()]
    if train:
        if aug_cfg.get("hflip_p", 0) > 0:
            tfs.append(transforms.RandomHorizontalFlip(aug_cfg["hflip_p"]))
        if aug_cfg.get("rot_deg", 0) > 0:
            tfs.append(transforms.RandomRotation(aug_cfg["rot_deg"]))
    tfs += [
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
    tf = transforms.Compose(tfs)

    def apply(rgb):
        if train and aug_cfg.get("jpeg_prob", 0) > 0:
            rgb = random_jpeg(rgb, aug_cfg["jpeg_prob"], aug_cfg["jpeg_qmin"], aug_cfg["jpeg_qmax"])
        return tf(rgb)
    return apply
