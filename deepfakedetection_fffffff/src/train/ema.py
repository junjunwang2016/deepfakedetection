try:
    from timm.utils import ModelEmaV2
except Exception:
    ModelEmaV2 = None

def maybe_build_ema(model, enabled, decay):
    if not enabled: return None
    if ModelEmaV2 is None: raise RuntimeError("timm not found")
    return ModelEmaV2(model, decay=decay)
