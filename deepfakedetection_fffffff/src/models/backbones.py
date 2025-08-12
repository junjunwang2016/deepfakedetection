import timm, torch.nn as nn

def create_backbone(name, pretrained=True, pretrained_path=None):
    if pretrained_path:
        return timm.create_model(name, pretrained=False, num_classes=0, global_pool='avg',
                                 checkpoint_path=pretrained_path)
    return timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool='avg')
