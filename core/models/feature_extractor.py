from torch import nn
from . import resnet


class resnet_feature_extractor(nn.Module):
    def __init__(self, backbone_name, pretrained_backbone=True):
        super(resnet_feature_extractor, self).__init__()
        self.backbone = resnet.__dict__[backbone_name](pretrained=pretrained_backbone)

    def forward(self, x):
        return self.backbone(x)

