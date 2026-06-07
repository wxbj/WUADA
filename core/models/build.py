from .feature_extractor import resnet_feature_extractor
from .classifier import UNet3PlusDecoder


def build_feature_extractor(cfg):
    model_name, backbone_name = cfg.MODEL.NAME.split('_')
    backbone = resnet_feature_extractor(backbone_name, pretrained_backbone=True)
    return backbone


def build_classifier(cfg):
    classifier = UNet3PlusDecoder(num_classes=cfg.MODEL.NUM_CLASSES, encoder_channels=[64, 256, 512, 1024, 2048])
    return classifier
