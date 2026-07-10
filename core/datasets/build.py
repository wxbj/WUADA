from . import transform
from .dataset_path_catalog import DatasetCatalog


def build_transform(cfg, mode, is_source):
    if mode == "train":
        w, h = cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN
        if is_source:
            trans_list = [
                transform.ToTensor(),
                transform.Normalize(mean=cfg.INPUT.SRC_PIXEL_MEAN, std=cfg.INPUT.SRC_PIXEL_STD)
            ]
        else:
            trans_list = [
                transform.ToTensor(),
                transform.Normalize(mean=cfg.INPUT.TGT_PIXEL_MEAN, std=cfg.INPUT.TGT_PIXEL_STD)
            ]
        if cfg.INPUT.ELASTIC_DEFORMATION:
            trans_list = [transform.ElasticDeformation()] + trans_list
        if cfg.INPUT.INPUT_SCALES_TRAIN:
            trans_list = [transform.Resize((h, w))] + trans_list
        if cfg.INPUT.HORIZONTAL_FLIP:
            trans_list = [transform.HorizontalFlip()] + trans_list
        if cfg.INPUT.ROTATION:
            trans_list = [transform.Rotation()] + trans_list
        if cfg.INPUT.ROTATION:
            trans_list = [transform.BrightnessContrast()] + trans_list

        trans = transform.Compose(trans_list)
    else:
        w, h = cfg.INPUT.INPUT_SIZE_TEST
        trans = transform.Compose([
            transform.Resize((h, w), resize_label=False),
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.TGT_PIXEL_MEAN, std=cfg.INPUT.TGT_PIXEL_STD)
        ])
    return trans


def build_dataset(cfg, mode='train', is_source=True, epochwise=False):
    assert mode in ['train', 'val', 'test', 'active']
    transforms = build_transform(cfg, mode, is_source)
    iters = None
    if mode == 'train' or mode == 'active':
        if not epochwise:
            iters = cfg.SOLVER.MAX_ITER * cfg.SOLVER.BATCH_SIZE
        if is_source:
            dataset = DatasetCatalog.get(cfg.DATASETS.DATASET_ROOT,
                                         cfg.DATASETS.SOURCE_TRAIN_DIR,
                                         cfg.DATASETS.SOURCE_TRAIN_LIST,
                                         mode,
                                         label_info=cfg.DATASETS.SOURCE_TRAIN_label_info,
                                         num_classes=cfg.MODEL.NUM_CLASSES,
                                         max_iters=iters, transform=transforms, cfg=cfg,
                                         is_source=is_source)
        else:
            dataset = DatasetCatalog.get(cfg.DATASETS.DATASET_ROOT,
                                         cfg.DATASETS.TARGET_TRAIN_DIR,
                                         cfg.DATASETS.TARGET_TRAIN_LIST,
                                         mode,
                                         num_classes=cfg.MODEL.NUM_CLASSES,
                                         max_iters=iters, transform=transforms, cfg=cfg, is_source=is_source)
    elif mode == 'test':
        dataset = DatasetCatalog.get(cfg.DATASETS.DATASET_ROOT,
                                     cfg.DATASETS.TEST_DIR,
                                     cfg.DATASETS.TEST_LIST,
                                     mode,
                                     num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters,
                                     transform=transforms, cfg=cfg, is_source=is_source)
    elif mode == 'val':
        dataset = DatasetCatalog.get(cfg.DATASETS.DATASET_ROOT,
                                     cfg.DATASETS.VAL_DIR,
                                     cfg.DATASETS.VAL_LIST,
                                     mode,
                                     num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters,
                                     transform=transforms, cfg=cfg, is_source=is_source)
    return dataset

