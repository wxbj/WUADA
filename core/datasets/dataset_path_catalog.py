import os
from .OtherDataSet import OtherDataSet
from .SourceTrainDataSet import SourceTrainDataSet
import numpy as np
import torch
from tqdm import tqdm
import shutil


class DatasetCatalog(object):
    @staticmethod
    def get(data_root, data_dir, data_list, mode, num_classes, label_info=None, max_iters=None, transform=None,
            cfg=None, is_source=False):
        args = dict(
            root=os.path.join(data_root, data_dir),
            data_list=os.path.join(data_root, data_dir, data_list),
        )
        if mode == 'train' and is_source:
            label_info = os.path.join(data_root, data_dir, label_info)
            return SourceTrainDataSet(args["root"], args["data_list"], label_info, max_iters=max_iters,
                                      num_classes=num_classes, split=mode, transform=transform)
        else:
            return OtherDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                split=mode, transform=transform, cfg=cfg)

    @staticmethod
    def initMask(cfg):

        data_list = os.path.join(cfg.DATASETS.DATASET_ROOT,
                                 cfg.DATASETS.TARGET_TRAIN_DIR,
                                 cfg.DATASETS.TARGET_TRAIN_LIST)

        with open(data_list, "r") as handle:
            content = [line.strip() for line in handle]

        mask_root_dir = os.path.join(cfg.OUTPUT_DIR, "gtMask")
        indicator_root_dir = os.path.join(cfg.OUTPUT_DIR, "gtIndicator")
        if os.path.exists(mask_root_dir):
            shutil.rmtree(mask_root_dir)
        if os.path.exists(indicator_root_dir):
            shutil.rmtree(indicator_root_dir)

        for name in tqdm(content):
            path2image = os.path.join(cfg.DATASETS.DATASET_ROOT,
                                      cfg.DATASETS.TARGET_TRAIN_DIR, "train", "images", name)
            path2mask = os.path.join(
                cfg.OUTPUT_DIR, "gtMask", os.path.normpath(name).split(os.sep)[0],
                f"{os.path.normpath(name).split(os.sep)[1].split('.npy')[0] + '_gtFine_labelIds.npy'}")
            path2indicator = os.path.join(
                cfg.OUTPUT_DIR,
                "gtIndicator", os.path.normpath(name).split(os.sep)[0],
                f"{os.path.normpath(name).split(os.sep)[1].split('.npy')[0] + '_indicator.pth'}")
            mask_dir = os.path.join(cfg.OUTPUT_DIR, "gtMask", os.path.normpath(name).split(os.sep)[0])
            indicator_dir = os.path.join(cfg.OUTPUT_DIR, "gtIndicator", os.path.normpath(name).split(os.sep)[0])

            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(indicator_dir, exist_ok=True)

            arr = np.load(path2image)
            h, w = arr.shape[:2]

            mask = np.ones((h, w), dtype=np.uint8) * 255
            np.save(path2mask, mask)
            indicator = {
                'active': torch.tensor([0], dtype=torch.bool),
                'selected': torch.tensor([0], dtype=torch.bool),
            }
            torch.save(indicator, path2indicator)

