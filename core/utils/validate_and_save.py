import re

from medpy.metric.binary import assd, dc
from tqdm import tqdm

import torch.utils
import torch.distributed

from core.utils.misc import inference

import warnings

warnings.filterwarnings('ignore')

import os
import torch
import numpy as np

_num = re.compile(r'(\d+)')


def z_from_name(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    return int(_num.findall(stem)[-1])


def _finalize_case(case_slices, num_classes):
    dice_total = np.zeros(num_classes, dtype=np.float64)
    dice_count = np.zeros(num_classes, dtype=np.int64)
    assd_total = np.zeros(num_classes, dtype=np.float64)
    assd_count = np.zeros(num_classes, dtype=np.int64)

    if not case_slices:
        return dice_total, dice_count, assd_total, assd_count

    case_slices.sort(key=lambda t: t[0])
    pred_vol = np.stack([p for (_, p, __) in case_slices], axis=0)  # [D,H,W]
    gt_vol = np.stack([g for (_, __, g) in case_slices], axis=0)  # [D,H,W]

    for c in range(1, num_classes):
        pred_c = (pred_vol == c).astype(np.uint8)
        gt_c = (gt_vol == c).astype(np.uint8)

        dice_total[c] += float(dc(pred_c, gt_c))
        dice_count[c] += 1

        try:
            a = assd(pred_c, gt_c)
        except Exception:
            a = 100.0
        assd_total[c] += float(a)
        assd_count[c] += 1

    return dice_total, dice_count, assd_total, assd_count


@torch.no_grad()
def validate_and_save(cfg, iteration, feature_extractor, classifier,
                      optimizer_fea, optimizer_cls, tgt_val_loader, logger, tgt_val_data):
    feature_extractor.eval()
    classifier.eval()

    if cfg.DATASETS.SOURCE_TRAIN_DIR != "bSSFP":
        trainid2name = {0: 'bg', 1: "MYO", 2: "LAC", 3: "LVC", 4: "AA"}
    else:
        trainid2name = {0: 'bg', 1: "MYO", 2: "LVC", 3: "RVC"}

    dice_per_class_total = np.zeros(cfg.MODEL.NUM_CLASSES, dtype=np.float64)
    dice_per_class_count = np.zeros(cfg.MODEL.NUM_CLASSES, dtype=np.int64)
    assd_per_class_total = np.zeros(cfg.MODEL.NUM_CLASSES, dtype=np.float64)
    assd_per_class_count = np.zeros(cfg.MODEL.NUM_CLASSES, dtype=np.int64)

    torch.cuda.empty_cache()
    logger.info("Evaluating on validation set (3D metrics from 2D slices)...")

    current_path = None
    case_slices = []

    for val_data in tqdm(tgt_val_loader):
        val_input = val_data['img'].cuda(non_blocking=True)
        val_label = val_data['label'].cuda(non_blocking=True).long()
        name = val_data['name'][0]
        path = val_data['path'][0]

        if current_path is not None and path != current_path:
            dT, dC, aT, aC = _finalize_case(case_slices, cfg.MODEL.NUM_CLASSES)
            dice_per_class_total += dT
            dice_per_class_count += dC
            assd_per_class_total += aT
            assd_per_class_count += aC
            case_slices = []

        current_path = path

        logits = inference(feature_extractor, classifier, val_input, val_label, True)  # [1,C,H,W]
        pred_2d = logits.detach().cpu().numpy().squeeze().argmax(0).astype(np.uint8)  # [H,W]

        gt_2d = val_label.cpu().numpy().squeeze().astype(np.uint8)  # [H,W]
        ignore = int(cfg.INPUT.IGNORE_LABEL)
        valid = (gt_2d != ignore)
        if not valid.all():
            pred_2d = pred_2d.copy()
            gt_2d = gt_2d.copy()
            pred_2d[~valid] = 0
            gt_2d[~valid] = 0

        z = z_from_name(name)
        case_slices.append((z, pred_2d, gt_2d))

    if case_slices:
        dT, dC, aT, aC = _finalize_case(case_slices, cfg.MODEL.NUM_CLASSES)
        dice_per_class_total += dT
        dice_per_class_count += dC
        assd_per_class_total += aT
        assd_per_class_count += aC

    dice_per_class_avg = dice_per_class_total / dice_per_class_count
    mDice = dice_per_class_avg[1:].mean()

    assd_per_class_avg = assd_per_class_total / assd_per_class_count
    mean_assd = assd_per_class_avg[1:].mean()

    logger.info('Val metrics (3D): mDice {:.4f}'.format(mDice))
    logger.info('Val metrics (3D): mean ASSD {:.4f}'.format(mean_assd))
    for i in range(1, cfg.MODEL.NUM_CLASSES):
        logger.info('{} {} Dice: {:.4f} ASSD: {:.4f}'.format(
            i, trainid2name[i], dice_per_class_avg[i], assd_per_class_avg[i]
        ))

    if cfg.SOLVER.CHECKPOINT_PERIOD == iteration:
        validate_and_save.best_score = -float('inf')
    current_score = float((mDice * 100 - mean_assd)+iteration/10000*0.5)
    if current_score > getattr(validate_and_save, "best_score", float('inf')):
        validate_and_save.best_score = current_score
        best_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.SEED}_best_model.pth")
        torch.save({
            'iteration': iteration,
            'feature_extractor': feature_extractor.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer_fea': optimizer_fea.state_dict(),
            'optimizer_cls': optimizer_cls.state_dict(),
            'best_score': current_score,
        }, best_path)
        logger.info("Saved best model with best score: {:.2f}".format(current_score))

    feature_extractor.train()
    classifier.train()

