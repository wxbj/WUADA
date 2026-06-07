import argparse
import os
import logging
import re

import numpy as np
from tqdm import tqdm
from medpy.metric.binary import assd, dc

import torch
import torch.backends.cudnn

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import mkdir, inference, get_gray_image
from core.utils.logger import setup_logger

import setproctitle
import warnings

warnings.filterwarnings('ignore')

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


def test(cfg):
    logger = logging.getLogger("WUADA.tester")
    logger.info("Start testing")

    if cfg.DATASETS.TEST_DIR != "LGE":
        trainid2name = {0: 'bg', 1: "MYO", 2: "LAC", 3: "LVC", 4: "AA"}
    else:
        trainid2name = {0: 'bg', 1: "MYO", 2: "LVC", 3: "RVC"}

    torch.cuda.set_device(int(cfg.MODEL.DEVICE.split(":")[1]))
    device = torch.device(cfg.MODEL.DEVICE)

    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)
    classifier = build_classifier(cfg)
    classifier.to(device)

    resume = str(os.path.join(cfg.OUTPUT_DIR, str(cfg.SEED) + cfg.RESUME))
    if cfg.RESUME and os.path.isfile(resume):
        logger.info("Loading checkpoint from {}".format(resume))
        checkpoint = torch.load(resume, map_location=torch.device('cpu'), weights_only=False)
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        classifier.load_state_dict(checkpoint['classifier'])

    feature_extractor.eval()
    classifier.eval()

    dice_per_class_total = np.zeros(cfg.MODEL.NUM_CLASSES, dtype=np.float64)
    dice_per_class_count = np.zeros(cfg.MODEL.NUM_CLASSES, dtype=np.int64)
    assd_per_class_total = np.zeros(cfg.MODEL.NUM_CLASSES, dtype=np.float64)
    assd_per_class_count = np.zeros(cfg.MODEL.NUM_CLASSES, dtype=np.int64)

    torch.cuda.empty_cache()
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST_DIR)
        mkdir(output_folder)

    test_datas = build_dataset(cfg, mode='test', is_source=False)

    assert cfg.TEST.BATCH_SIZE == 1, "Test batch size should be 1!"
    test_loader = torch.utils.data.DataLoader(test_datas, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True, sampler=None)

    current_path = None
    case_slices = []

    for test_data in tqdm(test_loader):
        test_input = test_data['img'].cuda(non_blocking=True)
        test_label = test_data['label'].cuda(non_blocking=True).long()
        name = test_data['name'][0]
        path = test_data['path'][0]

        if current_path is not None and path != current_path:
            dT, dC, aT, aC = _finalize_case(case_slices, cfg.MODEL.NUM_CLASSES)
            dice_per_class_total += dT
            dice_per_class_count += dC
            assd_per_class_total += aT
            assd_per_class_count += aC
            case_slices = []

        current_path = path

        pred_logits = inference(feature_extractor, classifier, test_input, test_label, True)  # [1,C,H,W]
        pred_2d = pred_logits.detach().cpu().numpy().squeeze().argmax(0).astype(np.uint8)

        mask = get_gray_image(pred_2d, cfg.MODEL.NUM_CLASSES)
        mask_filename = os.path.splitext(os.path.basename(name))[0] + ".png"
        mask.save(os.path.join(output_folder, mask_filename))

        gt_2d = test_label.cpu().numpy().squeeze().astype(np.uint8)  # [H,W]
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

    logger.info('Val metrics (3D): mDice {:.3f}'.format(mDice * 100))
    logger.info('Val metrics (3D): mean ASSD {:.3f}'.format(mean_assd))
    for i in range(1, cfg.MODEL.NUM_CLASSES):
        logger.info('{} {} Dice: {:.3f} ASSD: {:.3f}'.format(
            i, trainid2name[i], dice_per_class_avg[i] * 100, assd_per_class_avg[i]
        ))


def main():
    parser = argparse.ArgumentParser(description="Heat Start Active Domain Adaptive Semantic Segmentation Training")
    parser.add_argument("-cfg", "--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.opts:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    setproctitle.setproctitle(f'{cfg.PROCTITLE}')
    setup_logger("WUADA", cfg.OUTPUT_DIR, 0)

    test(cfg)


if __name__ == "__main__":
    main()
