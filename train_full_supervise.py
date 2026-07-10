import argparse
import os
import datetime
import logging
import time

import torch
import torch.nn as nn
import torch.utils
import torch.distributed
from torch.utils.data import DataLoader

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.loss.dice_loss import DiceLoss
from core.utils import set_random_seed
from core.utils.misc import adjust_lr_poly
from core.utils.validate_and_save import validate_and_save

import setproctitle
import warnings

warnings.filterwarnings('ignore')


def train(cfg):
    logger = logging.getLogger("WUADA.trainer")

    torch.cuda.set_device(int(cfg.MODEL.DEVICE.split(":")[1]))
    device = torch.device(cfg.MODEL.DEVICE)

    # Building Network
    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)
    classifier = build_classifier(cfg)
    classifier.to(device)

    logger.info("feature_extractor:\n{}".format(feature_extractor))
    logger.info("classifier:\n{}".format(classifier))

    # Construct optimizer
    optimizer_fea = torch.optim.AdamW(feature_extractor.parameters(),
                                      lr=cfg.SOLVER.FEA_BASE_LR, weight_decay=cfg.SOLVER.FEA_WEIGHT_DECAY)
    optimizer_cls = torch.optim.AdamW(classifier.parameters(),
                                      lr=cfg.SOLVER.CLS_BASE_LR, weight_decay=cfg.SOLVER.CLS_WEIGHT_DECAY)
    optimizer_fea.zero_grad()
    optimizer_cls.zero_grad()

    # Initialize data loader
    tgt_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_val_data = build_dataset(cfg, mode='val', is_source=False)
    tgt_train_loader = DataLoader(tgt_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
                                pin_memory=True, sampler=None)

    # Define loss function
    sup_criterion = nn.CrossEntropyLoss(ignore_index=255)
    dice_criterion = DiceLoss()

    iteration = 0
    start_training_time = time.time()
    end = time.time()
    max_iters = cfg.SOLVER.MAX_ITER
    meters = MetricLogger(delimiter="  ")

    logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
    feature_extractor.train()
    classifier.train()
    for batch_index, tgt_data in enumerate(tgt_train_loader):
        data_time = time.time() - end

        adjust_lr_poly(optimizer_fea, cfg.SOLVER.FEA_BASE_LR, iteration, cfg.SOLVER.MAX_ITER, power=cfg.SOLVER.FEA_POWER,
                       min_lr=cfg.SOLVER.FEA_MIN_LR)
        adjust_lr_poly(optimizer_cls, cfg.SOLVER.CLS_BASE_LR, iteration, cfg.SOLVER.MAX_ITER, power=cfg.SOLVER.CLS_POWER,
                       min_lr=cfg.SOLVER.CLS_MIN_LR)

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()

        # target data
        tgt_input, tgt_label = tgt_data['img'], tgt_data['label']
        tgt_input = tgt_input.cuda(non_blocking=True)
        tgt_label = tgt_label.cuda(non_blocking=True)
        tgt_size = tgt_input.size()[-2:]

        tgt_outs = classifier(feature_extractor(tgt_input), size=tgt_size)
        main_tgt_out = tgt_outs[0]
        aux_tgt_outs = tgt_outs[1:]

        loss = torch.Tensor([0]).cuda()

        # Target domain cross dice loss
        loss_dice_tgt = dice_criterion(main_tgt_out, tgt_label) * cfg.SOLVER.SRC_DICE_LOSS
        meters.update(loss_dice_tgt=loss_dice_tgt.item())
        loss += loss_dice_tgt

        # Target domain cross supervision loss
        loss_sup_tgt = sup_criterion(main_tgt_out, tgt_label) * cfg.SOLVER.SRC_CROSSENTROPY_LOSS
        for aux_out in aux_tgt_outs:
            loss_sup_tgt += sup_criterion(aux_out, tgt_label) * cfg.SOLVER.SRC_AUX_CROSSENTROPY_LOSS
        meters.update(loss_sup_tgt=loss_sup_tgt.item())
        loss += loss_sup_tgt

        loss.backward()
        optimizer_fea.step()
        optimizer_cls.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        iteration += 1
        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.02f} GB"
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
                )
            )

        # Validate and save the optimal model
        if iteration == cfg.SOLVER.MAX_ITER or iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            validate_and_save(cfg, iteration, feature_extractor, classifier, optimizer_fea, optimizer_cls,
                              tgt_val_loader, logger, tgt_val_data)

        if iteration == cfg.SOLVER.STOP_ITER:
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / cfg.SOLVER.STOP_ITER
        )
    )


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
    logger = setup_logger("WUADA", cfg.OUTPUT_DIR, 0)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    set_random_seed(cfg.SEED)

    train(cfg)


if __name__ == '__main__':
    main()

