import math
import os
import random
import concurrent.futures

import torch

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from .floating_region import FloatingRegionScore


def RegionSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):
    feature_extractor.eval()
    classifier.eval()

    floating_region_score = FloatingRegionScore(in_channels=cfg.MODEL.NUM_CLASSES,size=5).cuda()
    per_region_pixels = (2 * cfg.ACTIVE.RADIUS_K + 1) ** 2
    active_radius = cfg.ACTIVE.RADIUS_K
    mask_radius = cfg.ACTIVE.RADIUS_K * 2
    active_ratio = cfg.ACTIVE.RATIO / len(cfg.ACTIVE.SELECT_ITER)

    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):
            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)
            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            max_workers = os.cpu_count() or 8
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                def process_sample(i):
                    active_mask = origin_mask[i].cuda(non_blocking=True)
                    ground_truth = origin_label[i].cuda(non_blocking=True)
                    size = (origin_size[i][0], origin_size[i][1])
                    num_pixel_cur = size[0] * size[1]
                    active = active_indicator[i]
                    selected = selected_indicator[i]

                    output = tgt_out[i:i + 1, :, :, :]
                    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)

                    score = floating_region_score(tgt_input, output)

                    score[active] = -float('inf')

                    active_regions = math.floor(num_pixel_cur * active_ratio / per_region_pixels)

                    for pixel in range(active_regions):
                        values, indices_h = torch.max(score, dim=0)
                        max_score, indices_w = torch.max(values, dim=0)

                        max_score_value = max_score.item()
                        if max_score_value == 0:
                            break

                        w = indices_w.item()
                        h = indices_h[w].item()

                        active_start_w = max(w - active_radius, 0)
                        active_start_h = max(h - active_radius, 0)
                        active_end_w = w + active_radius + 1
                        active_end_h = h + active_radius + 1

                        mask_start_w = max(w - mask_radius, 0)
                        mask_start_h = max(h - mask_radius, 0)
                        mask_end_w = w + mask_radius + 1
                        mask_end_h = h + mask_radius + 1

                        # mask out
                        score[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
                        active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
                        selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
                        # active sampling
                        active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                            ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]

                    np.save(path2mask[i], active_mask.cpu().numpy().astype(np.uint8))

                    indicator = {
                        'active': active,
                        'selected': selected
                    }
                    torch.save(indicator, path2indicator[i])

                futures = [executor.submit(process_sample, i) for i in range(len(origin_mask))]
                concurrent.futures.wait(futures)

    feature_extractor.train()
    classifier.train()


def PixelSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):
    feature_extractor.eval()
    classifier.eval()

    active_pixels = math.ceil(cfg.ACTIVE.PIXELS * (1 - cfg.ACTIVE.RANDOM_PIXS_RATIO) / len(cfg.ACTIVE.SELECT_ITER))
    floating_region_score = FloatingRegionScore(in_channels=cfg.MODEL.NUM_CLASSES,size=5).cuda()
    mask_radius = cfg.ACTIVE.RADIUS_K

    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):

                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                score = floating_region_score(tgt_input, output)

                score[active] = -float('inf')

                for pixel in range(active_pixels):
                    values, indices_h = torch.max(score, dim=0)
                    max_score, indices_w = torch.max(values, dim=0)

                    max_score_value = max_score.item()
                    if max_score_value == 0:
                        break

                    w = indices_w.item()
                    h = indices_h[w].item()

                    start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    end_w = w + mask_radius + 1
                    end_h = h + mask_radius + 1
                    # mask out
                    score[start_h:end_h, start_w:end_w] = -float('inf')
                    active[start_h:end_h, start_w:end_w] = True
                    selected[h, w] = True
                    # active sampling
                    active_mask[h, w] = ground_truth[h, w]

                np.save(path2mask[i], active_mask.cpu().numpy().astype(np.uint8))

                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()


def RandomSelectionPix(cfg, tgt_epoch_loader):
    mask_radius = cfg.ACTIVE.RADIUS_K
    active_ratio = cfg.ACTIVE.RANDOM_PIXS_RATIO
    center_ratio = 0.8

    for tgt_data in tqdm(tgt_epoch_loader):
        path2mask = tgt_data['path_to_mask']
        origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
        origin_size = tgt_data['size']
        active_indicator = tgt_data['active']
        selected_indicator = tgt_data['selected']
        path2indicator = tgt_data['path_to_indicator']

        for i in range(len(origin_mask)):
            active_mask = origin_mask[i]
            ground_truth = origin_label[i]
            size = (origin_size[i][0], origin_size[i][1])
            active = active_indicator[i]
            selected = selected_indicator[i]

            H, W = size
            active_pixels = math.floor(active_ratio * cfg.ACTIVE.PIXELS)

            center_H = int(H * center_ratio)
            center_W = int(W * center_ratio)
            start_H = (H - center_H) // 2
            start_W = (W - center_W) // 2
            end_H = start_H + center_H
            end_W = start_W + center_W

            count = 0
            attempts = 0
            max_attempts = active_pixels * 10

            while count < active_pixels and attempts < max_attempts:
                h = random.randint(start_H, end_H - 1)
                w = random.randint(start_W, end_W - 1)

                if active[h, w]:
                    attempts += 1
                    continue

                start_w = max(w - mask_radius, 0)
                start_h = max(h - mask_radius, 0)
                end_w = min(w + mask_radius + 1, W)
                end_h = min(h + mask_radius + 1, H)

                active[start_h:end_h, start_w:end_w] = True
                selected[h, w] = True
                active_mask[h, w] = ground_truth[h, w]
                count += 1

            np.save(path2mask[i], active_mask.cpu().numpy().astype(np.uint8))

            indicator = {
                'active': active,
                'selected': selected
            }
            torch.save(indicator, path2indicator[i])


def RandomSelectionRegion(cfg, tgt_epoch_loader):
    per_region_pixels = (2 * cfg.ACTIVE.RADIUS_K + 1) ** 2
    active_radius = cfg.ACTIVE.RADIUS_K
    mask_radius = cfg.ACTIVE.RADIUS_K * 2
    active_ratio = cfg.ACTIVE.RANDOM_RATIO
    center_ratio = 0.8

    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):
            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            def process_single(i):
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                H, W = size
                num_pixel_cur = size[0] * size[1]
                active = active_indicator[i]
                selected = selected_indicator[i]

                active_regions = math.floor(num_pixel_cur * active_ratio / per_region_pixels)
                attempts = 0
                max_attempts = active_regions * 10

                center_H = int(H * center_ratio)
                center_W = int(W * center_ratio)
                start_H = (H - center_H) // 2
                start_W = (W - center_W) // 2
                end_H = start_H + center_H
                end_W = start_W + center_W

                count = 0
                while count < active_regions and attempts < max_attempts:
                    h = random.randint(max(start_H + mask_radius, 0), min(end_H - mask_radius - 1, H - 1))
                    w = random.randint(max(start_W + mask_radius, 0), min(end_W - mask_radius - 1, W - 1))
                    attempts += 1

                    if active[h, w]:
                        continue

                    active_start_w = max(w - active_radius, 0)
                    active_start_h = max(h - active_radius, 0)
                    active_end_w = min(w + active_radius + 1, size[1])
                    active_end_h = min(h + active_radius + 1, size[0])

                    mask_start_w = max(w - mask_radius, 0)
                    mask_start_h = max(h - mask_radius, 0)
                    mask_end_w = min(w + mask_radius + 1, size[1])
                    mask_end_h = min(h + mask_radius + 1, size[0])

                    # mask out
                    active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
                    selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
                    active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                        ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]

                    count += 1

                np.save(path2mask[i], active_mask.cpu().numpy().astype(np.uint8))

                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

            num_workers = os.cpu_count() or 8
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_single, i) for i in range(len(origin_mask))]
                for f in futures:
                    f.result()
