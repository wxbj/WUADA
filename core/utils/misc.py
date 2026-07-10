import errno
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F

import torch


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def adjust_lr_poly(optimizer, base_lr, curr_iter, max_iter, power=0.0, min_lr=0.0):
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    lr = max(lr, min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def inference(feature_extractor, classifier, image, label, flip=False):
    size = label.shape[-2:]
    if flip:
        image = torch.cat([image, torch.flip(image, [3])], 0)
    with torch.no_grad():
        output = classifier(feature_extractor(image))
    if isinstance(output, list):
        output = output[0]
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]

    return output.unsqueeze(dim=0)

def get_gray_image(npimg, number_class):
    if number_class == 8:
        class_to_gray = [0, 205, 164, 244, 38, 88, 52, 82]
    elif number_class == 5:
        class_to_gray = [0, 205, 164, 244, 52]
    else:
        class_to_gray = [0, 205, 244, 88]
    npimg = npimg.astype('uint8')

    h, w = npimg.shape
    gray_img = np.zeros((h, w), dtype='uint8')

    for class_id, gray_value in enumerate(class_to_gray):
        gray_img[npimg == class_id] = gray_value

    return Image.fromarray(gray_img, mode='L')

