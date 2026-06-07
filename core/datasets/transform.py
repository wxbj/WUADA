import cv2
import numpy as np
import random
import torch
from scipy.ndimage import gaussian_filter, map_coordinates


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, image, label):
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label).long()
        return image, label


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, image, label):
        mean = torch.as_tensor(self.mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
        std = torch.as_tensor(self.std, dtype=image.dtype, device=image.device).view(-1, 1, 1) + 1e-6
        image = (image - mean) / std
        return image, label


class Resize(object):
    def __init__(self, size, resize_label=True):
        self.size = size
        self.resize_label = resize_label

    def __call__(self, image, label):
        image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_CUBIC)

        if self.resize_label:
            if label.ndim == 3:
                channels = [
                    cv2.resize(label[..., c], (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
                    for c in range(label.shape[2])
                ]
                label = np.stack(channels, axis=-1)
            else:
                label = cv2.resize(label, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)

        return image, label


class HorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label):
        if random.random() < self.prob:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
        return image, label


class Rotation(object):
    def __init__(self, degrees=20, prob=0.5):
        self.degrees = degrees
        self.prob = prob

    def __call__(self, image, label):
        if random.random() >= self.prob:
            return image, label

        angle = random.uniform(-self.degrees, self.degrees)

        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)

        image = cv2.warpAffine(image, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT_101)

        if label.ndim == 2:
            label = cv2.warpAffine(label, M, (w, h),flags=cv2.INTER_NEAREST,borderValue=255)
        else:
            chans = []
            for c in range(label.shape[2]):
                chans.append(cv2.warpAffine(label[..., c], M, (w, h),flags=cv2.INTER_NEAREST,borderValue=255))
            label = np.stack(chans, axis=-1)

        return image, label


class BrightnessContrast(object):
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, prob=0.2):
        assert 0 <= brightness_limit <= 1
        assert 0 <= contrast_limit <= 1
        assert 0 <= prob <= 1
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.prob = prob

    def __call__(self, image, label):
        if random.random() >= self.prob:
            return image, label

        img = image.astype(np.float32, copy=False)

        bf = random.uniform(1 - self.brightness_limit, 1 + self.brightness_limit)
        cf = random.uniform(1 - self.contrast_limit, 1 + self.contrast_limit)

        img = img * bf

        mean = img.reshape(-1, img.shape[2]).mean(axis=0)[None, None, :]
        img = (img - mean) * cf + mean

        return img, label


class ElasticDeformation(object):
    def __init__(self, alpha=20, sigma=4, prob=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob

    def __call__(self, image, label):
        if random.random() >= self.prob:
            return image, label

        image_np = image
        label_np = label
        h, w = image_np.shape[:2]

        dx1 = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma=6, mode="reflect") * self.alpha * 0.8
        dy1 = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma=6, mode="reflect") * self.alpha * 0.8
        dx2 = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma=2, mode="reflect") * self.alpha * 0.2
        dy2 = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma=2, mode="reflect") * self.alpha * 0.2
        dx = (dx1 + dx2).astype(np.float32)
        dy = (dy1 + dy2).astype(np.float32)

        max_disp = int(min(h, w) * 0.075)
        dx = np.clip(dx, -max_disp, max_disp)
        dy = np.clip(dy, -max_disp, max_disp)

        x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        coords = np.array([y + dy, x + dx])

        if image_np.ndim == 3:
            image_def = np.empty_like(image_np, dtype=np.float32)
            for c in range(image_np.shape[2]):
                image_def[..., c] = map_coordinates(image_np[..., c], coords, order=1, mode='reflect')
        else:
            image_def = map_coordinates(image_np, coords, order=1, mode='reflect').astype(np.float32)

        if label_np.ndim == 2:
            label_def = map_coordinates(label_np, coords, order=0, mode='reflect')
        else:
            label_def = np.empty_like(label_np)
            for c in range(label_np.shape[2]):
                label_def[..., c] = map_coordinates(label_np[..., c], coords, order=0, mode='reflect')

        return image_def.astype(np.float32, copy=False), label_def.astype(np.uint8, copy=False)
