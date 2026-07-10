import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FloatingRegionScore(nn.Module):

    def __init__(self, in_channels=5, padding_mode='zeros', size=5):

        super(FloatingRegionScore, self).__init__()
        self.in_channels = in_channels
        assert size % 2 == 1, "error size"
        self.entropy_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size,
                                      stride=1, padding=int(size / 2), bias=False,
                                      padding_mode=padding_mode)
        weight = torch.ones((size, size), dtype=torch.float32)
        weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
        weight = nn.Parameter(weight)
        self.entropy_conv.weight = weight
        self.entropy_conv.requires_grad_(False)

    def forward(self,image, logit):
        logit = logit.squeeze(dim=0)  # [8, h ,w]
        p = torch.softmax(logit, dim=0)  # [8, h, w]
        pixel_entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0).unsqueeze(dim=0).unsqueeze(dim=0) / math.log(self.in_channels)  # [1, 1, h, w]
        region_sum_entropy = self.entropy_conv(pixel_entropy)  # [1, 1, h, w]
        count = self.entropy_conv(torch.ones_like(pixel_entropy))  # [1, 1, h, w]
        prediction_uncertainty = region_sum_entropy / count  # [1, 1, h, w]

        score = prediction_uncertainty

        pred = torch.argmax(p, dim=0, keepdim=True).unsqueeze(0).float()  # [1, 1, H, W]
        foreground_mask = (pred > 0).float()

        dilated_fg = F.max_pool2d(foreground_mask, kernel_size=3, stride=1, padding=1)

        score = score * dilated_fg + 0.5 * score * (1 - dilated_fg)

        return score.squeeze(dim=0).squeeze(dim=0)
