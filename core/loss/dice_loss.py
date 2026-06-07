import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1e-5, exclude_background=True):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.exclude_background = exclude_background

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        valid_mask = (targets != self.ignore_index)
        targets = torch.where(valid_mask, targets, torch.zeros_like(targets))

        target_one_hot = F.one_hot(targets, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)

        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        intersection = (probs * target_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        if self.exclude_background and num_classes > 1:
            dice = dice[:, 1:]
        loss = 1 - dice.mean()

        return loss
