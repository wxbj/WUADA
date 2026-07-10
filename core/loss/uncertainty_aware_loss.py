import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyAwareCELoss(nn.Module):
    def __init__(self, ignore_index=255, lambda_unc=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_unc = lambda_unc

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        valid_mask = (targets != self.ignore_index).float()

        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-10)), dim=1)
            entropy = entropy / torch.log(torch.tensor(float(num_classes), device=entropy.device))
            mean_entropy = (entropy * valid_mask).sum() / (valid_mask.sum() + 1e-10)
            entropy = entropy / (mean_entropy + 1e-10)
            weights = (1.0 + self.lambda_unc * entropy) * valid_mask

        weighted_loss = (weights * ce_loss).sum() / (weights.sum() + 1e-10)

        return weighted_loss

