import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != 255
        probs = torch.sigmoid(logits)

        p = probs[valid]
        t = targets[valid].float()

        intersection = (p * t).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != 255
        bce = F.binary_cross_entropy_with_logits(
            logits[valid], targets[valid].float()
        )
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def get_loss(name: str) -> nn.Module:
    if name == "dice":
        return DiceLoss()
    if name == "bce":
        return BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)
    if name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"Unknown loss: {name}. Choose from: dice, bce, bce_dice")
