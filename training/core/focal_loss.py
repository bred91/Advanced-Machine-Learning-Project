from typing import Optional

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import ops


class FocalLoss(nn.Module):
    def __init__(
            self, alpha: Optional[float], gamma: float = 2.0, reduction: str = "none", weight: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.alpha: Optional[float] = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.weight: Optional[Tensor] = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return ops.sigmoid_focal_loss(pred, target, self.alpha, self.gamma, self.reduction)
        # # -log(p)
        # ce_loss = F.cross_entropy(pred, target, reduction=self.reduction, weight=self.weight)
        # p = torch.exp(-ce_loss)
        # focal_loss = (1-p)**self.gamma * ce_loss
        # if self.reduction == "none":
        #     loss = focal_loss
        # elif self.reduction == "mean":
        #     loss = torch.mean(focal_loss)
        # elif self.reduction == "sum":
        #     loss = torch.sum(focal_loss)
        # else:
        #     raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        # return loss
