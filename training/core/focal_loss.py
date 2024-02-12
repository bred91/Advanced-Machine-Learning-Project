from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import ops


class FocalLoss(nn.Module):
    @dataclass
    class Parameters:
        alpha: Optional[float] = field(default=0.25)
        gamma: float = field(default=2.0)
        reduction: str = field(default="none")
        weight: Optional[Tensor] = field(default=None)


    def __init__(
            self, parameter: Parameters
    ) -> None:
        super().__init__()
        self.alpha: Optional[float] = parameter.alpha
        self.gamma: float = parameter.gamma
        self.reduction: str = parameter.reduction
        self.weight: Optional[Tensor] = parameter.weight

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
