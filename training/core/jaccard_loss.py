from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn
import torch


class JaccardLoss(nn.Module):

    @dataclass
    class Parameters:
        weight: Optional[torch.Tensor] = field(default=None)

    def __init__(self, parameters: Parameters):
        super(JaccardLoss, self).__init__()
        self.weight = parameters.weight

    def forward(self, outputs, targets):
        if self.weight is not None:
            outputs *= self.weight.view(1, 19, 1, 1)
        outputs = outputs.reshape(outputs.size()[0], -1)
        targets = (torch
                   .unsqueeze(targets, dim=1)
                   .expand(-1, 19, -1, -1)
                   .reshape(targets.size()[0], -1))
        jaccard = ((torch.sum(torch.min(outputs, targets), dim=1, keepdim=True) + 1e-8) /
                   (torch.sum(torch.max(outputs, targets), dim=1, keepdim=True) + 1e-8))
        return 1 - torch.mean(jaccard)
