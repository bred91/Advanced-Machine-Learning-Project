import torch.nn as nn
import torch

from training.core.cross_entropy_loss import CrossEntropyLoss
from training.core.focal_loss import FocalLoss
from training.core.jaccard_loss import JaccardLoss
from training.core.logitnorm import LogitNormLoss


def _forward(outputs, targets, l1: nn.Module, l2: nn.Module):
    return (l1.forward(outputs, targets) + l2.forward(outputs, targets)) / 2


class Joint_LnFl(nn.Module):
    def __init__(self, ln_parameters: LogitNormLoss.Parameters, fl_parameters: FocalLoss.Parameters):
        super(Joint_LnFl, self).__init__()
        self.ln = LogitNormLoss(ln_parameters)
        self.fl = FocalLoss(fl_parameters)

    def forward(self, outputs, targets):
        return _forward(outputs, targets, self.ln, self.fl)


class Joint_LnCe(nn.Module):
    def __init__(self, ln_parameters: LogitNormLoss.Parameters, ce_parameters: CrossEntropyLoss.Parameters):
        super(Joint_LnCe, self).__init__()
        self.ln = LogitNormLoss(ln_parameters)
        self.ce = CrossEntropyLoss(ce_parameters)

    def forward(self, outputs, targets):
        return _forward(outputs, targets, self.ln, self.ce)


class Joint_JlFl(nn.Module):
    def __init__(self, jl_parameters: JaccardLoss.Parameters, fl_parameters: FocalLoss.Parameters):
        super(Joint_JlFl, self).__init__()
        self.jl = JaccardLoss(jl_parameters)
        self.fl = FocalLoss(fl_parameters)

    def forward(self, outputs, targets):
        return _forward(outputs, targets, self.jl, self.fl)


class Joint_JlCe(nn.Module):
    def __init__(self, jl_parameters: JaccardLoss.Parameters, ce_parameters: CrossEntropyLoss.Parameters):
        super(Joint_JlCe, self).__init__()
        self.jl = JaccardLoss(jl_parameters)
        self.ce = CrossEntropyLoss(ce_parameters)

    def forward(self, outputs, targets):
        return _forward(outputs, targets, self.jl, self.ce)
