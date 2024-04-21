import torch
from torch import nn


class IoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        mask = (targets != self.ignore_index).float()
        targets = targets.float()
        num = torch.sum(outputs * targets * mask)
        den = torch.sum(outputs * mask + targets * mask - outputs * targets * mask)
        return 1 - num / den
