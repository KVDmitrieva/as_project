import torch
import torch.nn as nn


class CrossEntopyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, prediction, target, **batch):
        return {
            "loss": self.ce_loss(prediction, target)
        }
