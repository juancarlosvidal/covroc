import torch
from torch import Tensor
from torch.nn.modules import Module


class CustomMSELoss(Module):
    def __init__(self) -> None:
        super(CustomMSELoss, self).__init__()

    def forward(self, output: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        # weights = weights / torch.sum(weights)
        # return torch.sum(weights * torch.pow(output - target, 2), axis=0)
        return torch.sum(torch.pow(output - target, 2), axis=0)

