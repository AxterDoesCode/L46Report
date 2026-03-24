# losses.py
# Custom loss functions for quantization-aware training.
#
# Implements the squared hinge loss used in Binary Neural Networks:
#   L = mean( max(0, 1 - predictions * targets)^2 )
# where targets are one-hot encoded with values in {-1, +1}.

import torch
from torch.autograd import Function
import torch.nn as nn


# Custom autograd Function giving explicit control over the forward
# value and backward gradient of the squared hinge loss.
class squared_hinge_loss(Function):

    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        # margin = max(0, 1 - p*t)
        output = 1. - predictions.mul(targets)
        output[output.le(0.)] = 0.
        loss = torch.mean(output.mul(output))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        # Gradient: -2 * t * margin / N, zero where margin <= 0
        output = 1. - predictions.mul(targets)
        output[output.le(0.)] = 0.
        grad_output.resize_as_(predictions).copy_(targets).mul_(-2.).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(predictions.numel())
        return grad_output, None


# nn.Module wrapper so the loss can be used like nn.CrossEntropyLoss.
class SqrHingeLoss(nn.Module):

    def __init__(self):
        super(SqrHingeLoss, self).__init__()

    def forward(self, input, target):
        return squared_hinge_loss.apply(input, target)
