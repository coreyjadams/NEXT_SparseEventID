"""
An distributed-agnostic-aware gather layer, adapted from here:
https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py

"""

import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    def __init__(self, gather_fn, rank, size):
        self.gather_fn = gather_fn
        self.rank      = rank
        self.size      = size

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(self.size)]
        self.gather_fn(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[self.rank]
        return grad_out
