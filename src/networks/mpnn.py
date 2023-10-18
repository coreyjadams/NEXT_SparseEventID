import torch
import torch_geometric as geom

import numpy

from src.config.network import GrowthRate, DownSampling
from src.config.framework import DataMode


class Encoder(torch.nn.Module):

    def __init__(self, params, image_size):
        super().__init__()


        self.initial_layer = geom.nn.MessagePassing()
        pass

    def forward(self, batch):

        return 1.0