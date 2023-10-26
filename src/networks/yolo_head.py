import torch
import numpy

from . resnet import Encoder
from src.config.framework import DataMode

def build_networks(params, input_shape):

    resnet = Encoder(params, input_shape)
    output_shape = resnet.output_shape

    yolo_head = torch.nn.Sequential()


    # Only using dense blocks here:

    from . building_blocks import Block, ResidualBlock, BlockSeries
    from . building_blocks import ConvolutionDownsample
    from . building_blocks import MaxPooling


    current_number_of_filters = output_shape[0]

    for layer in params.head.layers:
        yolo_head.append(
            Block(
                nIn  = current_number_of_filters,
                nOut = layer,
                params = params.encoder
            )
        )
        current_number_of_filters = layer


    yolo_head.append(torch.nn.Sigmoid())

    return resnet, yolo_head
