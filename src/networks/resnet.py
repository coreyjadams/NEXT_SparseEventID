import torch
import sparseconvnet as scn

import numpy

from src.config.network import GrowthRate, DownSampling



def increase_filters(current_filters, params):
    if params.growth_rate == GrowthRate.multiplicative:
        return current_filters * 2
    else: # GrowthRate.additive
        return current_filters + params.n_initial_filters


def create_resnet(params, image_size):


    if params.framework.sparse:
        from . sparse_building_blocks import Block, ResidualBlock
        from . sparse_building_blocks import ConvolutionDownsample
        from . sparse_building_blocks import BlockSeries
    else:
        from . building_blocks import Block, ResidualBlock, BlockSeries
        from . building_blocks import ConvolutionDownsample
        from . building_blocks import MaxPooling


    # Build the encoder:
    encoder = torch.nn.Sequential()


    encoder.append(
        Block(
            nIn    = 1,
            nOut   = params.encoder.n_initial_filters,
            params = params.encoder
        ),
    )

    # How many filters did we start with?
    current_number_of_filters = params.encoder.n_initial_filters


    if params.encoder.downsampling == DownSampling.convolutional:
        downsampler = ConvolutionDownsample
    else:
        downsampler = MaxPooling

    for i_layer in range(params.encoder.depth):
        next_filters = increase_filters(current_number_of_filters, params.encoder)
        layer = torch.nn.Sequential(
            downsampler(
                nIn    = current_number_of_filters,
                nOut   = next_filters,
                params = params.encoder
            ),
            BlockSeries(
                nIn      = next_filters,
                n_blocks = params.encoder.blocks_per_layer,
                params   = params.encoder
            )
        )
        current_number_of_filters = next_filters
        encoder.append(layer)


    final_shape = [i // 2**params.encoder.depth for i in image_size]
    output_shape = [current_number_of_filters,] +  final_shape



    return encoder, output_shape
