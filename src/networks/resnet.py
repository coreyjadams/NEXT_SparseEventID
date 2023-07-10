import torch
import sparseconvnet as scn

import numpy

from src.config.network import GrowthRate, DownSampling


class Encoder(torch.nn.Module):

    def __init__(self, params, image_size):

        super().__init__()

        Block, BlockSeries, ConvDonsample, MaxPool = \
            self.import_building_blocks(params.framework.sparse)

        # How many filters did we start with?
        current_number_of_filters = params.encoder.n_initial_filters

        if params.framework.sparse:
            self.input_layer = scn.InputLayer(
                dimension    = 3,
                spatial_size = torch.tensor(image_size)
            )
        else:
            self.input_layer = torch.nn.Identity()

        self.first_block = Block(
            nIn    = 1,
            nOut   = params.encoder.n_initial_filters,
            params = params.encoder
        )

        if params.encoder.downsampling == DownSampling.convolutional:
            downsampler = ConvDonsample
        else:
            downsampler = MaxPool

        self.network_layers = torch.nn.ModuleList()



        for i_layer in range(params.encoder.depth):
            next_filters = self.increase_filters(current_number_of_filters, params.encoder)
            self.network_layers.append(downsampler(
                    nIn    = current_number_of_filters,
                    nOut   = next_filters,
                    params = params.encoder
                )
            )
            self.network_layers.append(
                BlockSeries(
                    nIn      = next_filters,
                    n_blocks = params.encoder.blocks_per_layer,
                    params   = params.encoder
                )
            )
            current_number_of_filters = next_filters

        final_shape = [i // 2**params.encoder.depth for i in image_size]
        self.output_shape = [current_number_of_filters,] +  final_shape

        # We apply a pooling layer to the image:
        if params.framework.sparse:
            self.pool = torch.nn.Sequential(
                scn.MaxPooling(
                    dimension = 3,
                    pool_size = self.output_shape[1:],
                    pool_stride = 1
                    ),
                scn.SparseToDense(
                    dimension=3, nPlanes=current_number_of_filters)

            )

        else:
            self.pool = torch.nn.MaxPool3d(self.output_shape[1:])


        if params.framework.sparse:
            self.final_activation = scn.Tanh()
        else:
            self.final_activation = torch.nn.Tanh()

        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)


    def forward(self, x):

        output = self.input_layer(x)
        output = self.first_block(output)

        for l in self.network_layers:
            output = l(output)
        output = self.pool(output)
        output = self.flatten(output)
        return output

        # return self.final_activation(output)


    def increase_filters(self, current_filters, params):
        if params.growth_rate == GrowthRate.multiplicative:
            return current_filters * 2
        else: # GrowthRate.additive
            return current_filters + params.n_initial_filters

    def import_building_blocks(self, sparse):
        if sparse:
            from . sparse_building_blocks import Block
            from . sparse_building_blocks import ConvolutionDownsample
            from . sparse_building_blocks import BlockSeries
            MaxPooling = None
        else:
            from . building_blocks import Block, BlockSeries
            from . building_blocks import ConvolutionDownsample
            from . building_blocks import MaxPooling
        return Block, BlockSeries, ConvolutionDownsample, MaxPooling
