import torch
import sparseconvnet as scn

import numpy

from src.config.network import GrowthRate, DownSampling


class Encoder(torch.nn.Module):

    def __init__(self, params, image_size):

        super().__init__()

        Block, BlockSeries, ConvDonsample, Pool, InputNorm = \
            self.import_building_blocks(params.framework.sparse)

        # How many filters did we start with?
        current_number_of_filters = params.encoder.n_initial_filters

        if params.framework.sparse:
            image_size = [64,64,128]
            self.input_layer = scn.InputLayer(
                dimension    = 3,
                # spatial_size = 512,
                spatial_size = torch.tensor(image_size)
            )
        else:
            self.input_layer = torch.nn.Identity()

        # self.input_norm = InputNorm(nIn=1, nOut=1)

        self.first_block = Block(
            nIn    = 1,
            nOut   = params.encoder.n_initial_filters,
            params = params.encoder
        )

        if params.encoder.downsampling == DownSampling.convolutional:
            downsampler = ConvDonsample
        else:
            downsampler = Pool

        self.network_layers = torch.nn.ModuleList()



        for i_layer in range(params.encoder.depth):
            self.network_layers.append(
                BlockSeries(
                    nIn      = current_number_of_filters,
                    n_blocks = params.encoder.blocks_per_layer,
                    params   = params.encoder
                )
            )
            next_filters = self.increase_filters(current_number_of_filters, params.encoder)
            self.network_layers.append(downsampler(
                    nIn    = current_number_of_filters,
                    nOut   = next_filters,
                    params = params.encoder
                )
            )
            current_number_of_filters = next_filters


        self.bottleneck  = Block(
            nIn    = current_number_of_filters,
            nOut   = params.encoder.n_output_filters,
            # kernel = [1,1,1],
            params = params.encoder
        )


        final_shape = [i // 2**params.encoder.depth for i in image_size]
        self.output_shape = [params.encoder.n_output_filters,] +  final_shape

        # We apply a global pooling layer to the image, to produce the encoding:
        if params.framework.sparse:
            self.pool = torch.nn.Sequential(
                # scn.AveragePooling(
                #     dimension = 3,
                #     pool_size = self.output_shape[1:],
                #     pool_stride = 1
                #     ),
                scn.SparseToDense(
                    dimension=3, nPlanes=self.output_shape[0]),
                torch.nn.AvgPool3d(self.output_shape[1:], divisor_override=1),

            )

        else:
            self.pool = torch.nn.AvgPool3d(self.output_shape[1:])


        # if params.framework.sparse:
        #     self.final_activation = scn.Tanh()
        # else:
        #     self.final_activation = torch.nn.Tanh()

        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)


    def forward(self, x):

        # print(x)

        scnTensor = self.input_layer(x)


        output = self.first_block(scnTensor)

        for l in self.network_layers:
            output = l(output)
  

        output = self.bottleneck(output)

        output = self.pool(output)

        output = self.flatten(output)

        summed = scn.SparseToDense(dimension=3, nPlanes=1)(scnTensor)
        summed = torch.nn.AvgPool3d([64,64,128],divisor_override=1)(summed)
        summed = summed.reshape((summed.shape[0],))


        return output, summed
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
            from . sparse_building_blocks import InputNorm
            from . sparse_building_blocks import Pooling
        else:
            from . building_blocks import Block, BlockSeries
            from . building_blocks import ConvolutionDownsample
            from . building_blocks import MaxPooling
            from . building_blocks import InputNorm
        return Block, BlockSeries, ConvolutionDownsample, Pooling, InputNorm
