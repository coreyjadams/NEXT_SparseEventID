import torch
import torch.nn as nn

from . building_blocks import BlockSeries, ConvolutionDownsample
from . network_config  import str2bool


class DiscriminatorFlags(object):

    def __init__(self):
        self._name = "discriminator"
        self._help = "Sparse Resnet"

    def build_parser(self, parser):

        parser.add_argument("--discriminator-n-initial-filters",
            type    = int,
            default = 8,
            help    = "Number of filters applied, per plane, for the initial convolution")

        parser.add_argument("--discriminator-blocks-per-layer",
            help    = "Number of residual blocks per layer",
            type    = int,
            default = 1)

        parser.add_argument("--discriminator-network-depth",
            help    = "Total number of downsamples to apply",
            type    = int,
            default = 4)

        parser.add_argument("--discriminator-use-bias",
            help    = "Use a bias activation in layers",
            type    = str2bool,
            default = True)


        # parser.add_argument("--discriminator-batch-norm",
        #     help    = "Run using batch normalization",
        #     type    = str2bool,
        #     default = True)

        # parser.add_argument("--discriminator-leaky-relu",
        #     help    = "Run using leaky relu",
        #     type    = str2bool,
        #     default = False)


class Discriminator(torch.nn.Module):

    def __init__(self, params):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the FLAGS module

        # We apply an initial convolution, to each plane, to get n_inital_filters

        n_filters = params.discriminator_n_initial_filters
        self.initial_convolution = torch.nn.Conv3d(
            in_channels  = 1,
            out_channels = n_filters,
            kernel_size  = [3, 3, 3], # Could be 5,5,5 with padding 2 or stride 2?
            stride       = [1, 1, 1],
            padding      = [1, 1, 1],
            bias         = params.discriminator_use_bias)

        # Next, build out the convolution steps


        self.convolutional_layers = torch.nn.ModuleList()
        for layer in range(params.discriminator_network_depth):

            self.convolutional_layers.append(
                BlockSeries(
                    n_filters,
                    params.discriminator_blocks_per_layer,
                    bias = params.discriminator_use_bias,
                    residual = True)
                )
            out_filters = 2*n_filters
            self.convolutional_layers.append(
                ConvolutionDownsample(
                    inplanes  = n_filters,
                    outplanes = out_filters,
                    bias = params.discriminator_use_bias)
                )
                # outplanes = n_filters + params.N_INITIAL_FILTERS))
            n_filters = out_filters


        # Here, take the final output and convert to a dense tensor:


        self.final_layer = BlockSeries(
                    inplanes = n_filters,
                    n_blocks = params.discriminator_blocks_per_layer,
                    residual = True,
                    bias = params.discriminator_use_bias)

        self.bottleneck  = torch.nn.Conv3d(
            in_channels  = n_filters,
            out_channels = 1,
            kernel_size  = [1, 1, 1],
            stride       = [1, 1, 1],
            padding      = [0, 0, 0],
            bias         = params.discriminator_use_bias
        )


        # # The rest of the final operations (reshape, softmax) are computed in the forward pass


        # # Configure initialization:
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d) or isinstance(m, scn.SubmanifoldConvolution):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, scn.BatchNormReLU):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)



    def forward(self, x):

        batch_size = x.shape[0]

        x = self.initial_convolution(x)



        for i in range(len(self.convolutional_layers)):
            x = self.convolutional_layers[i](x)

        # Apply the final steps to get the right output shape


        # Apply the final residual block:
        output = self.final_layer(x)
        # print " 1 shape: ", output.shape)

        # Apply the bottle neck to make the right number of output filters:
        output = self.bottleneck(output)

        # print(output.shape)
        # Apply global average pooling
        kernel_size = output.shape[2:]
        output = torch.squeeze(nn.AvgPool3d(kernel_size, ceil_mode=False)(output))
        # print(output.shape)

        output = torch.sigmoid(output)
        # print(output.shape)

        return output
