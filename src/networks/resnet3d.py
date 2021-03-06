import torch
import torch.nn as nn

from building_blocks import BlockSeries, ConvolutionDownsample


class ResNetFlags(network_config):

    def __init__(self):
        network_config.__init__(self)
        self._name = "sparseresnet3d"
        self._help = "Sparse Resnet"

    def build_parser(self, network_parser):
        # this_parser = network_parser
        this_parser = network_parser.add_parser(self._name, help=self._help)

        this_parser.add_argument("--n-initial-filters",
            type    = int,
            default = 2,
            help    = "Number of filters applied, per plane, for the initial convolution")

        this_parser.add_argument("--res-blocks-per-layer",
            help    = "Number of residual blocks per layer",
            type    = int,
            default = 2)

        this_parser.add_argument("--network-depth",
            help    = "Total number of downsamples to apply",
            type    = int,
            default = 8)

        this_parser.add_argument("--batch-norm",
            help    = "Run using batch normalization",
            type    = str2bool,
            default = True)

        this_parser.add_argument("--leaky-relu",
            help    = "Run using leaky relu",
            type    = str2bool,
            default = False)


class ResNet(torch.nn.Module):

    def __init__(self, output_shape, params):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the params module

        # We apply an initial convolution, to each plane, to get n_inital_filters

        n_filters = params.n_initial_filters
        self.initial_convolution = torch.nn.Conv3d(
            in_channels  = 1, 
            out_channels = n_filters, 
            kernel_size  = [3, 3, 3], # Could be 5,5,5 with padding 2 or stride 2?
            stride       = [1, 1, 1],
            padding      = [1, 1, 1],
            bias         = params.use_bias)

        # Next, build out the convolution steps




        self.convolutional_layers = []
        for layer in range(params.network_depth):

            self.convolutional_layers.append( 
                BlockSeries(
                    n_filters, 
                    params.res_blocks_per_layer,
                    residual = True)
                )
            out_filters = filter_increase(n_filters)
            self.convolutional_layers.append( 
                ConvolutionDownsample(
                    inplanes = n_filters,
                    outplanes = out_filters)
                )
                # outplanes = n_filters + params.n_initial_filters))
            n_filters = out_filters

            self.add_module("conv_{}".format(layer), self.convolutional_layers[-2])
            self.add_module("down_{}".format(layer), self.convolutional_layers[-1])

        # Here, take the final output and convert to a dense tensor:


        self.final_layer = BlockSeries(
                    inplanes = n_filters, 
                    n_blocks = params.RES_BLOCKS_PER_LAYER,
                    residual = True)
                
        self.bottleneck  = torch.nn.Conv3d(
            in_channels  = n_filters, 
            out_channels = 2, 
            kernel_size  = [1, 1, 1],
            stride       = [1, 1, 1],
            padding      = [0, 0, 0],
            bias         = params.USE_BIAS
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

        params = utils.params.params()



        x = self.initial_convolution(x)



        for i in range(len(self.convolutional_layers)):
            x = self.convolutional_layers[i](x)

        # Apply the final steps to get the right output shape


        # Apply the final residual block:
        output = self.final_layer(x)
        # print " 1 shape: ", output.shape)

        # Apply the bottle neck to make the right number of output filters:
        output = self.bottleneck(output)


        # Apply global average pooling 
        kernel_size = output.shape[2:]
        output = torch.squeeze(nn.AvgPool3d(kernel_size, ceil_mode=False)(output))
        output = output.view([batch_size, output.shape[-1]])


        return output



