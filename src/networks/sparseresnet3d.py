import torch
import torch.nn as nn
import sparseconvnet as scn

from . sparse_building_blocks import SparseBlockSeries, SparseConvolutionDownsample
from . network_config  import str2bool


class ResNetFlags(object):

    def __init__(self):
        self._name = "sparseresnet3d"
        self._help = "Sparse Resnet"

    def build_parser(self, parser):

        parser.add_argument("--n-initial-filters",
            type    = int,
            default = 2,
            help    = "Number of filters applied, per plane, for the initial convolution")

        parser.add_argument("--res-blocks-per-layer",
            help    = "Number of residual blocks per layer",
            type    = int,
            default = 2)

        parser.add_argument("--network-depth",
            help    = "Total number of downsamples to apply",
            type    = int,
            default = 8)

        parser.add_argument("--batch-norm",
            help    = "Run using batch normalization",
            type    = str2bool,
            default = True)

        parser.add_argument("--use-bias",
            help    = "Use a bias unit in layers",
            type    = str2bool,
            default = True)

        parser.add_argument("--leaky-relu",
            help    = "Run using leaky relu",
            type    = str2bool,
            default = False)



class ResNet(torch.nn.Module):

    def __init__(self, args):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the flags module


        # Create the sparse input tensor:
        self.input_tensor = scn.InputLayer(dimension=3, spatial_size=(512))

        # Here, define the layers we will need in the forward path:



        # The convolutional layers, which can be shared or not across planes,
        # are defined below

        # We apply an initial convolution, to each plane, to get n_inital_filters

        self.initial_convolution = scn.SubmanifoldConvolution(
                dimension   = 3,
                nIn         = 1,
                nOut        = args.n_initial_filters,
                filter_size = 5,
                bias        = args.use_bias
            )

        n_filters = args.n_initial_filters
        # Next, build out the convolution steps




        self.convolutional_layers = torch.nn.ModuleList()
        for layer in range(args.network_depth):

            self.convolutional_layers.append(
                SparseBlockSeries(
                    n_filters,
                    args.res_blocks_per_layer,
                    args.use_bias,
                    residual = True)
                )
            out_filters = 2*n_filters

            self.convolutional_layers.append(
                SparseConvolutionDownsample(
                    inplanes  = n_filters,
                    outplanes = out_filters,
                    bias = args.use_bias)
                )
                # outplanes = n_filters + args.n_initial_filters))
            n_filters = out_filters

        # Here, take the final output and convert to a dense tensor:


        self.final_layer = SparseBlockSeries(
                    inplanes = n_filters,
                    n_blocks = args.res_blocks_per_layer,
                    bias = args.use_bias,
                    residual = True)

        self.bottleneck  = scn.SubmanifoldConvolution(dimension=3,
                    nIn=n_filters,
                    nOut=2,
                    filter_size=3,
                    bias=args.use_bias)

        self.sparse_to_dense = scn.SparseToDense(dimension=3, nPlanes=2)


        # # The rest of the final operations (reshape, softmax) are computed in the forward pass


        # # Configure initialization:
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d) or isinstance(m, scn.SubmanifoldConvolution):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, scn.BatchNormReLU):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)



    def forward(self, x):

        batch_size = x[2]



        x = self.input_tensor(x)

        x = self.initial_convolution(x)



        for i in range(len(self.convolutional_layers)):
            x = self.convolutional_layers[i](x)

        # Apply the final steps to get the right output shape

         # Apply the final residual block:
        output = self.final_layer(x)
        # print " 1 shape: ", output.shape)

        # Apply the bottle neck to make the right number of output filters:
        output = self.bottleneck(output)

        output = self.sparse_to_dense(output)

        # Apply global average pooling
        kernel_size = output.shape[2:]
        output = torch.squeeze(nn.AvgPool3d(kernel_size, ceil_mode=False)(output))
        output = output.view([batch_size, output.shape[-1]])




        return output
