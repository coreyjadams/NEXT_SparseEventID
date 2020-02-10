import torch
from torch import nn

from . building_blocks import ResidualBlock
from . network_config  import str2bool

'''This module contains generators to map NEXT simulation to data and vice versa

There are two generators.  The first will map the data to simulation, and the 
other will map simulation to data.

As in the cycleGAN Paper, the models are inspired by "Perceptual Losses for Real-Time Style Transfer and Super-Resolution
".  This technique uses no pooling layers but instead uses strided and fractionally strided convolutions.

The networks use residual blocks such as the ones here: 
http://torch.ch/blog/2016/02/04/resnets.html

All kernels are 3x3x3 except for the first and last layers which are 9x9x9.

For style transfer, the networks use two stride 2x2x2 convolutions followed by
several residual blocks and then two convolutional layers with stride 1/2 to upsample.
The input and the output have the same size.
'''



class GeneratorFlags(object):

    def __init__(self):
        self._name = "generator"
        self._help = "Generator Network for CycleGAN"

    def build_parser(self, parser):

        parser.add_argument("--generator-n-initial-filters",
            type    = int,
            default = 2,
            help    = "Number of filters applied, per plane, for the initial convolution")

        parser.add_argument("--generator-use-bias",
            help    = "Use a bias unit in layers",
            type    = str2bool,
            default = True)

        parser.add_argument("--generator-n-residual-blocks",
            help    = "Number of residual blocks in the depth of the generator",
            type    = int,
            default = 1)
        # parser.add_argument("--batch-norm",
        #     help    = "Run using batch normalization",
        #     type    = str2bool,
        #     default = True)

        # parser.add_argument("--leaky-relu",
        #     help    = "Run using leaky relu",
        #     type    = str2bool,
        #     default = False)



class ResidualGenerator(nn.Module):

    def __init__(self, params):
        nn.Module.__init__(self)


        # Create an initial convolution to map from 1 to n_init filters

        self.initial_conv = torch.nn.Conv3d(
            in_channels  = 1, 
            out_channels = params.generator_n_initial_filters, 
            kernel_size  = [7, 7, 7], 
            stride       = [1, 1, 1],
            padding      = [3, 3, 3],
            bias         = params.generator_use_bias)
        
        # Create the first downsampling convolution:
        self.downsample_1 = torch.nn.Conv3d(
            in_channels  = params.generator_n_initial_filters, 
            out_channels = 2*params.generator_n_initial_filters, 
            kernel_size  = [3, 3, 3], 
            stride       = [2, 2, 2],
            padding      = [1, 1, 1],
            bias         = params.generator_use_bias)

        # Create the second downsampling convolution:
        self.downsample_2 = torch.nn.Conv3d(
            in_channels  = 2*params.generator_n_initial_filters, 
            out_channels = 4*params.generator_n_initial_filters,  
            kernel_size  = [3, 3, 3], 
            stride       = [2, 2, 2],
            padding      = [1, 1, 1],
            bias         = params.generator_use_bias)
        

        # Create the series of residual blocks:
        self.residual_blocks = torch.nn.ModuleList()
        for i in range(params.generator_n_residual_blocks):
            self.residual_blocks.append(ResidualBlock(
                inplanes  = 4*params.generator_n_initial_filters, 
                outplanes = 4*params.generator_n_initial_filters, 
                bias      = params.generator_use_bias)
            )

        # Create the first upsampling convolution:
        self.upsample_1 = torch.nn.ConvTranspose3d(
            in_channels  = 4*params.generator_n_initial_filters, 
            out_channels = 2*params.generator_n_initial_filters,  
            kernel_size  = [3, 3, 3], 
            stride       = [2, 2, 2],
            padding      = [1, 1, 1],
            output_padding = [1,1,1],
            bias         = params.generator_use_bias)


        # Create the second upsampling convolution:
        self.upsample_2 = torch.nn.ConvTranspose3d(
            in_channels  = 2*params.generator_n_initial_filters, 
            out_channels = params.generator_n_initial_filters,  
            kernel_size  = [3, 3, 3], 
            stride       = [2, 2, 2],
            padding      = [1, 1, 1],
            output_padding = [1,1,1],
            bias         = params.generator_use_bias)


        # Create a final convolution to map from n_init to 1 filters:
        self.final_conv = torch.nn.Conv3d(
            in_channels  = params.generator_n_initial_filters, 
            out_channels = 1, 
            kernel_size  = [7, 7, 7], 
            stride       = [1, 1, 1],
            padding      = [3, 3, 3],
            bias         = params.generator_use_bias)
        

    def forward(self, x):

        # Apply an initial convolution to map from 1 to n_init filters
        x = self.initial_conv(x)
        x = torch.nn.LeakyReLU()(x)
        # print("After initial_conv:", x.shape)

        # Apply the first downsampling convolution:
        x = self.downsample_1(x)
        x = torch.nn.LeakyReLU()(x)
        # print("After downsample_1:", x.shape)

        # Apply the second downsampling convolution:
        x = self.downsample_2(x)
        x = torch.nn.LeakyReLU()(x)
        # print("After downsample_2:", x.shape)

        # Apply the series of residual blocks:
        for i, block in enumerate(self.residual_blocks):
            x = block(x)

        # Apply the first upsampling convolution:
        x = self.upsample_1(x)
        x = torch.nn.LeakyReLU()(x)
        # print("After upsample_1: ", x.shape)

        # Apply the second upsampling convolution:
        x = self.upsample_2(x)
        x = torch.nn.LeakyReLU()(x)
        # print("After upsample_2: ", x.shape)
        
        # Apply a final convolution to map from n_init to 1 filters:
        x = self.final_conv(x)
        x = torch.nn.LeakyReLU()(x)
        # print("After final_conv: ", x.shape)

        return x












