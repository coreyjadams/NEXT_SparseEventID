import torch
import torch.nn as nn
import sparseconvnet as scn

from src import utils

FLAGS = utils.flags.FLAGS()


class Block(nn.Module):

    def __init__(self, inplanes, outplanes):

        nn.Module.__init__(self)
        


        self.conv1 = torch.nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3, 3], 
            stride       = [1, 1, 1],
            padding      = [1, 1, 1],
            bias         = FLAGS.USE_BIAS)
        
        # if FLAGS.BATCH_NORM:
        self.bn1  = torch.nn.BatchNorm3d(outplanes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        # if FLAGS.BATCH_NORM:
        out = self.bn1(out)
        out = self.relu(out)
        # else:
            # out = self.relu(out)

        return out


class ResidualBlock(nn.Module):

    def __init__(self, inplanes, outplanes):
        nn.Module.__init__(self)
        
        
        self.conv1 = torch.nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3, 3], 
            stride       = [1, 1, 1],
            padding      = [1, 1, 1],
            bias         = FLAGS.USE_BIAS)
        

        # if FLAGS.BATCH_NORM:
        self.bn1 = torch.nn.BatchNorm3d(outplanes)

        self.conv2 = torch.nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3, 3], 
            stride       = [1, 1, 1],
            padding      = [1, 1, 1],
            bias         = FLAGS.USE_BIAS)

        # if FLAGS.BATCH_NORM:
        self.bn2 = torch.nn.BatchNorm3d(outplanes)

        self.relu = torch.nn.ReLU()


    def forward(self, x):

        residual = x

        out = self.conv1(x)
        # if FLAGS.BATCH_NORM:
        out = self.bn1(out)
        # else:
            # out = self.relu(out)
        out = self.conv2(out)

        # if FLAGS.BATCH_NORM:
        out = self.bn2(out)

        # The addition of sparse tensors is not straightforward, since

        out = out + residual

        return self.relu(out)



class ConvolutionDownsample(nn.Module):

    def __init__(self, inplanes, outplanes):
        nn.Module.__init__(self)

        self.conv2 = torch.nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [2, 2, 2], 
            stride       = [2, 2, 2],
            padding      = [1, 1, 1],
            bias         = FLAGS.USE_BIAS)

        # if FLAGS.BATCH_NORM:
        self.bn2 = torch.nn.BatchNorm3d(outplanes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv(x)

        # if FLAGS.BATCH_NORM:
        out = self.bn(out)

        out = self.relu(out)
        return out


class BlockSeries(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, residual=False):
        torch.nn.Module.__init__(self)

        if residual:
            self.blocks = [ ResidualBlock(inplanes, inplanes) for i in range(n_blocks) ]
        else:
            self.blocks = [ Block(inplanes, inplanes) for i in range(n_blocks) ]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x


def filter_increase(input_filters):
    # return input_filters * 2
    return input_filters + FLAGS.N_INITIAL_FILTERS

class ResNet(torch.nn.Module):

    def __init__(self, output_shape):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the FLAGS module

        # We apply an initial convolution, to each plane, to get n_inital_filters

        self.initial_convolution = torch.nn.Conv3d(
            in_channels  = 1, 
            out_channels = outplanes, 
            kernel_size  = [3, 3, 3], # Could be 5,5,5 with padding 2 or stride 2?
            stride       = [1, 1, 1],
            padding      = [1, 1, 1],
            bias         = FLAGS.USE_BIAS)

        n_filters = FLAGS.N_INITIAL_FILTERS
        # Next, build out the convolution steps




        self.convolutional_layers = []
        for layer in range(FLAGS.NETWORK_DEPTH):

            self.convolutional_layers.append( 
                BlockSeries(
                    n_filters, 
                    FLAGS.RES_BLOCKS_PER_LAYER,
                    residual = True)
                )
            out_filters = filter_increase(n_filters)
            self.convolutional_layers.append( 
                ConvolutionDownsample(
                    inplanes = n_filters,
                    outplanes = out_filters)
                )
                # outplanes = n_filters + FLAGS.N_INITIAL_FILTERS))
            n_filters = out_filters

            self.add_module("conv_{}".format(layer), self.convolutional_layers[-2])
            self.add_module("down_{}".format(layer), self.convolutional_layers[-1])

        # Here, take the final output and convert to a dense tensor:


        if FLAGS.LABEL_MODE == 'all':
            self.final_layer = BlockSeries(
                        inplanes = n_filters, 
                        n_blocks = FLAGS.RES_BLOCKS_PER_LAYER,
                        residual = True)
                    
            self.bottleneck  = torch.nn.Conv3d(
                in_channels  = n_filters, 
                out_channels = 2, 
                kernel_size  = [1, 1, 1],
                stride       = [1, 1, 1],
                padding      = [0, 0, 0],
                bias         = FLAGS.USE_BIAS
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
        
        batch_size = x[2]

        FLAGS = utils.flags.FLAGS()


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


        # Apply global average pooling 
        kernel_size = output.shape[2:]
        output = torch.squeeze(nn.AvgPool3d(kernel_size, ceil_mode=False)(output))
        output = output.view([batch_size, output.shape[-1]])


        return output



