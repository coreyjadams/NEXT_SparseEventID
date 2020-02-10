import torch
from torch import nn



class Block(nn.Module):

    def __init__(self, inplanes, outplanes, bias):

        nn.Module.__init__(self)
        
        self.conv1 = torch.nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3, 3], 
            stride       = [1, 1, 1],
            padding      = [1, 1, 1],
            bias         = bias)
        
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

    def __init__(self, inplanes, outplanes, bias):
        nn.Module.__init__(self)
        
        
        self.conv1 = torch.nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3, 3], 
            stride       = [1, 1, 1],
            padding      = [1, 1, 1],
            bias         = bias)
        

        # if FLAGS.BATCH_NORM:
        self.bn1 = torch.nn.BatchNorm3d(outplanes)

        self.conv2 = torch.nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3, 3], 
            stride       = [1, 1, 1],
            padding      = [1, 1, 1],
            bias         = bias)

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

    def __init__(self, inplanes, outplanes, bias):
        nn.Module.__init__(self)

        self.conv = torch.nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [2, 2, 2], 
            stride       = [2, 2, 2],
            padding      = [1, 1, 1],
            bias         = bias)

        # if FLAGS.BATCH_NORM:
        self.bn = torch.nn.BatchNorm3d(outplanes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv(x)

        # if FLAGS.BATCH_NORM:
        out = self.bn(out)

        out = self.relu(out)
        return out




class BlockSeries(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, bias, residual=True):
        torch.nn.Module.__init__(self)

        if residual:
            self.blocks = [ ResidualBlock(inplanes, inplanes, bias) for i in range(n_blocks) ]
        else:
            self.blocks = [ Block(inplanes, inplanes, bias) for i in range(n_blocks) ]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x

