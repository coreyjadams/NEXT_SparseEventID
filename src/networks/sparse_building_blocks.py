import torch
import torch.nn as nn
import sparseconvnet as scn

from src.config.network import  Norm


class Block(nn.Module):

    def __init__(self, *, nIn, nOut, params, activation=scn.LeakyReLU):

        nn.Module.__init__(self)

        self.conv1 = scn.SubmanifoldConvolution(
            dimension   = 3,
            nIn         = nIn,
            nOut        = nOut,
            filter_size = [3,3,3],
            bias        = params.bias)

        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.relu = scn.BatchNormLeakyReLU(nOut)
        elif params.normalization == Norm.layer:
            raise Exception("Layer norm not supported in SCN")
        else:
            self.relu = activation()

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)

        return out



class ResidualBlock(nn.Module):

    def __init__(self, *, nIn, nOut, params):
        nn.Module.__init__(self)

        self.convolution_1 = Block(
            nIn         = nIn,
            nOut        = nOut,
            params      = params)

        self.convolution_2 = Block(
            nIn         = nIn,
            nOut        = nOut,
            activation  = scn.Identity,
            params      = params)

        self.residual = scn.Identity()
        self.relu = scn.LeakyReLU()

        self.add = scn.AddTable()


    def forward(self, x):

        residual = self.residual(x)

        out = self.convolution_1(x)

        out = self.convolution_2(out)


        # The addition of sparse tensors is not straightforward, since

        out = self.add([out, residual])

        out = self.relu(out)

        return out


class ConvolutionDownsample(nn.Module):

    def __init__(self, *, nIn, nOut, params):
        nn.Module.__init__(self)

        self.conv = scn.Convolution(
            dimension       = 3,
            nIn             = nIn,
            nOut            = nOut,
            filter_size     = [2,2,2],
            filter_stride   = [2,2,2],
            bias            = params.bias
        )

        if params.normalization == Norm.batch:
            self.relu = scn.BatchNormLeakyReLU(nOut)
        elif params.normalization == Norm.layer:
            raise Exception("Layer norm not supported in SCN")
        else:
            self.relu = scn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)

        out = self.relu(out)
        return out


class ConvolutionUpsample(nn.Module):

    def __init__(self, *, nIn, nOut, params):
        nn.Module.__init__(self)

        self.conv = scn.Deconvolution(dimension=3,
            nIn             = nIn,
            nOut            = nOut,
            filter_size     = [2,2,2],
            filter_stride   = [2,2,2],
            bias            = params.bias
        )

        if params.normalization == Norm.batch:
            self.relu = scn.BatchNormLeakyReLU(nOut)
        elif params.normalization == Norm.layer:
            raise Exception("Layer norm not supported in SCN")
        else:
            self.relu = scn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class BlockSeries(torch.nn.Module):


    def __init__(self, *, nIn, n_blocks, params):
        torch.nn.Module.__init__(self)

        if params.residual:
            self.blocks = [ 
                ResidualBlock(nIn    = nIn,
                              nOut   = nIn,
                              params = params)
                for i in range(n_blocks)
            ]
        else:
            self.blocks = [ 
                Block(nIn    = nIn,
                      nOut   = nIn,
                      params = params)
                for i in range(n_blocks)
            ]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x