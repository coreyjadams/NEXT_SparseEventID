import torch
import torch.nn as nn
import sparseconvnet as scn

class SparseBlock(nn.Module):

    def __init__(self, inplanes, outplanes, bias, batch_norm):

        nn.Module.__init__(self)

        self.conv1 = scn.SubmanifoldConvolution(
            dimension   = 3,
            nIn         = inplanes,
            nOut        = outplanes,
            filter_size = 3,
            bias        = bias)

        if batch_norm:
            self.activation = scn.BatchNormReLU(outplanes)
        else:
            self.activation = scn.ReLU()
        # self.relu = scn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.activation(out)

        return out



class SparseResidualBlock(nn.Module):

    def __init__(self, inplanes, outplanes, bias, batch_norm):
        nn.Module.__init__(self)


        self.conv1 = scn.SubmanifoldConvolution(
            dimension   = 3,
            nIn         = inplanes,
            nOut        = outplanes,
            filter_size = 3,
            bias        = bias)


        if batch_norm:
            self.activation1 = scn.BatchNormReLU(outplanes)
        else:
            self.activation1 = scn.ReLU()

        self.conv2 = scn.SubmanifoldConvolution(
            dimension   = 3,
            nIn         = outplanes,
            nOut        = outplanes,
            filter_size = 3,
            bias        = bias)

        if batch_norm:
            self.activation2 = scn.BatchNormReLU(outplanes)
        else:
            self.activation2 = scn.ReLU()


        self.residual = scn.Identity()

        self.add = scn.AddTable()

    def forward(self, x):

        # This is using the pre-activation variant of resnet

        residual = self.residual(x)

        out = self.activation1(x)

        out = self.conv1(out)

        out = self.activation2(out)

        out = self.conv2(out)

        out = self.add([out, residual])

        return out




class SparseConvolutionDownsample(nn.Module):

    def __init__(self, inplanes, outplanes, bias, batch_norm):
        nn.Module.__init__(self)

        self.conv = scn.Convolution(
            dimension       = 3,
            nIn             = inplanes,
            nOut            = outplanes,
            filter_size     = 2,
            filter_stride   = 2,
            bias            = bias
        )
        # if FLAGS.BATCH_NORM:
        if batch_norm:
            self.activation = scn.BatchNormReLU(outplanes)
        else:
            self.activation = scn.ReLU()


    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        return out


class SparseBlockSeries(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, bias, batch_norm, residual=False):
        torch.nn.Module.__init__(self)

        if residual:
            self.blocks = [ SparseResidualBlock(inplanes, inplanes, bias, batch_norm) for i in range(n_blocks) ]
        else:
            self.blocks = [ SparseBlock(inplanes, inplanes, bias, batch_norm) for i in range(n_blocks)]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x
