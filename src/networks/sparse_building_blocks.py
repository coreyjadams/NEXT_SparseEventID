import torch
import torch.nn as nn
import sparseconvnet as scn

class SparseBlock(nn.Module):

    def __init__(self, inplanes, outplanes):

        nn.Module.__init__(self)
        
        self.conv1 = scn.SubmanifoldConvolution(
            dimension   = 3, 
            nIn         = inplanes, 
            nOut        = outplanes, 
            filter_size = 3, 
            bias        = FLAGS.USE_BIAS)
        
        # if FLAGS.BATCH_NORM:
        self.bn1 = scn.BatchNormReLU(outplanes)
        # self.relu = scn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        # if FLAGS.BATCH_NORM:
        out = self.bn1(out)
        # else:
            # out = self.relu(out)

        return out



class SparseResidualBlock(nn.Module):

    def __init__(self, inplanes, outplanes):
        nn.Module.__init__(self)
        
        
        self.conv1 = scn.SubmanifoldConvolution(
            dimension   = 3, 
            nIn         = inplanes, 
            nOut        = outplanes, 
            filter_size = 3, 
            bias        = FLAGS.USE_BIAS)
        

        # if FLAGS.BATCH_NORM:
        self.bn1 = scn.BatchNormReLU(outplanes)

        self.conv2 = scn.SubmanifoldConvolution(
            dimension   = 3, 
            nIn         = outplanes,
            nOut        = outplanes,
            filter_size = 3,
            bias        = FLAGS.USE_BIAS)

        # if FLAGS.BATCH_NORM:
        self.bn2 = scn.BatchNormalization(outplanes)

        self.residual = scn.Identity()
        self.relu = scn.ReLU()

        self.add = scn.AddTable()

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)
        # if FLAGS.BATCH_NORM:
        out = self.bn1(out)
        # else:
            # out = self.relu(out)
        out = self.conv2(out)

        # if FLAGS.BATCH_NORM:
        out = self.bn2(out)

        # The addition of sparse tensors is not straightforward, since

        out = self.add([out, residual])

        out = self.relu(out)

        return out




class SparseConvolutionDownsample(nn.Module):

    def __init__(self, inplanes, outplanes):
        nn.Module.__init__(self)

        self.conv = scn.Convolution(
            dimension       = 3,
            nIn             = inplanes,
            nOut            = outplanes,
            filter_size     = 2,
            filter_stride   = 2,
            bias            = FLAGS.USE_BIAS
        )
        # if FLAGS.BATCH_NORM:
        self.bn   = scn.BatchNormalization(outplanes)
        self.relu = scn.ReLU()

    def forward(self, x):
        out = self.conv(x)

        # if FLAGS.BATCH_NORM:
        out = self.bn(out)

        out = self.relu(out)
        return out


class SparseBlockSeries(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, residual=False):
        torch.nn.Module.__init__(self)

        if residual:
            self.blocks = [ SparseResidualBlock(inplanes, inplanes) for i in range(n_blocks) ]
        else:
            self.blocks = [ SparseBlock(inplanes, inplanes) for i in range(n_blocks)]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x