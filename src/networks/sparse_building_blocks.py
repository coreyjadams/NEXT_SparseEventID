import torch
import torch.nn as nn
import sparseconvnet as scn

from src.config.network import  Norm, BlockStyle

class InputNorm(nn.Module):

    def __init__(self, *, nIn, nOut):

        nn.Module.__init__(self)
        self.layer = scn.SparseGroupNorm(num_groups=1, num_channels=nOut)

    def forward(self, x):

        return self.layer(x)

class Block(nn.Module):

    def __init__(self, *, nIn, nOut, params, activation=scn.LeakyReLU):

        nn.Module.__init__(self)

        kernel = 3*[params.filter_size,]



        self.conv1 = scn.SubmanifoldConvolution(
            dimension   = 3,
            nIn         = nIn,
            nOut        = nOut,
            filter_size = kernel,
            bias        = params.bias)

        self._do_normalization = False
        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.norm = scn.BatchNormalization(nOut)
        elif params.normalization == Norm.group:
            self._do_normalization = True
            self.norm = scn.SparseGroupNorm(num_groups=1, num_channels=nOut)
        elif params.normalization == Norm.layer:
            raise Exception("Layer norm not supported in SCN")
        self.activation = activation()

    def forward(self, x):

        # print("Pre conv batch locations: ", x.get_spatial_locations()[:,-1])
        out = self.conv1(x)
        # print("Post conv batch locations: ", out.get_spatial_locations()[:,-1])
        if self._do_normalization:
            out = self.norm(out)
            # print("Post normalization: ", out.get_spatial_locations()[:,-1])
        out = self.activation(out)
        # print("Post activation: ", out.get_spatial_locations()[:,-1])
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

from sparseconvnet import SparseConvNetTensor

class SparseGELU(nn.Module):
    def forward(self, input):
        output = SparseConvNetTensor()
        output.features = nn.functional.gelu(input.features)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output
    def input_spatial_size(self, out_size):
        return out_size

class ConvNextBlock(nn.Module):

    def __init__(self, *, nIn, nOut, params):
        nn.Module.__init__(self)

        self.convolution_1 = scn.SubmanifoldConvolution(
            dimension   = 3,
            nIn         = nIn,
            nOut        = nIn,
            filter_size = [7,7,7],
            groups      = nIn,
            bias        = params.bias)
        
        self.convolution_2 = scn.SubmanifoldConvolution(
            dimension   = 3,
            nIn         = nIn,
            nOut        = 4*nIn,
            filter_size = [1,1,1],
            groups      = 1,
            bias        = params.bias)

        self.convolution_3 = scn.SubmanifoldConvolution(
            dimension   = 3,
            nIn         = 4*nIn,
            nOut        = nIn,
            filter_size = [1,1,1],
            groups      = 1,
            bias        = params.bias)

        self.residual = scn.Identity()
        self.gelu = SparseGELU()

        self._do_normalization = False
        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.norm = scn.BatchNormalization(nOut)
        elif params.normalization == Norm.group:
            self._do_normalization = True
            self.norm = scn.SparseGroupNorm(num_groups=1, num_channels=nOut)
        elif params.normalization == Norm.layer:
            raise Exception("Layer norm not supported in SCN")
            self.norm = lambda x : x

        self.add = scn.AddTable()

    def forward(self, x):

        residual = self.residual(x)

        x = self.convolution_1(x)

        x = self.norm(x)

        x = self.convolution_2(x)

        x = self.gelu(x)

        x = self.convolution_3(x)

        # The addition of sparse tensors is not straightforward, since



        out = self.add([x, residual])

        return out

class ConvolutionDownsample(nn.Module):

    def __init__(self, *, nIn, nOut, params):
        nn.Module.__init__(self)

        self.conv = scn.Convolution(
            dimension       = 3,
            nIn             = nIn,
            nOut            = nOut,
            filter_size     = 2,
            filter_stride   = 2,
            bias            = False
        )

        self._do_normalization = False
        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.norm = scn.BatchNormalization(nOut)
        elif params.normalization == Norm.group:
            self._do_normalization = True
            self.norm = scn.SparseGroupNorm(num_groups=1, num_channels=nOut)
        elif params.normalization == Norm.layer:
            raise Exception("Layer norm not supported in SCN")
        self.relu = scn.LeakyReLU()

    def forward(self, x):
        # print("Downsample pre locs: ", x.get_spatial_locations())
        # print("Downsample pre size: ", x.spatial_size)
        out = self.conv(x)
        # print("Downsample post locs: ", out.get_spatial_locations()[:])
        # print("Downsample post size: ", out.spatial_size)
        if self._do_normalization:
            out = self.norm(out)
        out = self.relu(out)
        return out


class Pooling(nn.Module):

    def __init__(self, *, nIn, nOut, params):
        nn.Module.__init__(self)

        self.pooling = scn.AveragePooling(
            dimension   = 3, 
            pool_size   = 2,
            pool_stride = 2,
        )

        self.filter_update = Block(
            nIn     = nIn,
            nOut    = nOut,
            params  = params,
            kernel  = [1,1,1]
        )

    def forward(self, x):

        # print("Entering pooling layer with locs: ", x.get_spatial_locations()[:,-1])
        # Apply the pooling:
        # print("Before pooling with spatial size: ", x.spatial_size)
        # print("Before Pooling with x locs: ", x.get_spatial_locations()[:,0])
        # print("Before Pooling with y locs: ", x.get_spatial_locations()[:,1])
        # print("Before Pooling with z locs: ", x.get_spatial_locations()[:,2])
        out = self.pooling(x)
        # print("After pooling with spatial size: ", out.spatial_size)
        # print("Before Pooling with x locs: ", out.get_spatial_locations()[:,0])
        # print("Before Pooling with y locs: ", out.get_spatial_locations()[:,1])
        # print("Before Pooling with z locs: ", out.get_spatial_locations()[:,2])
        # print("After pooling op with locs: ", out.get_spatial_locations()[:,-1])
        out = self.filter_update(out)
        # print("After filter update with locs: ", out.get_spatial_locations()[:,-1])
        return out

# class SparseConvolutionDownsample(nn.Module):

#     def __init__(self, inplanes, outplanes, batch_norm, leaky_relu):
#         nn.Module.__init__(self)

#         self.batch_norm = batch_norm
#         self.leaky_relu = leaky_relu

#         self.conv = scn.Convolution(dimension=3,
#             nIn             = inplanes,
#             nOut            = outplanes,
#             filter_size     = 2,
#             filter_stride   = 2,
#             bias            = False
#         )

#         if self.batch_norm:
#             self.bn   = scn.BatchNormalization(outplanes)

#         if self.leaky_relu: self.relu = scn.LeakyReLU()
#         else:                self.relu = scn.ReLU()

#     def forward(self, x):
#         out = self.conv(x)

#         if self.batch_norm:
#             out = self.bn(out)

#         out = self.relu(out)
#         return out


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

        self._do_normalization = False
        if params.normalization == Norm.batch:
            self.norm = scn.BatchNorm(nOut)
        elif params.normalization == Norm.group:
            self._do_normalization = True
            self.norm = scn.SparseGroupNorm(num_groups=1, num_channels=nOut)
        elif params.normalization == Norm.layer:
            raise Exception("Layer norm not supported in SCN")
        self.relu = scn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        if self._do_normalization: out = self.norm(out)
        out = self.relu(out)
        return out

class BlockSeries(torch.nn.Module):


    def __init__(self, *, nIn, n_blocks, params):
        torch.nn.Module.__init__(self)

        if params.block_style == BlockStyle.residual:
            self.blocks = [ 
                ResidualBlock(nIn    = nIn,
                              nOut   = nIn,
                              params = params)
                for i in range(n_blocks)
            ]
        elif params.block_style == BlockStyle.convnext:
            self.blocks = [
                ConvNextBlock(nIn    = nIn,
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
        # print("Entering block series of length ", len(self.blocks))
        # print("  Batch locs: ", x.get_spatial_locations()[:,-1])
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            # print("    after a block: ", x.get_spatial_locations()[:,-1])
        return x