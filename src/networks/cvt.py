import torch
import sparseconvnet as scn

import numpy

from src.config.network import GrowthRate, DownSampling
from src.config.framework import DataMode

class ConvolutionalTransformer(torch.nn.Module):
    pass


class ConvolutionalTokenEmbedding(torch.nn.Module):

    def __init__(self, nIn, nOut):
        super().__init__()

        self.conv = torch.nn.Conv3d(
            in_channels= nIn,
            out_channels= nOut,
            kernel_size = 7,
            stride = 2,
            padding = 3
        )


    def forward(self, x):

        x = self.conv(x)

        return x


# class ViTEncoderBlock(torch.nn.Module):

#     def __init__(self, params):
#         super().__init__()

#         # self.norm1 = torch.nn.LayerNorm()
#         # self.norm2 = torch.nn.LayerNorm()

#         # Create the key, query and value all at once:
#         self.qkv = torch.nn.Linear(params.encoder.embed_dim, 3*params.encoder.embed_dim)


#         self.norm1 = torch.nn.InstanceNorm1d(params.encoder.embed_dim)
#         self.norm2 = torch.nn.InstanceNorm1d(params.encoder.embed_dim)

#         self.attention = torch.nn.MultiheadAttention(
#             embed_dim = params.encoder.embed_dim, 
#             num_heads = params.encoder.num_heads,
#             batch_first = True,
#         )
#         hidden_dim = 4*params.encoder.embed_dim

#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(params.encoder.embed_dim, hidden_dim),
#             torch.nn.GELU(),
#             torch.nn.Dropout(params.encoder.dropout),
#             torch.nn.Linear(hidden_dim, params.encoder.embed_dim),
#             torch.nn.Dropout(params.encoder.dropout)
#         )
#     def forward(self, inputs):

#         x = inputs
#         # First, apply a normalization:
#         x = self.norm1(x)

#         # Generate the query, key and values:
#         qkv = self.qkv(x)
#         # print("qkv.shape: ", qkv.shape)
#         query, key, value = torch.chunk(qkv, chunks=3, dim = -1)
#         # print("query.shape: ", query.shape)
#         # print("key.shape: ", key.shape)
#         # print("value.shape: ", value.shape)

#         # Now, apply multihead attention:
#         attention = self.attention(query, key, value)[0]
#         # print(attention)
#         # Residual connection:
#         mid_x = inputs + attention    

#         x = self.norm2(mid_x)

#         x = self.mlp(x)

#         return x + mid_x

class Block(torch.nn.Module):

    def __init__(self, nIn, nOut):
        super().__init__()

        self.cte = ConvolutionalTokenEmbedding(nIn, nOut)

    def forward(self, x):

        x = self.cte(x)

        return x

class Encoder(torch.nn.Module):

    def __init__(self, params, image_size):

        super().__init__()

        # How many filters did we start with?
        if params.framework.mode == DataMode.sparse:
            # image_size = [64,64,128]
            self.input_layer = scn.InputLayer(
                dimension    = 3,
                # spatial_size = 512,
                spatial_size = torch.tensor(image_size)
            )
        else:
            self.input_layer = torch.nn.Identity()

        if params.framework.mode == DataMode.sparse:
            self.patchify = torch.nn.Sequential(
                scn.Convolution(
                    dimension     = 3,
                    nIn           = 1,
                    nOut          = params.encoder.n_initial_filters,
                    filter_size   = 7,
                    filter_stride = 1,
                    bias          = params.encoder.bias
                ),
                scn.SparseToDense(
                    dimension=3, nPlanes=params.encoder.embed_dim),
            )
        
        else:
            self.patchify = torch.nn.Conv3d(
                in_channels  = 1,
                out_channels = params.encoder.n_initial_filters,
                kernel_size  = 7,
                stride       = 1,
                padding      = 3
            )

        nIn = params.encoder.n_initial_filters

        self.network_layers = torch.nn.ModuleList()


        for i_layer in range(params.encoder.depth):
            nOut = 2*nIn
            self.network_layers.append(Block(nIn, nOut))
            nIn = nOut

        self.bottleneck = torch.nn.Conv3d(
            in_channels  = nOut,
            out_channels = params.encoder.n_output_filters,
            kernel_size  = 1,
            stride       = 1
        )

        nOut = params.encoder.n_output_filters

        self.output_shape = [nOut, 8, 8, 8]
        # self.output_shape = [params.encoder.embed_dim,]
        print(image_size)



    def forward(self, x):

        print("Initial Size: ", x.shape)
        x = self.input_layer(x)
        print("Size after input_layer: ", x.shape)
       
        x = self.patchify(x)
        print("Size after patchify: ", x.shape)

        for i, l in enumerate(self.network_layers):
            x = l(x)
            print(f"Size after layer {i}: ", x.shape)

        x = self.bottleneck(x)
        print(x.shape)
        # Pull off the classification token:
        # x = torch.mean(x, axis=(2,3,4))
        # print(x.shape)

        return x

