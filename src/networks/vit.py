import torch
import sparseconvnet as scn

import numpy

from src.config.network import GrowthRate, DownSampling
from src.config.framework import DataMode


class ViTEncoderBlock(torch.nn.Module):

    def __init__(self, params):
        super().__init__()

        # self.norm1 = torch.nn.LayerNorm()
        # self.norm2 = torch.nn.LayerNorm()

        # Create the key, query and value all at once:
        self.qkv = torch.nn.Linear(params.encoder.embed_dim, 3*params.encoder.embed_dim)


        self.norm1 = torch.nn.InstanceNorm1d(params.encoder.embed_dim)
        self.norm2 = torch.nn.InstanceNorm1d(params.encoder.embed_dim)

        self.attention = torch.nn.MultiheadAttention(
            embed_dim = params.encoder.embed_dim, 
            num_heads = params.encoder.num_heads,
            batch_first = True,
        )
        hidden_dim = 4*params.encoder.embed_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(params.encoder.embed_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(params.encoder.dropout),
            torch.nn.Linear(hidden_dim, params.encoder.embed_dim),
            torch.nn.Dropout(params.encoder.dropout)
        )
    def forward(self, inputs):

        x = inputs
        # First, apply a normalization:
        x = self.norm1(x)

        # Generate the query, key and values:
        qkv = self.qkv(x)
        # print("qkv.shape: ", qkv.shape)
        query, key, value = torch.chunk(qkv, chunks=3, dim = -1)
        # print("query.shape: ", query.shape)
        # print("key.shape: ", key.shape)
        # print("value.shape: ", value.shape)

        # Now, apply multihead attention:
        attention = self.attention(query, key, value)[0]
        # print(attention)
        # Residual connection:
        mid_x = inputs + attention    

        x = self.norm2(mid_x)

        x = self.mlp(x)

        return x + mid_x

class Encoder(torch.nn.Module):

    def __init__(self, params, image_size):

        super().__init__()

        Block, BlockSeries, ConvDonsample, Pool, InputNorm = \
            self.import_building_blocks(params.framework.mode)

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

        # self.first_block = Block(
        #     nIn    = 1,
        #     nOut   = params.encoder.n_initial_filters,
        #     params = params.encoder
        # )
        # self.output_shape = [128, 8, 8, 8]
        self.output_shape = [params.encoder.embed_dim,]
        print(image_size)

        patches_shape = [p // 8 for p in image_size]

        if params.framework.mode == DataMode.sparse:
            self.patchify = torch.nn.Sequential(
                scn.Convolution(
                    dimension     = 3,
                    nIn           = 1,
                    nOut          = params.encoder.embed_dim,
                    filter_size   = 7,
                    filter_stride = 1,
                    bias          = params.encoder.bias
                ),
                scn.SparseToDense(
                    dimension=3, nPlanes=params.encoder.embed_dim),
            )
        
        else:
            raise Exception("Dense patchify needed")        

        self.network_layers = torch.nn.ModuleList()



        for i_layer in range(params.encoder.depth):
            self.network_layers.append(ViTEncoderBlock(params))


        # Parameters/Embeddings
        self.cls_token     = torch.nn.Parameter(torch.randn(1,1,params.encoder.embed_dim))
        # 512 here should be num_patches
        self.pos_embedding = torch.nn.Parameter(torch.randn(1,1+512,params.encoder.embed_dim))

        # if params.framework.mode == DataMode.sparse:
        #     self.final_activation = scn.Tanh()
        # else:
        #     self.final_activation = torch.nn.Tanh()

        # self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)


    def forward(self, x):


        x = self.input_layer(x)
       
        x = self.patchify(x)
        
        B = x.shape[0]
        
        x = x.reshape( (x.shape[0:2]) + (-1,))
        # move the tokens to the first non-batch dim
        x = torch.transpose(x, 1, 2)
        
        # Add the classification token:
        x = torch.cat([x, self.cls_token.repeat(B,1,1)], dim=1)

        # Add the positional embedding:
        x = x + self.pos_embedding

        #
        for i, l in enumerate(self.network_layers):
            x = l(x)

        # Pull off the classification token:
        x = x[:,0,:]

        return x


    def increase_filters(self, current_filters, params):
        if params.growth_rate == GrowthRate.multiplicative:
            return current_filters * 2
        else: # GrowthRate.additive
            return current_filters + params.n_initial_filters

    def import_building_blocks(self, mode):
        if mode == DataMode.sparse:
            from . sparse_building_blocks import Block
            from . sparse_building_blocks import ConvolutionDownsample
            from . sparse_building_blocks import BlockSeries
            from . sparse_building_blocks import InputNorm
            from . sparse_building_blocks import Pooling
        else:
            from . building_blocks import Block, BlockSeries
            from . building_blocks import ConvolutionDownsample
            from . building_blocks import MaxPooling
            from . building_blocks import InputNorm
        return Block, BlockSeries, ConvolutionDownsample, Pooling, InputNorm
