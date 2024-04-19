import torch
import sparseconvnet as scn

import numpy

from src.config.network import GrowthRate, DownSampling
from src.config.framework import DataMode

class ConvolutionalTransformerBlock(torch.nn.Module):

    def __init__(self, nIn, params):
        super().__init__()

        # Create the attention generator:
        self.attn = ConvolutionalAttention(nIn, params)

        # if params.framework.mode == DataMode.sparse:
        #     self.norm = scn.SparseGroupNorm(num_groups=1, num_channels=nIn)
        # else:
        #     self.norm = torch.nn.InstanceNorm3d(nIn)


    def forward(self, x):
        
        attn = self.attn(x)


        return attn
    
class ConvolutionalAttention(torch.nn.Module):
    
    def __init__(self, nIn, params):
        super().__init__()

        self.num_heads = params.encoder.num_heads

        if params.framework.mode == DataMode.sparse:

            self.norm = scn.SparseGroupNorm(num_groups=1, num_channels=nIn)

            self.k, self.q, self.v = [
                torch.nn.Sequential(
                    scn.SubmanifoldConvolution(dimension=3,
                        nIn         = nIn,
                        nOut        = nIn,
                        filter_size = 3,
                        groups      = nIn,
                        bias        = False
                    ),
                    
                ) for _ in range(3) ]
        else:
            self.norm = torch.nn.LayerNorm(nIn)
            self.q, self.k, self.v = [
                torch.nn.Sequential(
                    torch.nn.Conv3d(
                        in_channels  = nIn,
                        out_channels = nIn,
                        kernel_size  = 3,
                        stride       = 1,
                        padding      = "same"
                    ),
                ) for _ in range(3) ]

            self.attn = torch.nn.MultiheadAttention(nIn, self.num_heads, batch_first = True)

            self.mlp = torch.nn.Linear(nIn, nIn)


    def forward(self, x):
        

        q, k, v = self.q(x), self.k(x), self.v(x)


        input_shape = q.shape

        B = input_shape[0]
        c = input_shape[1]
        # h, w, d = input_shape[2], input_shape[3], input_shape[4]

        # This is the target shape before permuting:
        q_shape = (B, c, -1)

        q = torch.permute(torch.reshape(q, q_shape), (0,2,1))
        k = torch.permute(torch.reshape(k, q_shape), (0,2,1))
        v = torch.permute(torch.reshape(v, q_shape), (0,2,1))

        # Reshape for the number of heads:

        attn = self.attn(q, k, v, need_weights=False)[0]


        # Take the input, and shape it into flat tokens for the attention:
        token_x = torch.permute(torch.reshape(x, (B, c, -1)), (0,2,1))


        # Intermediate addition:
        inter_x = token_x + attn


        # Pass through the MLP
        output = inter_x + self.mlp(self.norm(inter_x))


        # Reshape the attention to match the input shape:
        output = torch.permute(output, (0,2,1))

        return torch.reshape(output, x.shape)


class ConvolutionalTokenEmbedding(torch.nn.Module):

    def __init__(self, nIn, nOut, params):
        super().__init__()

        if params.framework.mode == DataMode.sparse:

            # Padding isn't available for sparse convolutions.
            # So, the shape can't come out right in one conv.
            # Using a depthwise conv to increase the filters,
            # and a stride=2, filter=2 conv to downsample that.
            # Putting in a normalization in between
            self.conv = torch.nn.Sequential(
                scn.SubmanifoldConvolution(dimension=3,
                    nIn             = nIn,
                    nOut            = nOut,
                    filter_size     = 7,
                    groups          = nIn,
                    bias            = False
                ),
                scn.SparseGroupNorm(num_groups=1, num_channels=nOut),
                scn.Convolution(dimension=3,
                    nIn             = nOut,
                    nOut            = nOut,
                    filter_size     = 2,
                    filter_stride   = 2,
                    bias            = False
                )
            )
        else:
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

    def __init__(self, nIn, nOut, params):
        super().__init__()

        self.cte = ConvolutionalTokenEmbedding(nIn, nOut, params)

        self.encoder_layers = torch.nn.Sequential()
        for _ in range(params.encoder.blocks_per_layer):
            self.encoder_layers.append(
                ConvolutionalTransformerBlock(nOut, params)
            )
            pass

    def forward(self, x):

        x = self.cte(x)

        x = self.encoder_layers(x)

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
                scn.SubmanifoldConvolution(
                    dimension     = 3,
                    nIn           = 1,
                    nOut          = params.encoder.n_initial_filters,
                    filter_size   = 7,
                    bias          = params.encoder.bias
                ),
                # scn.SparseToDense(
                    # dimension=3, nPlanes=params.encoder.embed_dim),
            )
        
        else:
            self.patchify = torch.nn.Conv3d(
                in_channels  = 1,
                out_channels = params.encoder.n_initial_filters,
                kernel_size  = 7,
                stride       = 2,
                padding      = 3
            )

        nIn = params.encoder.n_initial_filters

        self.network_layers = torch.nn.ModuleList()


        for i_layer in range(params.encoder.depth):
            nOut = 2*nIn
            self.network_layers.append(Block(nIn, nOut, params))
            nIn = nOut

        if params.framework.mode == DataMode.sparse:
            self.bottleneck =  torch.nn.Sequential(
                scn.SubmanifoldConvolution(
                    dimension     = 3,
                    nIn           = nOut,
                    nOut          = params.encoder.n_output_filters,
                    filter_size   = 1,
                    # filter_stride = 1,
                    bias          = params.encoder.bias
                ),
                scn.SparseToDense(
                    dimension=3, nPlanes=params.encoder.n_output_filters),
            )
        else:
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

        # print("Initial Size: ", x.spatial_size)
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

