import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import sparseconvnet as scn
except:
    pass

import numpy

from src.config.network import GrowthRate, DownSampling
from src.config.framework import DataMode

class Attention(nn.Module):
    """Attention Mechanism from TIMM

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm:  bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)


        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class PatchEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, in_channels, embedding_dim):
        super().__init__()

        grid_size = []
        self.num_patches = 1
        for i in image_size:
            res = i % patch_size
            assert res == 0
            grid_size.append(i // patch_size)    
            self.num_patches *= grid_size[-1]


        self.output_shape = [self.num_patches, embedding_dim]

        self.proj = nn.Conv3d(in_channels, embedding_dim, 
                              kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.reshape( (x.shape[0:2]) + (-1,))

        # move the tokens to the first non-batch dim
        x = torch.permute(x, (0, 2, 1))

        return x

class Encoder(torch.nn.Module):

    def __init__(self, params, image_size):

        super().__init__()

        print(image_size)

        # Only Dense mode supported for ViT
        assert params.framework.mode == DataMode.dense

        self.patch_embedding = PatchEmbedding(
            image_size,
            patch_size  = params.encoder.patch_size,
            in_channels = 1,
            embedding_dim = params.encoder.embed_dim
        )

        # self.first_block = Block(
        #     nIn    = 1,
        #     nOut   = params.encoder.n_initial_filters,
        #     params = params.encoder
        # )
        # self.output_shape = [128, 8, 8, 8]
        self.output_shape = self.patch_embedding.output_shape
        self.output_shape = [params.encoder.embed_dim,]
        num_patches = self.patch_embedding.num_patches


        self.network_layers = torch.nn.ModuleList()



        for i_layer in range(params.encoder.depth):
            self.network_layers.append(
                Block(
                    params.encoder.embed_dim,
                    num_heads = params.encoder.embed_dim,
                )
            )


        # Parameters/Embeddings
        self.cls_token     = torch.nn.Parameter(torch.zeros(1,1,params.encoder.embed_dim))
        # 512 here should be num_patches
        self.pos_embedding = torch.nn.Parameter(torch.rand(1,1+num_patches,params.encoder.embed_dim)*0.02)

        # if params.framework.mode == DataMode.sparse:
        #     self.final_activation = scn.Tanh()
        # else:
        #     self.final_activation = torch.nn.Tanh()

        # self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)


    def forward(self, x):

        B = x.shape[0]
       
        x = self.patch_embedding(x)

        cls_token = self.cls_token.repeat(B,1,1)

        # Add the classification token:
        x = torch.cat([cls_token, x], dim=1)


        # Add the positional embedding:
        x = x + self.pos_embedding

        # Apply the encoder blocks:
        for i, l in enumerate(self.network_layers):
            x = l(x)

        # Pull off the classification token:
        # x = x[:,0,:]
        x = x.mean(dim=1)


        return x

