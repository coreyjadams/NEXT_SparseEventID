import torch
import torch.nn as nn
try:
    import sparseconvnet as scn
except:
    pass

import numpy

from src.config.network import GrowthRate, DownSampling
from src.config.framework import DataMode



# Most of the blocks here are adapted to 3D use case from here:
# https://github.com/changzy00/pytorch-attention/blob/master/vision_transformers/cvt.py

class ConvTransBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, kernel_size=3):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.attn = Attention(dim, num_heads, kernel_size)
        self.mlp = ConvMlp(dim, hidden_features)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(self.norm(x))
        return x
    
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, ks=3, attn_drop=0, proj_drop=0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim  // num_heads
        self.scale = head_dim ** -0.5
        self.conv_proj_qkv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=ks, stride=1, padding=(ks - 1) // 2, groups=dim),
            nn.InstanceNorm3d(dim),
            nn.Conv3d(dim, 3 * dim, 1)
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W, D = x.shape
        qkv = self.conv_proj_qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H, W, D)
        qkv = qkv.flatten(4).permute(1, 0, 2, 4, 3)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2).reshape(B, C, H * W * D)
        x = x.reshape(B, C, H, W, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class ConvolutionalTokenEmbedding(torch.nn.Module):

    def __init__(self, nIn, nOut, kernel, stride, padding):
        super().__init__()

        self.conv = torch.nn.Conv3d(
            in_channels= nIn,
            out_channels= nOut,
            kernel_size = kernel,
            stride = stride,
            padding = padding
        )
        self.norm = LayerNorm(nOut, data_format="channels_first")

    def forward(self, x):

        x = self.conv(x)
        return self.norm(x)

class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.conv2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        return x
    


class Encoder(torch.nn.Module):

    def __init__(self, params, image_size):

        super().__init__()


        # Only Dense mode supported for ViT
        assert params.framework.mode == DataMode.dense


        nIn  = 1

        dims   = [64, 192, 384]
        layers = [1, 2, 3]
        num_heads = [1, 3, 6]

        self.network_layers = nn.ModuleList()

        for i in range(len(dims)):
            nOut = dims[i] 
            if i == 0:
                self.network_layers.append(
                    ConvolutionalTokenEmbedding(nIn, nOut, kernel=7, stride=4, padding=2)
                )
            else:
                self.network_layers.append(
                    ConvolutionalTokenEmbedding(nIn, nOut, kernel=3, stride=2, padding=1)
                )
            for j in range(layers[i]):

                self.network_layers.append(
                    ConvTransBlock(
                        nOut, num_heads[i]
                    )
                )

            nIn = nOut

        self.output_shape = [params.encoder.n_output_filters, 4, 4, 4]
        # # self.output_shape = [params.encoder.embed_dim,]
        # # print(image_size)

        self.bottleneck = nn.Conv3d(nOut, params.encoder.n_output_filters, 1)


    def forward(self, x):



        for i, l in enumerate(self.network_layers):
            x = l(x)

        x = self.bottleneck(x)



        return x

