import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat


# The basis of this implementation is from here:
# https://github.com/berniwal/swin-transformer-pytorch/blob/master/swin_transformer_pytorch/swin_transformer.py


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement, self.displacement), dims=(1, 2, 3))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, D, H, W, C)
#         window_size (tuple[int]): window size
#     Returns:
#         windows: (B*num_windows, window_size*window_size, C)
#     """
#     B, D, H, W, C = x.shape
#     x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
#     windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
#     return windows

# @lru_cache()
# def compute_mask(D, H, W, window_size, shift_size, device):
#     img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
#     cnt = 0
#     for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
#         for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
#             for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
#                 img_mask[:, d, h, w, :] = cnt
#                 cnt += 1
#     mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
#     mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
#     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#     return attn_mask


# def create_mask(window_size, displacement, upper_lower, left_right, front_back):
#     mask = torch.zeros(window_size ** 2, window_size ** 2, window_size**2)
#     # mask = torch.zeros(window_size ** 2, window_size ** 2)

#     mask = rearrange(mask, '(h1 w1 d1) (h2 w2 d2) (h3 w3 d3)-> h1 w1 d1 h2 w2 d2 h3 w3 d3', 
#                      h1=window_size, h2=window_size, h3 = window_size,
#                      w1 = window_size, w2 = window_size, w3 = window_size,
#                     )

#     if upper_lower:
#         mask[-displacement:, :, :, -displacement:, :, :, -displacement:, :, :] = float('-inf')
#         mask[:-displacement, :, :, :-displacement, :, :, :-displacement, :, :] = float('-inf')

#     if left_right:
#         mask[:, -displacement:, :, :, -displacement:, :, :, -displacement:, :] = float('-inf')
#         mask[:, :-displacement, :, :, :-displacement, :, :, :-displacement, :] = float('-inf')
        
#     if front_back:
#         mask[:, :, -displacement:, :, :, -displacement:, :, :, -displacement:] = float('-inf')
#         mask[:, :, :-displacement, :, :, :-displacement, :, :, :-displacement] = float('-inf')

#     mask = rearrange(mask, 'h1 w1 d1 h2 w2 d2 h3 w3 d3 -> (h1 w1 d1) (h2 w2 d2) (h3 w3 d3)')

#     return mask


def get_relative_distances(window_size):
    indices = torch.tensor(
        np.array(
            [[x, y, z] \
             for x in range(window_size) \
             for y in range(window_size) \
             for z in range(window_size)
            ] 
        )
    )
    # indices = torch.tensor(
    #     np.array(
    #         [[x, y] \
    #          for x in range(window_size) \
    #          for y in range(window_size) \
    #         ] 
    #     )
    # )

    # print(indices.shape)
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            # self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
            #                                                  upper_lower=True, left_right=False, front_back=False), 
            #                                                  requires_grad=False)
            # self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
            #                                                 upper_lower=False, left_right=True, front_back=False), 
            #                                                 requires_grad=False)
            # self.top_bottom_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
            #                                                 upper_lower=False, left_right=False, front_back=True),
            #                                                 requires_grad=False)


        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(
                torch.randn(2 * window_size - 1, 
                            2 * window_size - 1,
                            2 * window_size - 1)
                        )
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, n_d, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        nw_d = n_d // self.window_size


        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (nw_d w_d) (h c) -> b h (nw_h nw_w nw_d) (w_h w_w w_d) c',
                                h=h, w_h=self.window_size, w_w=self.window_size, w_d=self.window_size), qkv)


        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[
                self.relative_indices[:, :, 0], 
                self.relative_indices[:, :, 1],
                self.relative_indices[:, :, 2]]
        else:
            dots += self.pos_embedding

        # if self.shifted:
        #     print("dots.shape: ", dots.shape)
        #     print(self.upper_lower_mask.shape)
        #     dots[:, :, -nw_w:] += self.upper_lower_mask
        #     dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)


        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w nw_d) (w_h w_w w_d) d -> b (nw_h w_h) (nw_w w_w) (nw_d w_d) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, w_d=self.window_size,
                        nw_h=nw_h, nw_w=nw_w, nw_d = nw_d)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Conv3d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = self.downscaling_factor,
            stride       = self.downscaling_factor,
            padding      = 0,
        )

        # self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        # self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w, d = x.shape
        # new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        # x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        # x = self.linear(x)
        x = self.patch_merge(x)
        x = rearrange(x, " b c h w d -> b h w d c")
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(
            in_channels        = in_channels, 
            out_channels       = hidden_dimension,
            downscaling_factor = downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 4, 1, 2, 3)
    

# @dataclass
# class Swin(Representation):
#     type: EncoderType = EncoderType.swin
#     hidden_dim:   int = 96
#     layers: List[int] = field(default_factory=lambda : [2, 2, 6, 8 ])
#     heads:  List[int] = field(default_factory=lambda : [3, 6, 12, 24 ])
#     embed_dim:    int = 64
#     head_dim:     int = 32
#     window_size:  int = 7
#     downscaling_factors: List[int] = field(default_factory=lambda : [4, 2, 2, 2] ) 
#     relative_pos_embedding: bool = True

class Encoder(nn.Module):
    # def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
    #              downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
    def __init__(self, params, input_size):
        super().__init__()
        downscaling_factors = [2, 2, 2, 2]
        self.stage1 = StageModule(
            in_channels            = 1, 
            hidden_dimension       = params.encoder.hidden_dim, 
            layers                 = params.encoder.layers[0],
            downscaling_factor     = 2, 
            num_heads              = params.encoder.heads[0], 
            head_dim               = params.encoder.head_dim,
            window_size            = params.encoder.window_size, 
            relative_pos_embedding = params.encoder.relative_pos_embedding
        )

        self.stage2 = StageModule(
            in_channels            = params.encoder.hidden_dim,
            hidden_dimension       = params.encoder.hidden_dim * 2,
            layers                 = params.encoder.layers[1],
            downscaling_factor     = downscaling_factors[1],
            num_heads              = params.encoder.heads[1],
            head_dim               = params.encoder.head_dim,
            window_size            = params.encoder.window_size,
            relative_pos_embedding = params.encoder.relative_pos_embedding
        )
        
        self.stage3 = StageModule(
            in_channels            = params.encoder.hidden_dim * 2,
            hidden_dimension       = params.encoder.hidden_dim * 4,
            layers                 = params.encoder.layers[2],
            downscaling_factor     = downscaling_factors[2], 
            num_heads              = params.encoder.heads[2], 
            head_dim               = params.encoder.head_dim,
            window_size            = params.encoder.window_size, 
            relative_pos_embedding = params.encoder.relative_pos_embedding
        )
        self.stage4 = StageModule(
            in_channels            = params.encoder.hidden_dim * 4,
            hidden_dimension       = params.encoder.hidden_dim * 8,
            layers                 = params.encoder.layers[3],
            downscaling_factor     = downscaling_factors[3], 
            num_heads              = params.encoder.heads[3], 
            head_dim               = params.encoder.head_dim,
            window_size            = params.encoder.window_size, 
            relative_pos_embedding = params.encoder.relative_pos_embedding
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(params.encoder.hidden_dim * 8),
            nn.Linear(params.encoder.hidden_dim * 8, params.encoder.embed_dim)
        )
        self.output_shape = [params.encoder.embed_dim,]

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3, 4])
        return self.mlp_head(x)


def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return Encoder(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return Encoder(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return Encoder(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return Encoder(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)
