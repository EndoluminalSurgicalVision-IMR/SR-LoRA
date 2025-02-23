"""
BackBone: Swin-TRM
PET method: SSF - <<Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning>>
Code: https://github.com/dongzelian/SSF
Paper: https://arxiv.org/abs/2210.08823
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from copy import deepcopy
from mmcls.models import BACKBONES
from mmcls.models.backbones.swin_transformer import (SwinBlock,
                                                     SwinBlockSequence,
                                                     SwinTransformer)
from mmcls.models.utils import resize_pos_embed, to_2tuple
from mmcls.models.utils.attention import ShiftWindowMSA, WindowMSA
from mmcv.cnn.bricks.transformer import (AdaptivePadding, PatchEmbed,
                                         PatchMerging)
from mmcv.runner.base_module import BaseModule, ModuleList
from typing import List, Sequence
from mmcls.models.utils import resize_pos_embed
from mmcv.cnn import (Linear, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import AdaptivePadding
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)
from mmcv.cnn.utils.weight_init import trunc_normal_
import warnings




class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


class SSFAdaLayer(nn.Module):
    def __init__(self, dim):
        super(SSFAdaLayer, self).__init__()
        self.ssf_scale, self.ssf_shift = self.init_ssf_scale_shift(dim)

    def forward(self, x):
        ssf_ada_x = self.ssf_ada(x)
        return ssf_ada_x

    def init_ssf_scale_shift(self, dim):
        scale = nn.Parameter(torch.ones(dim))
        shift = nn.Parameter(torch.zeros(dim))

        nn.init.normal_(scale, mean=1, std=0.02)
        nn.init.normal_(shift, std=0.02)

        return scale, shift

    def ssf_ada(self, x):
        assert self.ssf_scale.shape == self.ssf_shift.shape
        if x.shape[-1] == self.ssf_scale.shape[0]:
            return x * self.ssf_scale + self.ssf_shift
        elif x.shape[1] == self.ssf_scale.shape[0]:
            return x * self.ssf_scale.view(1, -1, 1, 1) + self.ssf_shift.view(1, -1, 1, 1)
        else:
            raise ValueError('The input tensor shape does not match the shape of the scale factor.')


class PatchEmbedSSF(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None
        
        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dims)
        if self.norm is not None:
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(embed_dims)

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """
        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        x = ssf_ada(x,self.ssf_scale_1,self.ssf_shift_1)
        if self.norm is not None:
            x = self.norm(x)
            x = ssf_ada(x,self.ssf_scale_2,self.ssf_shift_2)
        return x, out_size
    
class PatchMergingSSF(BaseModule):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map ((used in Swin Transformer)).
    Our implementation uses `nn.Unfold` to
    merge patches, which is about 25% faster than the original
    implementation. However, we need to modify pretrained
    models for compatibility.

    Args:
        in_channels (int): The num of input channels.
            to gets fully covered by filter and stride you specified.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adaptive_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(4 * in_channels)

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        if self.adaptive_padding:
            x = self.adaptive_padding(x)
            H, W = x.shape[-2:]

        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)
        x = self.sampler(x)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        
        x = ssf_ada(x,self.ssf_scale_1,self.ssf_shift_1)
        
        x = self.reduction(x)
        return x, output_size


class WindowMSASSF(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        
        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dims*3)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(embed_dims)

    def init_weights(self):
        super(WindowMSASSF, self).init_weights()

        trunc_normal_(self.relative_position_bias_table, std=0.02)



    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor, Optional): mask with shape of (num_windows, Wh*Ww,
                Wh*Ww), value should be between (-inf, 0].
        """
        B_, N, C = x.shape
        qkv = (ssf_ada(self.qkv(x),self.ssf_scale_1,self.ssf_shift_1)).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        s = ssf_ada(x,self.ssf_scale_2, self.ssf_shift_2)
        x = self.proj_drop(x)
        return x


    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSASSF(BaseModule):
    """Shift Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Defaults to True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults to dict(type='DropPath', drop_prob=0.).
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        version (str, optional): Version of implementation of Swin
            Transformers. Defaults to `v1`.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0,
                 proj_drop=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 pad_small_map=False,
                 input_resolution=None,
                 auto_pad=None,
                 window_msa=WindowMSASSF,
                 msa_cfg=dict(),
                 init_cfg=None):
        super().__init__(init_cfg)

        if input_resolution is not None or auto_pad is not None:
            warnings.warn(
                'The ShiftWindowMSA in new version has supported auto padding '
                'and dynamic input shape in all condition. And the argument '
                '`auto_pad` and `input_resolution` have been deprecated.',
                DeprecationWarning)

        self.shift_size = shift_size
        self.window_size = window_size
        assert 0 <= self.shift_size < self.window_size

        assert issubclass(window_msa, BaseModule), \
            'Expect Window based multi-head self-attention Module is type of' \
            f'{type(BaseModule)}, but got {type(window_msa)}.'
        self.w_msa = window_msa(
            embed_dims=embed_dims,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **msa_cfg,
        )

        self.drop = build_dropout(dropout_layer)
        self.pad_small_map = pad_small_map
        
        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dims)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(embed_dims)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, f"The query length {L} doesn't match the input "\
            f'shape ({H}, {W}).'
        query = query.view(B, H, W, C)

        window_size = self.window_size
        shift_size = self.shift_size

        if min(H, W) == window_size:
            # If not pad small feature map, avoid shifting when the window size
            # is equal to the size of feature map. It's to align with the
            # behavior of the original implementation.
            shift_size = shift_size if self.pad_small_map else 0
        elif min(H, W) < window_size:
            # In the original implementation, the window size will be shrunk
            # to the size of feature map. The behavior is different with
            # swin-transformer for downstream tasks. To support dynamic input
            # shape, we don't allow this feature.
            assert self.pad_small_map, \
                f'The input shape ({H}, {W}) is smaller than the window ' \
                f'size ({window_size}). Please set `pad_small_map=True`, or ' \
                'decrease the `window_size`.'

        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))

        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if shift_size > 0:
            query = torch.roll(
                query, shifts=(-shift_size, -shift_size), dims=(1, 2))

        attn_mask = self.get_attn_mask((H_pad, W_pad),
                                       window_size=window_size,
                                       shift_size=shift_size,
                                       device=query.device)

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(query, window_size)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, window_size, window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad,
                                        window_size)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if H != H_pad or W != W_pad:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)

        return x

    @staticmethod
    def window_reverse(windows, H, W, window_size):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    @staticmethod
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    @staticmethod
    def get_attn_mask(hw_shape, window_size, shift_size, device=None):
        if shift_size > 0:
            img_mask = torch.zeros(1, *hw_shape, 1, device=device)
            h_slices = (slice(0, -window_size), slice(-window_size,
                                                      -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size,
                                                      -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = ShiftWindowMSASSF.window_partition(
                img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        return attn_mask


class SwinBlockSSF(BaseModule):
    """Swin Transformer block.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        shift (bool): Shift the attention window or not. Defaults to False.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_path (float): The drop path rate after attention and ffn.
            Defaults to 0.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        attn_cfgs (dict): The extra config of Shift Window-MSA.
            Defaults to empty dict.
        ffn_cfgs (dict): The extra config of FFN. Defaults to empty dict.
        norm_cfg (dict): The config of norm layers.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size=7,
                 shift=False,
                 ffn_ratio=4.,
                 drop_path=0.,
                 pad_small_map=False,
                 attn_cfgs=dict(),
                 ffn_cfgs=dict(),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlockSSF, self).__init__(init_cfg)
        self.with_cp = with_cp

        _attn_cfgs = {
            'embed_dims': embed_dims,
            'num_heads': num_heads,
            'shift_size': window_size // 2 if shift else 0,
            'window_size': window_size,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'pad_small_map': pad_small_map,
            **attn_cfgs
        }
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSASSF(**_attn_cfgs)
        
        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dims)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(embed_dims)

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': 0,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': dict(type='GELU'),
            **ffn_cfgs
        }
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(**_ffn_cfgs)
        
    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = ssf_ada(x,self.ssf_scale_1,self.ssf_shift_1)
            x = self.attn(x, hw_shape)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = ssf_ada(x,self.ssf_scale_2,self.ssf_shift_2)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x

class SwinBlockSequenceSSF(BaseModule):
    """Module with successive Swin Transformer blocks and downsample layer.

    Args:
        embed_dims (int): Number of input channels.
        depth (int): Number of successive swin transformer blocks.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        downsample (bool): Downsample the output of blocks by patch merging.
            Defaults to False.
        downsample_cfg (dict): The extra config of the patch merging layer.
            Defaults to empty dict.
        drop_paths (Sequence[float] | float): The drop path rate in each block.
            Defaults to 0.
        block_cfgs (Sequence[dict] | dict): The extra config of each block.
            Defaults to empty dicts.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 depth,
                 num_heads,
                 window_size=7,
                 downsample=False,
                 downsample_cfg=dict(),
                 drop_paths=0.,
                 block_cfgs=dict(),
                 with_cp=False,
                 pad_small_map=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        if not isinstance(block_cfgs, Sequence):
            block_cfgs = [deepcopy(block_cfgs) for _ in range(depth)]

        self.embed_dims = embed_dims
        self.blocks = ModuleList()
        for i in range(depth):
            _block_cfg = {
                'embed_dims': embed_dims,
                'num_heads': num_heads,
                'window_size': window_size,
                'shift': False if i % 2 == 0 else True,
                'drop_path': drop_paths[i],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                **block_cfgs[i]
            }
            block = SwinBlockSSF(**_block_cfg)
            self.blocks.append(block)

        if downsample:
            _downsample_cfg = {
                'in_channels': embed_dims,
                'out_channels': 2 * embed_dims,
                'norm_cfg': dict(type='LN'),
                **downsample_cfg
            }
            self.downsample = PatchMergingSSF(**_downsample_cfg)
        else:
            self.downsample = None

    def forward(self, x, in_shape, do_downsample=True):
        for block in self.blocks:
            x = block(x, in_shape)

        if self.downsample is not None and do_downsample:
            x, out_shape = self.downsample(x, in_shape)
        else:
            out_shape = in_shape
        return x, out_shape

    @property
    def out_channels(self):
        if self.downsample:
            return self.downsample.out_channels
        else:
            return self.embed_dims



class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)
        
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    SSFAdaLayer(feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(SSFAdaLayer(embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
    


@BACKBONES.register_module()
class SwinTransformerSSF(SwinTransformer):

    def __init__(
        self,
        arch='base',
        img_size=224,
        patch_size=4,
        in_channels=3,
        window_size=7,
        drop_path_rate=0.1,
        with_cp=False,
        pad_small_map=False,
        stage_cfgs=dict(),
        patch_cfg=dict()
    ):
        super().__init__(arch=arch)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.stages = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth,
                num_heads) in enumerate(zip(self.depths, self.num_heads)):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i < self.num_layers - 1 else False
            _stage_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'window_size': window_size,
                'downsample': downsample,
                'drop_paths': dpr[:depth],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                **stage_cfg
            }
            _patch_cfg = dict(
                in_channels=in_channels,
                input_size=img_size,
                embed_dims=self.embed_dims,
                conv_type='Conv2d',
                kernel_size=patch_size,
                stride=patch_size,
                norm_cfg=dict(type='LN'),
            )
            _patch_cfg.update(patch_cfg)
            self.patch_embed = PatchEmbedSSF(**_patch_cfg)

            stage = SwinBlockSequenceSSF(**_stage_cfg)
            self.stages.append(stage)
            dpr = dpr[depth:]
            embed_dims.append(stage.out_channels)
        # for param in self.parameters():
        #     param.requires_grad = False
        for name, param in self.named_parameters():
            if 'ssf' in name:
                # import pdb; pdb.set_trace()
                continue
            else:
                param.requires_grad = False

        # self.prompt_layers = [0] if prompt_layers is None else prompt_layers
        # prompt = torch.empty(
        #     len(self.prompt_layers), prompt_length, self.embed_dims)
        # if prompt_init == 'uniform':
        #     nn.init.uniform_(prompt, -0.08, 0.08)
        # elif prompt_init == 'zero':
        #     nn.init.zeros_(prompt)
        # elif prompt_init == 'kaiming':
        #     nn.init.kaiming_normal_(prompt)
        # elif prompt_init == 'token':
        #     nn.init.zeros_(prompt)
        #     self.prompt_initialized = False
        # else:
        #     nn.init.normal_(prompt, std=0.02)
        # self.prompt = nn.Parameter(prompt, requires_grad=True)

        self.num_features = int(self.embed_dims * 2 ** (self.num_layers - 1))
        self.ssf_scale_1,self.ssf_shift_1 = init_ssf_scale_shift(self.num_features)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.absolute_pos_embed, self.patch_resolution, hw_shape,
                self.interpolate_mode, self.num_extra_tokens)
        x = self.drop_after_pos(x)

        # # Add prompt
        # if hasattr(self, 'prompt_initialized') and not self.prompt_initialized:
        #     with torch.no_grad():
        #         self.prompt.data += x.mean([0, 1]).detach().clone()
        #     self.prompt_initialized = True
        # prompt = self.prompt.unsqueeze(1).expand(-1, x.shape[0], -1, -1)
        # # prompt: [layer, batch, length, dim]
        # if self.prompt_pos == 'prepend':
        #     # x = torch.cat([x[:, :1, :], prompt[0, :, :, :], x[:, 1:, :]],
        #     #               dim=1)
        #     x = torch.cat([prompt[0, :, :, :], x], dim=1)
        #     # vpt_swin: (batch_size, n_prompt + n_patches, hidden_dim)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(
                x, hw_shape, do_downsample=self.out_after_downsample)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                # out = out.view(-1, *hw_shape,
                #                self.num_features[i]).permute(0, 3, 1,
                #                                              2).contiguous()
                out = ssf_ada(out, self.ssf_scale_1, self.ssf_shift_1)
                
                out = self.avgpool(out.transpose(1, 2))  # B C 1
                out = torch.flatten(out, 1)
                outs.append(out)
            if stage.downsample is not None and not self.out_after_downsample:
                x, hw_shape = stage.downsample(x, hw_shape)

        return tuple(outs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args,
                              **kwargs):
        """load checkpoints."""
        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and \
                self.__class__ is SwinTransformerSSF:
            final_stage_num = len(self.stages) - 1
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if k.startswith('norm.') or k.startswith('backbone.norm.'):
                    convert_key = k.replace('norm.', f'norm{final_stage_num}.')
                    state_dict[convert_key] = state_dict[k]
                    del state_dict[k]
        if (version is None or version < 3) and \
                self.__class__ is SwinTransformerSSF:
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if 'attn_mask' in k:
                    del state_dict[k]

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      *args, **kwargs)
