import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
import numpy as np
from mmcls.models import BACKBONES
from mmcls.models.backbones import VisionTransformer
from typing import List
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS
from typing import Sequence
from mmcls.models.utils import resize_pos_embed
from mmcv.cnn import (Linear, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.drop import build_dropout
from sklearn.metrics import roc_auc_score
import logging
import loralib as lora
from loralib.layers import LoRALayer
from .psm import psm_Linear, psm_MergedLinear


class MultiheadAttention(BaseModule):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 lora_rank=[8, 8],
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.qkv = psm_MergedLinear(self.input_dims, embed_dims * 3, r=lora_rank[0], enable_lora=[True, False, True])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = psm_Linear(self.embed_dims, self.embed_dims, bias=proj_bias, r=lora_rank[1])
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x

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
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
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

class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 lora_rank=[8, 8],
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            lora_rank = lora_rank,
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.ffn(self.norm2(x), identity=x)
        return x


@BACKBONES.register_module()
class VitSSR_LoRA(VisionTransformer):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # for imagnet21k
        self.lora_ranks = [
        [9, 10],
        [25, 35],
        [44, 54],
        [69, 78],
        [78, 86],
        [85, 94],
        [106, 69],
        [102, 53],
        [105, 21],
        [116, 73],
        [99, 85],
        [78, 42]
    ]
    # for dino
        # self.lora_ranks = [[7, 5],
        #                    [19, 18], 
        #                    [33, 16], 
        #                    [61, 33], 
        #                    [60, 52], 
        #                    [93, 71], 
        #                    [99, 78], 
        #                    [104, 90], 
        #                    [108, 81], 
        #                    [106, 45],
        #                    [130, 27], 
        #                    [153, 27]]
    # for MAE
        # self.lora_ranks = [
        #     [7, 9],
        #     [17, 54],
        #     [27, 77],
        #     [47, 89],
        #     [70, 61],
        #     [57, 65],
        #     [64, 88],
        #     [71,108],
        #     [78, 115],
        #     [94, 86],
        #     [90, 84],
        #     [106, 43]
        # ]
    # #for imagenet large
    #     self.lora_ranks =[
    #         [7, 5],
    #         [15, 22],
    #         [19, 44],
    #         [30, 73],
    #         [44, 52],
    #         [49, 62],
    #         [50, 59],
    #         [64, 104],
    #         [71, 129],
    #         [78, 100],
    #         [109, 114],
    #         [103, 116],
    #         [115, 134],
    #         [118, 128],
    #         [120, 115],
    #         [133, 98],
    #         [138, 78],
    #         [133, 124],
    #         [128, 158],
    #         [143, 104],
    #         [120, 118],
    #         [104, 69],
    #         [116, 61],
    #         [100, 45]
    #         ]
        
    # #for large mae:
        # self.lora_ranks = [
        #     [12,3],
        #     [38,35],
        #     [44,64],
        #     [59,61],
        #     [73,94],
        #     [52,64],
        #     [72,79],
        #     [57,99],
        #     [67,90],
        #     [82,96],
        #     [69,111],
        #     [93,85],
        #     [88,84],
        #     [86,109],
        #     [95,102],
        #     [103,102],
        #     [98,107],
        #     [99,94],
        #     [119,106],
        #     [121,107],
        #     [151,112],
        #     [123,85],
        #     [144,68],
        #     [104,34],
        # ]            
    #for large dino:
        # self.lora_ranks = [
        #     [4,56],
        #     [20,55],
        #     [29,75],
        #     [26,67],
        #     [48,91],
        #     [47,69],
        #     [61,121],
        #     [59,138],
        #     [79,130],
        #     [89,115],
        #     [107,118],
        #     [99,121],
        #     [113,100],
        #     [115,88],
        #     [120,75],
        #     [112,82],
        #     [126,83],
        #     [125,73],
        #     [108,86],
        #     [132,77],
        #     [134,74],
        #     [131,88],
        #     [86,45],
        #     [79,27]
        # ]
    # #for full:
    #     self.lora_ranks = [ 
    #         [768,768],
    #         [768,768],
    #         [768,768],
    #         [768,768],
    #         [768,768],
    #         [768,768],
    #         [768,768],
    #         [768,768],
    #         [768,768],
    #         [768,768],
    #         [768,768],
    #         [768,768]
    #     ]

        self.layers = ModuleList()
        for i in range(self.num_layers):
            _layer_cfg = dict(
                lora_rank=self.lora_ranks[i],
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=0.0,
                drop_path_rate=0.0,
                qkv_bias=True,
                norm_cfg=dict(type='LN', eps=1e-6))
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))
     
        for name, param in self.named_parameters():
            if 'lora' in name: 
                continue
            else:
                param.requires_grad = False

        self.psm_lora_layers = []
        self.psm_cur_p = 0.0
    def init_psm_lora(self):
        """initialize psm mask for LoRA layers."""
        self.psm_lora_layers = []
        def find_lora_layers(module):
            for child in module.children():
                if hasattr(child, 'r') and getattr(child, 'r', 0) > 0:
                    self.psm_lora_layers.append(child)
                    child.set_psm_rank(np.zeros(child.r, dtype=np.float32)) 
                find_lora_layers(child)
        find_lora_layers(self)

    def forward(self, x):
        """Following mmcls implementation."""
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                outs.append(x[:, 0])

        return tuple(outs)