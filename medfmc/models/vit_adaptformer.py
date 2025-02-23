"""
Backbone: ViT
PET method: Adaptformer - <<Adaptformer: Adapting vision transformers for scalable visual recognition>>
Code: https://github.com/ShoufaChen/AdaptFormer
Paper: https://proceedings.neurips.cc/paper_files/paper/2022/file/69e2f49ab0837b71b0e0cb7c555990f8-Paper-Conference.pdf
"""

import torch
import torch.nn as nn
from mmcls.models import BACKBONES
from mmcls.models.backbones import VisionTransformer
from mmcls.models.utils import resize_pos_embed
from typing import List
from mmcv.runner.base_module import BaseModule, ModuleList
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence
import math
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmcls.models.utils import MultiheadAttention, resize_pos_embed
from timm.models.layers import DropPath


class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


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
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 ffn_option='parallel'):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.ffn_option = ffn_option

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias)
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate >0. else nn.Identity()
        
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)
        
        self.fc1 = nn.Linear(embed_dims,feedforward_channels)
        self.fc2 = nn.Linear(feedforward_channels,embed_dims)
        self.mlp_drop = nn.Dropout(drop_rate)
        self.activation = getattr(nn, act_cfg['type'])() if act_cfg else None

        self.adapter = Adapter(config=None, d_model=embed_dims, bottleneck=3072, dropout=0.1, init_option='lora')
   
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            add_identity=False,
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
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.ffn_option == 'parallel':
            adapt_x = self.adapter(x,add_residual=False)
        
        residual = x
        # x = self.mlp_drop(self.activation(self.fc1(self.norm2(x))))
        # x = self.mlp_drop(self.fc2(x))
        
        # x = self.mlp_drop(self.activation(self.fc1(self.norm2(x))))
        # x = self.drop_path(self.mlp_drop(self.fc2(x)))
        x = self.ffn(self.norm2(x))
        
        if self.ffn_option == 'sequential':
            x = self.adapter(x)*0.1
        elif self.ffn_option == 'parallel':
            x = x + adapt_x*0.1
        else:
            raise ValueError(self.ffn_adapt)
        # x = self.adapter(x)

        # import pdb; pdb.set_trace()
        # x = self.ffn(self.norm2(x), identity=x) + self.adapter(self.norm2(x)) * 0.001 ## SOTA
        # x = self.ffn(self.norm2(x), identity=x) ## Original
        
        # x = self.ffn(self.norm2(x), identity=x) + self.adapter(x) * 0.001
        x = residual + x
        return x

@BACKBONES.register_module()
class VitAdaptFormer(VisionTransformer):
    
    def __init__(self,
                 ffn_option: str = 'parallel',
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.layers = ModuleList()
        self.ffn_option = ffn_option
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=0.0,
                drop_path_rate=0.0,
                qkv_bias=True,
                norm_cfg=dict(type='LN', eps=1e-6))
            self.layers.append(TransformerEncoderLayer(**_layer_cfg,ffn_option=self.ffn_option))
        # import pdb; pdb.set_trace()
        # for param in self.parameters():
        #     # import pdb; pdb.set_trace()
        #     param.requires_grad = False
        for name, param in self.named_parameters():
            if 'adapter' in name:
                # import pdb; pdb.set_trace()
                continue
            else:
                param.requires_grad = False
        # import pdb; pdb.set_trace()
        self.ffn_option = ffn_option
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
        # self.prompt_length = prompt_length
        # self.prompt_pos = prompt_pos

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

        # # Add prompt
        # if hasattr(self, 'prompt_initialized') and not self.prompt_initialized:
        #     with torch.no_grad():
        #         self.prompt.data += x.mean([0, 1]).detach().clone()
        #     self.prompt_initialized = True
        # prompt = self.prompt.unsqueeze(1).expand(-1, x.shape[0], -1, -1)
        # # prompt: [layer, batch, length, dim]
        # if self.prompt_pos == 'prepend':
        #     x = torch.cat([x[:, :1, :], prompt[0, :, :, :], x[:, 1:, :]],
        #                   dim=1)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]
        outs = []
        for i, layer in enumerate(self.layers):
        #     if i in self.prompt_layers:
        #         if self.prompt_pos == 'prepend':
        #             x = torch.cat([
        #                 x[:, :1, :], prompt[i, :, :, :],
        #                 x[:, 1 + self.prompt_length:, :]
        #             ],
        #                           dim=1)
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                outs.append(x[:, 0])

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                if self.with_cls_token:
                    patch_token = x[:, 1:].reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = x[:, 0]
                else:
                    patch_token = x.reshape(B, *patch_resolution, C)
                    patch_token = patch_token.permute(0, 3, 1, 2)
                    cls_token = None

                if self.output_cls_token:
                    out = cls_token
                elif self.output_vithead:
                    out = [patch_token, cls_token]
                else:
                    raise NotImplementedError(
                        f'The output must be cls_token or [patch_token, cls_token]!'
                    )
                outs.append(out)


        return tuple(outs)