"""
BackBone: ViT
PET method: 
"""
import torch
from mmcls.models import BACKBONES
from mmcls.models.utils import resize_pos_embed
from typing import Sequence
from .vision_transformer import MedFMC_VisionTransformer
from typing import List
from mmcv.runner.base_module import BaseModule, ModuleList
import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmcls.models.utils import MultiheadAttention, resize_pos_embed
from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer



@BACKBONES.register_module()
class VisionTransformerFT(MedFMC_VisionTransformer):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = ModuleList()
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
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))
        for param in self.parameters():
            param.requires_grad = True


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


