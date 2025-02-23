"""
BackBone: ViT
PET method: BitFit
"""

import torch
from mmcls.models import BACKBONES
from .vision_transformer import MedFMC_VisionTransformer
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer
import torch
import torch.nn as nn
import re


# @BACKBONES.register_module()
# class VisionTransformer_Bitfit(MedFMC_VisionTransformer):

#     def __init__(self,
#                  *args,
#                  **kwargs):
#         super().__init__(*args, **kwargs)
#         self.layers = ModuleList()
#         for i in range(self.num_layers):
#             _layer_cfg = dict(
#                 embed_dims=self.embed_dims,
#                 num_heads=self.arch_settings['num_heads'],
#                 feedforward_channels=self.
#                 arch_settings['feedforward_channels'],
#                 drop_rate=0.0,
#                 drop_path_rate=0.0,
#                 qkv_bias=True,
#                 norm_cfg=dict(type='LN', eps=1e-6))
#             self.layers.append(TransformerEncoderLayer(**_layer_cfg))
       
#         for name, param in self.named_parameters():
#             if 'bias' in name:
#                 param.requires_grad = True
#                 continue
#             else:
#                 param.requires_grad = False
      

  
# def modify_with_bitfit(transformer, config):
#     for m_name, module in dict(transformer.named_modules()).items():
#         if re.fullmatch(config.bitfit_modules, m_name):
#             for c_name, layer in dict(module.named_children()).items():
#                 if re.fullmatch(config.bitfit_layers, c_name):
#                     layer.bias = nn.Parameter(torch.zeros(layer.out_features))
#     return transformer


#################################################################
class BitFitConfig:
        def __init__(self):
            self.bitfit_modules = ".*"
            self.bitfit_layers = "q|k|v|o|w.*"
            self.trainable_param_names = ".*layer_norm.*|.*bias"
            # lora_modules and lora_layers are speicified with regular expressions
            # see https://www.w3schools.com/python/python_regex.asp for reference

@BACKBONES.register_module()
class ViT_Bitfit(MedFMC_VisionTransformer):
    def __init__(self, *args, **kwargs):
        super(ViT_Bitfit, self).__init__(*args, **kwargs)
        self.bitfit_modules = ".*"
        self.bitfit_layers = "q|k|v|o|w.*"
        self.trainable_param_names = ".*layer_norm.*|.*bias"

        for m_name, module in dict(self.named_modules()).items():
            if re.fullmatch(self.bitfit_modules, m_name):
                for c_name, layer in dict(module.named_children()).items():
                    if re.fullmatch(self.bitfit_layers, c_name):
                        layer.bias = nn.Parameter(torch.zeros(layer.out_features))

        for p_name, param in dict(self.named_parameters()).items():
            if re.fullmatch(self.trainable_param_names, p_name):
                param.requires_grad = True
            else:
                param.requires_grad = False