from mmcls.models import BACKBONES
from medfmc.models.swin_transformer import MedFMC_SwinTransformer
import torch
import torch.nn as nn
import re


@BACKBONES.register_module()
class SwinTransformer_Bitfit(MedFMC_SwinTransformer):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


#################################################################
class BitFitConfig:
        def __init__(self):
            self.bitfit_modules = ".*"
            self.bitfit_layers = "q|k|v|o|w.*"
            self.trainable_param_names = ".*layer_norm.*|.*bias"
            # lora_modules and lora_layers are speicified with regular expressions
            # see https://www.w3schools.com/python/python_regex.asp for reference

@BACKBONES.register_module()
class Swin_Bitfit(MedFMC_SwinTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bitfit_modules = ".*"
        self.bitfit_layers = "q|k|v|o|w.*"
        self.trainable_param_names = ".*norm.*|.*bias"

        for m_name, module in dict(self.named_modules()).items():
            if re.fullmatch(self.bitfit_modules, m_name):
                for c_name, layer in dict(module.named_children()).items():
                    if re.fullmatch(self.bitfit_layers, c_name):
                        if hasattr(layer, 'bias'):
                            layer.bias = nn.Parameter(torch.zeros(layer.out_features))

        for p_name, param in dict(self.named_parameters()).items():
            if re.fullmatch(self.trainable_param_names, p_name):
                param.requires_grad = True
            else:
                param.requires_grad = False