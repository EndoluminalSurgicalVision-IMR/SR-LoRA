# ######progressive stochastic masking######

# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Changes made to the original code:
# 2022.08.20 - Integrate the DyLoRA layer for the LoRA Linear layer
\
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from typing import Optional, List

import torch
import torch.nn as nn
from loralib.layers import LoRALayer

#!/usr/bin/env python
# coding=utf-8


class psm_dynamic(nn.Module):
    def __init__(
        self,
        maximum_rank: int = 1,
    ):
        '''
        maximum_rank: maximum rank of the input matrix
        '''
        super(psm_dynamic, self).__init__()
        self.maximum_rank = maximum_rank

        self.frozen = False
        self.selected_ranks = []

    def get_dimension(self):
        return self.maximum_rank
    
    def set_psm_rank(self, selected_ranks):
        self.selected_ranks = list(selected_ranks)

    def forward(self, inputs, mode: bool = False):
        # inputs: (N, maximum_rank)
        if not hasattr(self, 'selected_ranks') or len(self.selected_ranks) == 0:
            return inputs.new_zeros((inputs.size(0), 0))
        if len(self.selected_ranks) != self.maximum_rank:
            raise ValueError("selected_ranks length must equal maximum_rank")
        idxs = [i for i, v in enumerate(self.selected_ranks) if v]
        
        if len(idxs) == 0:
            return inputs.new_zeros((inputs.size(0), 0))
        
        sel = inputs[:, idxs]
        scale = math.sqrt(self.maximum_rank / max(1, sel.size(1)))
        return sel * scale 
    
    
       
class psm_Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))

            self.nd_lora_A = psm_dynamic(maximum_rank=self.r)
            self.nd_lora_B = psm_dynamic(maximum_rank=self.r)

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
            
    def set_psm_mask(self, mask):
        self.nd_lora_A.set_psm_rank(mask)
        self.nd_lora_B.set_psm_rank(mask)
        
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    lora_A = self.nd_lora_A(self.lora_A.T, mode=mode).T
                    lora_B = self.nd_lora_B(self.lora_B, mode=mode)
                    self.weight.data -= T(lora_B @ lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    p_i = np.full(self.r, 0.5)
                    mask = np.random.binomial(1, p_i).astype(np.float32)
                    if mask.sum() == 0:
                        mask[np.random.randint(0, self.r)] = 1.0
                    self.nd_lora_A.set_psm_rank(mask)
                    self.nd_lora_B.set_psm_rank(mask)
                    lora_A = self.nd_lora_A(self.lora_A.T, mode=mode).T
                    lora_B = self.nd_lora_B(self.lora_B, mode=mode)
                    self.weight.data += T(lora_B @ lora_A) * self.scaling
                self.merged = True
            
    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                lora_A = self.nd_lora_A(self.lora_A.T, mode=self.training).T
                lora_B = self.nd_lora_B(self.lora_B, mode=self.training)
                result += (self.lora_dropout(x) @ lora_A.T @ lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        


    


class psm_MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):  
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.nd_lora_A = psm_dynamic(maximum_rank=r * sum(enable_lora))
            self.nd_lora_B = psm_dynamic(maximum_rank=r)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
            
    def set_psm_mask(self, mask):
        mask_expanded = mask.repeat(sum(self.enable_lora))
        self.nd_lora_A.set_psm_rank(mask_expanded)
        self.nd_lora_B.set_psm_rank(mask)
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result
    
    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        # if mode:
        lora_A = self.nd_lora_A(self.lora_A.T).T
        lora_B = self.nd_lora_B(self.lora_B)

            
        delta_w = F.conv1d(
            lora_A.unsqueeze(0), 
            lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        
        return T(self.zero_pad(delta_w))
    
    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
            
                    lora_A = self.nd_lora_A(self.lora_A.T, mode=mode).T
                    lora_B = self.nd_lora_B(self.lora_B, mode=mode)
            
                    delta_w = F.conv1d(
                        lora_A.unsqueeze(0), 
                        lora_B.unsqueeze(-1), 
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data -= T(self.zero_pad(delta_w)) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    p_i = np.full(self.r, 0.5)
                    mask = np.random.binomial(1, p_i).astype(np.float32)
                    if mask.sum() == 0:
                        mask[np.random.randint(0, self.r)] = 1.0
                    mask_expanded = mask.repeat(sum(self.enable_lora))
                    self.nd_lora_A.set_psm_rank(mask_expanded)
                    self.nd_lora_B.set_psm_rank(mask)
                    lora_A = self.nd_lora_A(self.lora_A.T, mode=mode).T
                    lora_B = self.nd_lora_B(self.lora_B, mode=mode)
                    delta_w = F.conv1d(
                        lora_A.unsqueeze(0), 
                        lora_B.unsqueeze(-1), 
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    
                    self.weight.data += T(self.zero_pad(delta_w)) * self.scaling
                    
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0 and any(self.enable_lora):
                lora_A = self.nd_lora_A(self.lora_A.T, mode=self.training).T
                lora_B = self.nd_lora_B(self.lora_B, mode=self.training)
                delta_w = F.conv1d(
                    lora_A.unsqueeze(0), 
                    lora_B.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).squeeze(0)
                result += (self.lora_dropout(x) @ self.zero_pad(delta_w).T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


