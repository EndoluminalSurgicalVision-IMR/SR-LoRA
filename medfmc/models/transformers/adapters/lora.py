# Code adapted from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py.
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..configuration_utils import PretrainedConfig
from .composition import AdapterCompositionBlock
from .configuration import LoRAConfig
from .layer import AdapterLayerBase
from .modeling import LoraSubnet
from .tensor_buffer import TensorBuffer

# LAYER_NUM = 4
# DENSITY=0.5

class LoRA(nn.Module):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
        gating_heads: int = 1,
    ):
        super().__init__()
        self.r = config.r
        self.lora_alpha = config.alpha
        self.composition_mode = config.composition_mode
        self.attn_matrices = config.attn_matrices
        self.share_adapter = config.share_adapter
        self.use_gating = config.use_gating
        # Optional dropout
        if config.dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=config.dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        if self.r > 1 and self.composition_mode == "scale":
            raise ValueError("Can only use composition_mode='scale' when r == 1.")
        if self.r > 0:
            if self.composition_mode == "add":
                self.lora_A = nn.ModuleList([nn.Linear(lora_A_shape[0],lora_A_shape[1], bias=False) for _ in range(config.num_layer)])
            self.lora_B = nn.ModuleList([nn.Linear(lora_B_shape[0],lora_B_shape[1], bias=False) for _ in range(config.num_layer)])
            self.scaling = self.lora_alpha / self.r

            if self.use_gating:
                self.gate = nn.Linear(lora_A_shape[-1], gating_heads)

            if config.init_weights == "lora":
                # initialize A the same way as the default for nn.Linear and B to zero
                if self.composition_mode == "add":
                    nn.init.kaiming_uniform_(self.lora_A[0].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[0].weight)
                print('lora_shape', lora_A_shape[0],lora_A_shape[1], lora_B_shape[0],lora_B_shape[1])
                if self.use_gating:
                    nn.init.normal_(self.gate.weight, std=0.02)
            elif config.init_weights == "bert":
                if self.composition_mode == "add":
                    nn.init.normal_(self.lora_A[0], std=0.02)
                nn.init.normal_(self.lora_B[0], std=0.02)
                if self.use_gating:
                    nn.init.normal_(self.gate.weight, std=0.02)
            elif config.init_weights == "ia3":
                if self.composition_mode == "add":
                    nn.init.ones_(self.lora_A[0])
                nn.init.ones_(self.lora_B[0])
                if self.use_gating:
                    nn.init.normal_(self.gate.weight, std=0.02)
            else:
                raise ValueError("Unknown init_weights type: {}".format(config.init_weights))
            for i in range(1, config.num_layer):
                self.lora_A[i].weight.data.copy_(self.lora_A[0].weight.data)

    def com(self, weights: torch.Tensor, added: torch.Tensor, scaling=None) -> torch.Tensor:
        """Performs the composition operation between existing and injected weights."""
        if scaling is None:
            scaling = self.scaling
        if self.composition_mode == "add":
            return weights + added * scaling
        elif self.composition_mode == "scale":
            return weights * (added * scaling)
        else:
            raise ValueError("Invalid composition mode.")

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights."""
        if self.composition_mode == "add":
            return weights - added * self.scaling
        elif self.composition_mode == "scale":
            return weights / (added * self.scaling)
        else:
            raise ValueError("Invalid composition mode.")


class LoRALayer(AdapterLayerBase):
    def __init__(self, location_key: str, config: LoRAConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.location_key = location_key + "_lora"
        self.config = config
        self.loras = nn.ModuleDict(dict())
        self.lora_A_masks = {}
        self.lora_B_masks = {}
        self.lora_A_dense_tied_weights = {}
        self.lora_B_dense_tied_weights = {}
        self.adapters_mask = nn.ModuleDict(dict())
        self.adapter_name = ''
        self.merged = False
      

    def get_n_heads(self, lora: Union[LoRA, LoRAConfig]):
        return 1

    def _check_lora_location(self, config: LoRAConfig):
        return True

    def _get_lora_shapes(self, config: LoRAConfig):
        raise NotImplementedError()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        lora_config = self.config
        # lora_config = self.config.adapters.match(
        #     adapter_name,
        #     config_type=LoRAConfig,
        #     layer_idx=self.layer_idx,
        #     location_key=self.location_key,
        # )
        if lora_config is not None and self._check_lora_location(lora_config):
            lora = LoRA(
                *self._get_lora_shapes(lora_config),
                lora_config,
                gating_heads=self.get_n_heads(lora_config),
            )
            lora.train(self.training)
            self.loras[adapter_name] = lora
            self.init_masks_for_tied_weights(adapter_name, lora_config.density, sparsity_type=lora_config.sparsity_type)
            self.get_latest_dense_tied_weights(adapter_name)
            self.apply_masks_to_dense_tied_weights(adapter_name)
            self.adapter_name = adapter_name
            return True

        return False

    def add_shared_adapter(self, adapter_name: str, layer_idx: int, sparsity: float, shared_adapter=None, tasks=None):
        raise ValueError('not supported operation')
        status = self.add_adapter(adapter_name, layer_idx)
        if not status:
            return False
        # define mask
        mask = LoraSubnet(self.loras[adapter_name].lora_A.size(),
                          self.loras[adapter_name].lora_B.size(),
                          sparsity,
                          )

        self.adapters_mask[adapter_name] = mask

        return True

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.loras:
            del self.loras[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to lora

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to lora

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool, tasks=None):
        if unfreeze_adapters:
            for name in adapter_setup.flatten():
                if name in self.loras:
                    for param in self.loras[name].parameters():
                        param.requires_grad = True
                if name in self.adapters_mask:
                    for param in self.adapters_mask[name].parameters():
                        param.requires_grad = True

    def get_adapter(self, adapter_name: str) -> nn.Module:
        if adapter_name in self.loras:
            return self.loras[adapter_name]
        else:
            return None

    def init_masks_for_tied_weights(
        self, adapter_name, mask_density, rng_state=np.random.RandomState(0), sparsity_type='equal_per_layer', 
    ):  # net is a weight-tied block
        def _get_masks(layer):
            if sparsity_type == "equal_per_layer":
                mask = torch.zeros_like(layer.weight.view(-1))
                mask_length = len(mask)
                num_params_to_keep_per_layer = int(mask_length * mask_density)
                selected_keep_pos = rng_state.choice(
                    np.arange(mask_length), num_params_to_keep_per_layer, replace=False
                )
                mask[selected_keep_pos] = 1
                return mask.view(layer.weight.size())
            # support NVIDIA sparse tensor core: N:M sparsity
            elif sparsity_type == "NM_structured": 
                N, M = 2, 4
                grads_abs = torch.randn(layer.weight.shape)

                group = int(grads_abs.numel()/M)
                weight_temp = grads_abs.detach().reshape(group, M)
                index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]
                w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
                w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(grads_abs.shape)

                mask = (
                    w_b >= 1e-10
                ).float()
                return mask.view(layer.weight.size())
            else:
                raise NotImplementedError
        lora_config = self.config
        # lora_config = self.config.adapters.match(
        #     adapter_name,
        #     config_type=LoRAConfig,
        #     layer_idx=self.layer_idx,
        #     location_key=self.location_key,
        # )
        if adapter_name in self.loras:
            self.lora_A_masks[adapter_name] = []
            self.lora_B_masks[adapter_name] = []
            for i in range(lora_config.num_layer):
                self.lora_A_masks[adapter_name].append(_get_masks(self.loras[adapter_name].lora_A[i]))
                self.lora_B_masks[adapter_name].append(_get_masks(self.loras[adapter_name].lora_B[i]))

    def get_latest_dense_tied_weights(self, adapter_name):
        if adapter_name in self.loras:
            self.lora_A_dense_tied_weights[adapter_name] = TensorBuffer(
                self.loras[adapter_name].lora_A[0].weight.data.clone()
            ).buffer
            self.lora_B_dense_tied_weights[adapter_name] = TensorBuffer(
                self.loras[adapter_name].lora_B[0].weight.data.clone()
            ).buffer

        
    def restore_dense_tied_weights(self, adapter_name):
        if adapter_name in self.loras:
            lora_config = self.config
            # lora_config = self.config.adapters.match(
            #     adapter_name,
            #     config_type=LoRAConfig,
            #     layer_idx=self.layer_idx,
            #     location_key=self.location_key,
            # )
            for i in range(lora_config.num_layer):
            
                device = self.loras[adapter_name].lora_A[0].weight.device
                d_type = self.loras[adapter_name].lora_A[0].weight.dtype

                self.loras[adapter_name].lora_A[i].weight.data = (self.lora_A_dense_tied_weights[adapter_name].clone()).reshape(self.loras[adapter_name].lora_A[i].weight.data.shape).to(d_type).to(device)
                self.loras[adapter_name].lora_B[i].weight.data = (self.lora_B_dense_tied_weights[adapter_name].clone()).reshape(self.loras[adapter_name].lora_B[i].weight.data.shape).to(d_type).to(device)


    def aggregate_grads_for_tied_weights(self, adapter_name, avg_tied_grads=False):
        # extract grads.
        if adapter_name in self.loras:
            lora_config = self.config
            # lora_config = self.config.adapters.match(
            #     adapter_name,
            #     config_type=LoRAConfig,
            #     layer_idx=self.layer_idx,
            #     location_key=self.location_key,
            # )
            buffers = []
            for i in range(lora_config.num_layer):
                hp_grad = self.loras[adapter_name].lora_A[i].weight.grad.clone()
                tb = TensorBuffer(
                    [hp_grad]
                )
                buffers.append(tb.buffer)

            # aggregate grads.
            aggregated_grads = (
                sum(buffers) / len(buffers) if avg_tied_grads else sum(buffers)
            )

            # assign grads back (inplace).
            for i in range(lora_config.num_layer):
                grads = self.loras[adapter_name].lora_A[i].weight.grad
                tb = TensorBuffer(grads)
                tb.buffer = aggregated_grads.clone()
                tb.unpack(grads)

            #for B
            buffers = []
            for i in range(lora_config.num_layer):
                hp_grad = self.loras[adapter_name].lora_B[i].weight.grad.clone()
                tb = TensorBuffer(
                    [hp_grad]
                )
                buffers.append(tb.buffer)
            # aggregate grads.
            aggregated_grads = (
                sum(buffers) / len(buffers) if avg_tied_grads else sum(buffers)
            )
            # assign grads back (inplace).
            for i in range(lora_config.num_layer):
                grads = self.loras[adapter_name].lora_B[i].weight.grad
                tb = TensorBuffer(grads)
                tb.buffer = aggregated_grads.clone()
                tb.unpack(grads)


    def apply_masks_to_grads_of_tied_weights(self, adapter_name):
        if adapter_name in self.loras:
            lora_config = self.config
            # lora_config = self.config.adapters.match(
            #     adapter_name,
            #     config_type=LoRAConfig,
            #     layer_idx=self.layer_idx,
            #     location_key=self.location_key,
            # )
            for i in range(lora_config.num_layer):
                device = self.loras[adapter_name].lora_A[i].weight.device
                mask = self.lora_A_masks[adapter_name][i].to(device)
                if self.loras[adapter_name].lora_A[i].weight.grad is not None:
                    self.loras[adapter_name].lora_A[i].weight.grad.data.mul_(mask)
                mask = self.lora_B_masks[adapter_name][i].to(device)
                if self.loras[adapter_name].lora_B[i].weight.grad is not None:
                    self.loras[adapter_name].lora_B[i].weight.grad.data.mul_(mask)


    def apply_masks_to_dense_tied_weights(self, adapter_name):

        if adapter_name in self.loras:
            lora_config = self.config
            # lora_config = self.config.adapters.match(
            #     adapter_name,
            #     config_type=LoRAConfig,
            #     layer_idx=self.layer_idx,
            #     location_key=self.location_key,
            # )
            for i in range(lora_config.num_layer):
                mask = self.lora_A_masks[adapter_name][i].to(self.loras[adapter_name].lora_A[i].weight.device)
                self.loras[adapter_name].lora_A[i].weight.data.mul_(mask)
                mask = self.lora_B_masks[adapter_name][i].to(self.loras[adapter_name].lora_B[i].weight.device)
                self.loras[adapter_name].lora_B[i].weight.data.mul_(mask)


class Linear(LoRALayer, nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        config: LoRAConfig,
        attn_key: str = None,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs
    ):
        LoRALayer.__init__(self, location_key, config, in_features, out_features, **kwargs)

        self.attn_key = attn_key
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = torch.t(self.weight.data)

    def _check_lora_location(self, config: LoRAConfig):
        return self.attn_key is None or self.attn_key in config.attn_matrices

    def _get_lora_shapes(self, config: LoRAConfig):
        return (config.r, self.in_features), (self.out_features, config.r)

    def reset_adapter(self):
        def T(w):
            return torch.t(w) if self.fan_in_fan_out else w
        lora_config = self.config
        # lora_config = self.config.adapters.match(
        #     self.adapter_name,
        #     config_type=LoRAConfig,
        #     layer_idx=self.layer_idx,
        #     location_key=self.location_key,
        # )
        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            if lora.r > 0:
                if lora.composition_mode == "scale":
                    delta_w = T(lora.lora_B)
                    raise ValueError('not supported operation')
                else:
                    delta_w = T(lora.lora_B[0] @ lora.lora_A[0])
                    for i in range(1, lora_config.num_layer):
                        delta_w += T(lora.lora_B[i] @ lora.lora_A[i])
                self.weight.data = lora.com_inv(self.weight.data, delta_w)
            self.merged = None

    def _compute_adapted_weight(self, lora, scaling=None):
        def T(w):
            return torch.t(w) if self.fan_in_fan_out else w

        weight = self.weight
        # Merge the weights and mark it
        lora_config = self.config
        # lora_config = self.config.adapters.match(
        #     self.adapter_name,
        #     config_type=LoRAConfig,
        #     layer_idx=self.layer_idx,
        #     location_key=self.location_key,
        # )
        if lora.r > 0:
            if lora.composition_mode == "scale":
                delta_w = T(lora.lora_B)
                raise ValueError('not supported operation')
            
            else:
                delta_w = T(lora.lora_B[0] @ lora.lora_A[0])
                for i in range(1, lora_config.num_layer):
                    delta_w += T(lora.lora_B[i] @ lora.lora_A[i])
            weight = lora.com(weight, delta_w, scaling=scaling)

        return weight

    def merge_adapter(self, name: str):
        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                if lora.use_gating:
                    raise ValueError("Cannot merge LoRA layer with gating.")
                self.weight.data = self._compute_adapted_weight(lora)
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

    def forward(self, x: torch.Tensor):
        def T(w):
            return torch.transpose(w, -2, -1) if self.fan_in_fan_out else w
        lora_config = self.config
        # lora_config = self.config.adapters.match(
        #     self.adapter_name,
        #     config_type=LoRAConfig,
        #     layer_idx=self.layer_idx,
        #     location_key=self.location_key,
        # )
        if not self.merged:
            adapter_setup = self.get_active_setup(self.loras)
            if adapter_setup is not None:
                if len(adapter_setup) == 1:
                    lora = self.loras[adapter_setup[0]]
                    # result shape: <batch_size> x <seq_len> x <head_dim>
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    if lora.r > 0:
                        if lora.share_adapter:
                            assert 1==2
                            lora_A_mask, lora_B_mask = self.adapters_mask[adapter_setup[0]]()
                            lora_A_mat = lora.lora_A * lora_A_mask
                            lora_B_mat = lora.lora_B * lora_B_mask
                        else:
                            #for i in range(lora_config.num_layer):
                            #    lora.lora_B[i] = lora.lora_B[i].to(x.device)
                            #    lora.lora_A[i] = lora.lora_A[i].to(x.device)
                            #    print('device change', lora.lora_B[i].device)
                            lora_B_mat = lora.lora_B
                            lora_A_mat = lora.lora_A

                        if lora.composition_mode == "scale":
                            raise ValueError('not supported operation')
                            delta_w = lora_B_mat.view(1, 1, -1)
                        else:
                            delta_w = lora.lora_dropout(x) @ (lora_A_mat[0].weight) @ (lora_B_mat[0].weight)
                            for i in range(1,lora_config.num_layer):
                                delta_w += lora.lora_dropout(x) @ (lora_A_mat[i].weight) @ (lora_B_mat[i].weight)
                        if lora.use_gating:
                            raise ValueError('not supported operation')
                            gate = torch.sigmoid(lora.gate(x))
                            gate = torch.mean(gate, dim=1).unsqueeze(-1)
                            self._store_gating_score(adapter_setup[0], gate)
                        else:
                            gate = None
                        result = lora.com(result, delta_w, scaling=gate)
                    return result
                else:
                    raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with LoRA.")

        return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(LoRALayer, nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        config: LoRAConfig,
        fan_in_fan_out: bool = False,
        **kwargs
    ):
        LoRALayer.__init__(self, location_key, config, in_features, out_features, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def get_n_heads(self, lora: Union[LoRA, LoRAConfig]):
        return len(set(lora.attn_matrices))

    def _get_lora_shapes(self, config: LoRAConfig):
        n_heads = self.get_n_heads(config)
        return (config.r * n_heads, self.in_features), (
            self.out_features // 3 * n_heads,
            config.r,
        )

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        is_added = super().add_adapter(adapter_name, layer_idx)
        if is_added:
            lora_config = self.config
            # lora_config = lora_config = self.config.adapters.match(
            #     adapter_name,
            #     config_type=LoRAConfig,
            #     layer_idx=self.layer_idx,
            #     location_key=self.location_key,
            # )
            lora = self.loras[adapter_name]
            lora.enable_lora = [
                "q" in lora_config.attn_matrices,
                "k" in lora_config.attn_matrices,
                "v" in lora_config.attn_matrices,
            ]
            # Actual trainable parameters
            if any(lora.enable_lora):
                # Compute the indices
                lora.lora_ind = self.weight.new_zeros((self.out_features,), dtype=torch.bool).view(
                    len(lora.enable_lora), -1
                )
                lora.lora_ind[lora.enable_lora, :] = True
                lora.lora_ind = lora.lora_ind.view(-1)

    def pad(self, x, lora, fill_value=None):
        if fill_value is None:
            if lora.composition_mode == "add":
                fill_value = 0
            else:
                fill_value = 1
        result = x.new_full((*x.shape[:-1], self.out_features), fill_value)
        result = result.view(-1, self.out_features)
        result[:, lora.lora_ind] = x.reshape(-1, self.out_features // 3 * self.get_n_heads(lora))
        return result.view((*x.shape[:-1], self.out_features))

    def reset_adapter(self):
        def T(w):
            return w if self.fan_in_fan_out else torch.t(w)
        lora_config = self.config
        # lora_config = self.config.adapters.match(
        #     self.adapter_name,
        #     config_type=LoRAConfig,
        #     layer_idx=self.layer_idx,
        #     location_key=self.location_key,
        # )
        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            if lora.r > 0 and any(lora.enable_lora):
                if lora.composition_mode == "scale":
                    assert 1==2
                    delta_w = lora.lora_B
                else:
                    delta_w = F.conv1d(
                        lora.lora_A[0].data.unsqueeze(0), lora.lora_B[0].data.unsqueeze(-1), groups=sum(lora.enable_lora)
                    ).squeeze(0)
                    for i in range(1,lora_config.num_layer):
                        delta_w += F.conv1d(
                            lora.lora_A[i].data.unsqueeze(0), lora.lora_B[i].data.unsqueeze(-1), groups=sum(lora.enable_lora)
                        ).squeeze(0)
                # shape after transpose: <head_dim> x <head_dim * n_heads>
                delta_w = delta_w.transpose(-2, -1)
                self.weight.data = lora.com_inv(self.weight.data, T(self.pad(delta_w, lora)))
            self.merged = None

    def _compute_adapted_weight(self, name, lora):
        def T(w):
            return w if self.fan_in_fan_out else torch.t(w)

        weight = self.weight
        lora_config = self.config
        # lora_config = self.config.adapters.match(
        #     self.adapter_name,
        #     config_type=LoRAConfig,
        #     layer_idx=self.layer_idx,
        #     location_key=self.location_key,
        # )
        if lora.r > 0:
            if lora.composition_mode == "scale":
                assert 1==2
                delta_w = lora.lora_B
            else:
                delta_w = F.conv1d(
                    lora.lora_A[0].data.unsqueeze(0), lora.lora_B[0].data.unsqueeze(-1), groups=sum(lora.enable_lora)
                ).squeeze(0)
                for i in range(1,lora_config.num_layer):
                    delta_w += F.conv1d(
                        lora.lora_A[i].data.unsqueeze(0), lora.lora_B[i].data.unsqueeze(-1), groups=sum(lora.enable_lora)
                    ).squeeze(0)
            # shape after transpose: <head_dim> x <head_dim * n_heads>
            delta_w = delta_w.transpose(-2, -1)
            weight = lora.com(weight, T(self.pad(delta_w, lora)))

        return weight

    def merge_adapter(self, name: str):
        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                if lora.use_gating:
                    raise ValueError("Cannot merge LoRA layer with gating.")
                self.weight.data = self._compute_adapted_weight(name, lora)
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

    def forward(self, x: torch.Tensor):
        def T(w):
            return torch.t(w) if self.fan_in_fan_out else w
        lora_config = self.config

        # lora_config = self.config.adapters.match(
        #     self.adapter_name,
        #     config_type=LoRAConfig,
        #     layer_idx=self.layer_idx,
        #     location_key=self.location_key,
        # )
        if not self.merged:
            adapter_setup = self.get_active_setup(self.loras)
            if adapter_setup is not None:
                if len(adapter_setup) == 1:
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    lora = self.loras[adapter_setup[0]]
                    if lora.r > 0:
                        if lora.composition_mode == "scale":
                            raise ValueError('not supported operation')
                            delta_w = lora.lora_B.view(1, 1, -1)
                        else:
                            after_A = F.linear(lora.lora_dropout(x), lora.lora_A[0])
                            after_B = F.conv1d(
                                after_A.transpose(-2, -1), lora.lora_B[0].unsqueeze(-1), groups=sum(lora.enable_lora)
                            ).transpose(-2, -1)
                            delta_w = after_B
                            for i in range(1,lora_config.num_layer):
                                after_A = F.linear(lora.lora_dropout(x), lora.lora_A[i])
                                after_B = F.conv1d(
                                    after_A.transpose(-2, -1), lora.lora_B[i].unsqueeze(-1), groups=sum(lora.enable_lora)
                                ).transpose(-2, -1)
                                delta_w += after_B
                        if lora.use_gating:
                            raise ValueError('not supported operation')
                            gate = torch.sigmoid(lora.gate(x))
                            gate = torch.mean(gate, dim=1)
                            self._store_gating_score(adapter_setup[0], gate)
                            gate = self.pad(
                                gate.repeat_interleave(self.out_features // 3, dim=-1), lora, fill_value=1
                            ).unsqueeze(1)
                        else:
                            gate = None
                        # result = (batch_size, seq_len, head_dim * 3)
                        result = lora.com(result, self.pad(delta_w, lora), scaling=gate)
                    return result
                else:
                    raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with LoRA.")

        return F.linear(x, T(self.weight), bias=self.bias)
