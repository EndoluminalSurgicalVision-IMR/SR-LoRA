from abc import ABC, abstractmethod
from typing import List, Mapping, Union

import numpy as np
import torch
from torch import nn
from .tensor_buffer import TensorBuffer

from .composition import (
    AdapterCompositionBlock,
    BatchSplit,
    Fuse,
    Parallel,
    Split,
    Stack,
)
from .configuration import AdapterConfig
from .context import AdapterSetup, ForwardContext
from .modeling import Adapter, AdapterSubnet, BertFusion, ParallelAdapter


class AdapterLayerBase(ABC):
    """
    Base class for all adaptation methods that require per-layer modules.
    """

    @property
    def layer_idx(self):
        return getattr(self, "_layer_idx", -1)

    @layer_idx.setter
    def layer_idx(self, layer_idx):
        idx = getattr(self, "_layer_idx", layer_idx)
        assert idx == layer_idx
        setattr(self, "_layer_idx", idx)

    def get_active_setup(self, module_dict):
        if getattr(self.config, "is_adaptable", False):
            # First check current context before falling back to defined setup
            context = AdapterSetup.get_context()
            if context is not None:
                adapter_setup = context.adapter_setup
            else:
                adapter_setup = self.config.adapters.active_setup
        else:
            adapter_setup = None
        skip_adapters = adapter_setup is None or (
            self.config.adapters.skip_layers is not None and self.layer_idx in self.config.adapters.skip_layers
        )
        if not skip_adapters and (len(set(module_dict.keys()) & adapter_setup.flatten()) > 0):
            return adapter_setup
        else:
            return None

    def _store_gating_score(self, adapter_name, gating_score):
        context = ForwardContext.get_context()
        if context.output_adapter_gating_scores:
            gating_cache = context.adapter_gating_scores
            if self.layer_idx not in gating_cache[adapter_name]:
                gating_cache[adapter_name][self.layer_idx] = {}
            gating_score = gating_score.detach().squeeze().cpu().numpy()
            if len(gating_score.shape) == 0:
                gating_score = np.expand_dims(gating_score, axis=0)
            cache_score = gating_cache[adapter_name][self.layer_idx].get(self.location_key, None)
            if cache_score is not None:
                gating_cache[adapter_name][self.layer_idx][self.location_key] = np.column_stack(
                    (cache_score, gating_score)
                )
            else:
                gating_cache[adapter_name][self.layer_idx][self.location_key] = gating_score

    def _store_fusion_attentions(self, fusion_name, attentions):
        context = ForwardContext.get_context()
        if context.output_adapter_fusion_attentions:
            attention_cache = context.adapter_fusion_attentions
            if self.layer_idx not in attention_cache[fusion_name]:
                attention_cache[fusion_name][self.layer_idx] = {}
            attention_cache[fusion_name][self.layer_idx][self.location_key] = attentions

    @abstractmethod
    def add_adapter(self, adapter_name: str, layer_idx: int):
        raise NotImplementedError()
    
    @abstractmethod
    def add_shared_adapter(self, adapter_name: str, layer_idx: int, sparsity: float, shared_adapter, tasks):
        raise NotImplementedError()

    @abstractmethod
    def delete_adapter(self, adapter_name: str):
        raise NotImplementedError()

    @abstractmethod
    def add_fusion_layer(self, adapter_names: Union[List, str]):
        raise NotImplementedError()

    @abstractmethod
    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        raise NotImplementedError()

    @abstractmethod
    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool, tasks=None):
        raise NotImplementedError()

    @abstractmethod
    def get_adapter(self, adapter_name: str) -> nn.Module:
        raise NotImplementedError()


class AdapterLayer(AdapterLayerBase, nn.Module):
    def __init__(self, location_key: str, config):
        super().__init__()
        self.location_key = location_key
        self.config = config
        self.masks_for_tied_weights = {}
        self.dense_tied_weights = {}

    def _init_adapter_modules(self):
        self.adapters = nn.ModuleDict(dict())
        self.adapters_mask = nn.ModuleDict(dict())
        self.adapter_fusion_layer = nn.ModuleDict(dict())

    def add_adapter(self, adapter_name: str, layer_idx: int):
        self.layer_idx = layer_idx
        adapter_config = self.config.adapters.match(
            adapter_name,
            config_type=AdapterConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if adapter_config is not None:
            reduction_factor = adapter_config["reduction_factor"]
            if isinstance(reduction_factor, Mapping):
                if str(self.layer_idx) in reduction_factor:
                    reduction_factor = reduction_factor[str(self.layer_idx)]
                elif "default" in reduction_factor:
                    reduction_factor = reduction_factor["default"]
                else:
                    raise KeyError(
                        "The given reduction factor mapping does not give a default value and does not specify each "
                        "reduction factor individually. You need to provide a default value like this: "
                        '{"1": 16, "default": 16}'
                    )

            if adapter_config.is_parallel:
                adapter_class = ParallelAdapter
            else:
                adapter_class = Adapter
            adapter = adapter_class(
                adapter_name=adapter_name,
                input_size=self.config.hidden_size,
                down_sample=int(self.config.hidden_size // reduction_factor),
                config=adapter_config,
            )
            adapter.train(self.training)  # make sure training mode is consistent
            self.adapters[adapter_name] = adapter
            self.adapter_name = adapter_name
            # if adapter_config.num_layer > 1:
            self.init_masks_for_tied_weights(adapter_name, adapter_config.density)
            self.get_latest_dense_tied_weights(adapter_name)
            self.apply_masks_to_dense_tied_weights(adapter_name)

    def add_shared_adapter(self, adapter_name: str, layer_idx: int, sparsity: float, shared_adapter, tasks=None):
        self.layer_idx = layer_idx
        adapter_config = self.config.adapters.match(
            adapter_name,
            config_type=AdapterConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if adapter_config is not None:
            reduction_factor = adapter_config["reduction_factor"]
            if isinstance(reduction_factor, Mapping):
                if str(self.layer_idx) in reduction_factor:
                    reduction_factor = reduction_factor[str(self.layer_idx)]
                elif "default" in reduction_factor:
                    reduction_factor = reduction_factor["default"]
                else:
                    raise KeyError(
                        "The given reduction factor mapping does not give a default value and does not specify each "
                        "reduction factor individually. You need to provide a default value like this: "
                        '{"1": 16, "default": 16}'
                    )
            
            # define mask for the adapter
            input_size = self.config.hidden_size
            output_size = int(self.config.hidden_size // reduction_factor)

            if tasks is not None:
                for task in tasks:
                    adapters_mask = AdapterSubnet(adapter_name, input_size, output_size, sparsity)
                    adapters_mask.train(self.training)  # make sure training mode is consistent
                    self.adapters_mask[task] = adapters_mask
            else:
                adapters_mask = AdapterSubnet(adapter_name, input_size, output_size, sparsity)
                adapters_mask.train(self.training)  # make sure training mode is consistent
                self.adapters_mask[adapter_name] = adapters_mask

            self.adapters[adapter_name] = shared_adapter

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.adapters:
            del self.adapters[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        """See BertModel.add_fusion_layer"""
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(",")
        if self.config.adapters.common_config_value(adapter_names, self.location_key):
            fusion_config = self.config.adapters.get_fusion(adapter_names)
            fusion = BertFusion(
                fusion_config,
                self.config.hidden_size,
                self.config.attention_probs_dropout_prob,
            )
            fusion.train(self.training)  # make sure training mode is consistent
            self.adapter_fusion_layer[",".join(adapter_names)] = fusion

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        adapter_names = adapter_names if isinstance(adapter_names, str) else ",".join(adapter_names)
        if adapter_names in self.adapter_fusion_layer:
            del self.adapter_fusion_layer[adapter_names]

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool, tasks=None):
        """
        Unfreezes a given list of adapters, the adapter fusion layer, or both

        Args:
            adapter_names: names of adapters to unfreeze (or names of adapters part of the fusion layer to unfreeze)
            unfreeze_adapters: whether the adapter weights should be activated
            unfreeze_fusion: whether the adapter fusion layer for the given adapters should be activated
        """
        if unfreeze_adapters:
            for adapter_name in adapter_setup.flatten():
                if adapter_name in self.adapters:
                    for param in self.adapters[adapter_name].parameters():
                        param.requires_grad = True
                if adapter_name in self.adapters_mask:
                    for param in self.adapters_mask[adapter_name].parameters():
                        param.requires_grad = True
            if tasks is not None:
                for task in tasks:
                    if task in self.adapters_mask:
                        for param in self.adapters_mask[task].parameters():
                            param.requires_grad = True

        if unfreeze_fusion:
            if isinstance(adapter_setup, Fuse):
                if adapter_setup.name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[adapter_setup.name].parameters():
                        param.requires_grad = True
            for sub_setup in adapter_setup:
                if isinstance(sub_setup, Fuse):
                    if sub_setup.name in self.adapter_fusion_layer:
                        for param in self.adapter_fusion_layer[sub_setup.name].parameters():
                            param.requires_grad = True

    def get_adapter(self, adapter_name):
        if adapter_name in self.adapters:
            return self.adapters[adapter_name]
        else:
            return None
    
    def init_masks_for_tied_weights(self, adapter_name, mask_density, rng_state=np.random.RandomState(0), sparsity_type='equal_per_layer'):
        def _get_masks(layer):
            masks = {}
            if sparsity_type == "equal_per_layer":
                for layer_idx in range(len(layer)):
                    module = layer[layer_idx]
                    if isinstance(module, nn.Linear):
                        mask = torch.zeros_like(module.weight.view(-1))
                        mask_length = len(mask)
                        num_params_to_keep_per_layer = int(mask_length * mask_density)
                        selected_keep_pos = rng_state.choice(
                            np.arange(mask_length), num_params_to_keep_per_layer, replace=False
                        )
                        mask[selected_keep_pos] = 1
                        masks[layer_idx] = mask.view(module.weight.size())
                return masks
            else:
                raise NotImplementedError
        
        if adapter_name in self.adapters:
            self.masks_for_tied_weights[adapter_name] = []
            adapter_module = self.adapters[adapter_name].adapter_module
            for i in range(len(adapter_module)):
                masks = _get_masks(adapter_module[i])
                self.masks_for_tied_weights[adapter_name].append(masks)

    def get_latest_dense_tied_weights(self, adapter_name):
        self.dense_tied_weights[adapter_name] = TensorBuffer(
            list(self.adapters[adapter_name].adapter_module[0].parameters())
        ).buffer

    def restore_dense_tied_weights(self, adapter_name):
        if adapter_name in self.adapters:
            for i in range(len(self.adapters[adapter_name].adapter_module)):
                module = self.adapters[adapter_name].adapter_module[i]
                params = list(module.parameters())
                tb = TensorBuffer(params)
                tb.buffer = self.dense_tied_weights[adapter_name].clone()
                tb.unpack(params)

    def aggregate_grads_for_tied_weights(self, adapter_name, avg_tied_grads=False):
        # extract grads
        if adapter_name in self.adapters:
            adapter_module = self.adapters[adapter_name].adapter_module
            # for adapter_down
            buffers = []
            for i in range(len(adapter_module)):
                hp_grad = adapter_module[i][0].weight.grad.clone()
                tb = TensorBuffer([hp_grad])
                buffers.append(tb.buffer)
            
            # aggregate grads
            aggregated_grads = (
                sum(buffers) / len(buffers) if avg_tied_grads else sum(buffers)
            )
            # assign grads back (inplace)
            for i in range(len(adapter_module)):
                grads = adapter_module[i][0].weight.grad
                tb = TensorBuffer(grads)
                tb.buffer = aggregated_grads.clone()
                tb.unpack(grads)

            # for adapter_up
            buffers = []
            for i in range(len(adapter_module)):
                hp_grad = adapter_module[i][2].weight.grad.clone()
                tb = TensorBuffer([hp_grad])
                buffers.append(tb.buffer)
            
            # aggregate grads
            aggregated_grads = (
                sum(buffers) / len(buffers) if avg_tied_grads else sum(buffers)
            )
            # assign grads back (inplace)
            for i in range(len(adapter_module)):
                grads = adapter_module[i][2].weight.grad
                tb = TensorBuffer(grads)
                tb.buffer = aggregated_grads.clone()
                tb.unpack(grads)

    def apply_masks_to_grads_of_tied_weights(self, adapter_name):
        if adapter_name in self.adapters:
            adapter_module = self.adapters[adapter_name].adapter_module
            for i in range(len(adapter_module)):
                module = adapter_module[i] # nn.Sequential
                device = module[0].weight.device
                # adapter_down
                down_mask = self.masks_for_tied_weights[adapter_name][i][0].to(device)
                if module[0].weight.grad is not None:
                    module[0].weight.grad.data.mul_(down_mask)
                up_mask = self.masks_for_tied_weights[adapter_name][i][2].to(device)
                if module[2].weight.grad is not None:
                    module[2].weight.grad.data.mul_(up_mask)

    def apply_masks_to_dense_tied_weights(self, adapter_name):
        if adapter_name in self.adapters:
            adapter_module = self.adapters[adapter_name].adapter_module
            for i in range(len(adapter_module)):
                device = adapter_module[i][0].weight.device
                masks = self.masks_for_tied_weights[adapter_name][i]
                for layer_idx in range(len(adapter_module[i])):
                    module = adapter_module[i][layer_idx]
                    if isinstance(module, nn.Linear):
                        module.weight.data.mul_(masks[layer_idx].to(device))

    def adapter_stack(self, adapter_setup: Stack, hidden_states, input_tensor, layer_norm, lvl=0, task=None):
        """
        Forwards the given input through the given stack of adapters.
        """
        for i, adapter_stack_layer in enumerate(adapter_setup):
            # Break if setup is too deep
            if isinstance(adapter_stack_layer, AdapterCompositionBlock) and lvl >= 1:
                raise ValueError(
                    "Specified adapter setup is too deep. Cannot have {} at level {}".format(
                        adapter_stack_layer.__class__.__name__, lvl
                    )
                )
            # Case 1: We have a nested fusion layer -> call fusion method
            if isinstance(adapter_stack_layer, Fuse):
                hidden_states = self.adapter_fusion(
                    adapter_stack_layer, hidden_states, input_tensor, layer_norm, lvl=lvl + 1
                )
            # Case 2: We have a nested split layer -> call split method
            elif isinstance(adapter_stack_layer, Split):
                hidden_states = self.adapter_split(
                    adapter_stack_layer, hidden_states, input_tensor, layer_norm, lvl=lvl + 1
                )
            # Case 3: We have a nested parallel layer -> call parallel method
            elif isinstance(adapter_stack_layer, Parallel):
                hidden_states, input_tensor = self.adapter_parallel(
                    adapter_stack_layer, hidden_states, input_tensor, layer_norm, lvl=lvl + 1
                )
            # Case 4: We have a nested batch split block -> call batchsplit method
            elif isinstance(adapter_stack_layer, BatchSplit):
                hidden_states = self.adapter_batchsplit(
                    adapter_stack_layer, hidden_states, input_tensor, layer_norm, lvl=lvl + 1
                )
            # Case 5: We have a single adapter which is part of this module -> forward pass
            elif adapter_stack_layer in self.adapters:
                adapter_layer = self.adapters[adapter_stack_layer]
                hidden_states, _, residual = adapter_layer.pre_forward(hidden_states, input_tensor, layer_norm)
                context = ForwardContext.get_context()

                if adapter_stack_layer in self.adapters_mask or (task is not None and task in self.adapters_mask):
                    # shared
                    if task is not None:
                        down_mask, up_mask = self.adapters_mask[task]()
                    else:
                        down_mask, up_mask = self.adapters_mask[adapter_stack_layer]()

                    layer_output = adapter_layer.forward_with_mask(
                        hidden_states, down_mask, up_mask, residual_input=residual, output_gating=context.output_adapter_gating_scores
                    )
                else:
                    layer_output = adapter_layer(
                        hidden_states, residual_input=residual, output_gating=context.output_adapter_gating_scores
                    )

                hidden_states, up = layer_output[0], layer_output[1]
                self._store_gating_score(adapter_stack_layer, layer_output[-1])
                # as this stack might be part of a fusion block, return the adapter up-projection output here
                # together with the final output (with potential residuals & norms) if we reached the last block of the stack
                if i == len(adapter_setup) - 1:
                    return hidden_states, up, input_tensor
            # Case X: No adapter which is part of this module -> ignore

        # If we got here, we either had another nested composition block
        # or no adapter was found. In both cases, we don't need to set the second return value for fusion
        return hidden_states, None, input_tensor

    def adapter_fusion(self, adapter_setup: Fuse, hidden_states, input_tensor, layer_norm, lvl=0):
        """
        Performs adapter fusion with the given adapters for the given input.
        """
        context = ForwardContext.get_context()

        # config of _last_ fused adapter is significant
        fusion_config = self.config.adapters.get_fusion(adapter_setup.name)
        last_adapter = self.adapters[adapter_setup.last()]
        hidden_states, query, residual = last_adapter.pre_forward(
            hidden_states, input_tensor, layer_norm, fusion_config=fusion_config
        )

        up_list = []

        for adapter_block in adapter_setup:
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                _, up, _ = self.adapter_stack(adapter_block, hidden_states, input_tensor, layer_norm, lvl=lvl + 1)
                if up is not None:  # could be none if stack is empty
                    up_list.append(up)
            # Case 2: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapters:
                adapter_layer = self.adapters[adapter_block]
                layer_output = adapter_layer(
                    hidden_states, residual_input=residual, output_gating=context.output_adapter_gating_scores
                )
                up = layer_output[2]
                self._store_gating_score(adapter_block, layer_output[-1])
                up_list.append(up)
            # Case 3: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore

        if len(up_list) > 0:
            up_list = torch.stack(up_list)
            up_list = up_list.permute(1, 2, 0, 3)

            fusion_output = self.adapter_fusion_layer[adapter_setup.name](
                query,
                up_list,
                up_list,
                residual,
                output_attentions=context.output_adapter_fusion_attentions,
            )
            if context.output_adapter_fusion_attentions:
                hidden_states = fusion_output[0]
                self._store_fusion_attentions(adapter_setup.name, fusion_output[-1])
            else:
                hidden_states = fusion_output

        return hidden_states

    def adapter_split(self, adapter_setup: Split, hidden_states, input_tensor, layer_norm, lvl=0):
        """
        Splits the given input between the given adapters.
        """
        # config of _first_ of splitted adapters is significant
        first_adapter = self.adapters[adapter_setup.first()]
        hidden_states, query, residual = first_adapter.pre_forward(hidden_states, input_tensor, layer_norm)

        # split hidden representations and residuals at split index
        split_hidden_states = [
            hidden_states[:, : adapter_setup.split_index, :],
            hidden_states[:, adapter_setup.split_index :, :],
        ]
        split_input_tensor = [
            input_tensor[:, : adapter_setup.split_index, :],
            input_tensor[:, adapter_setup.split_index :, :],
        ]
        split_residual = [
            residual[:, : adapter_setup.split_index, :],
            residual[:, adapter_setup.split_index :, :],
        ]

        for i, adapter_block in enumerate(adapter_setup):
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                split_hidden_states[i], _, _ = self.adapter_stack(
                    adapter_block, split_hidden_states[i], split_input_tensor[i], layer_norm, lvl=lvl + 1
                )
            # Case 2: We have a nested split -> recursively call split
            elif isinstance(adapter_block, Split):
                split_hidden_states[i] = self.adapter_split(
                    adapter_block, split_hidden_states[i], split_input_tensor[i], layer_norm, lvl=lvl + 1
                )
            # Case 3: We have a nested batch split -> call batch split method
            elif isinstance(adapter_block, BatchSplit):
                split_hidden_states[i] = self.adapter_batchsplit(
                    adapter_block, split_hidden_states[i], split_input_tensor[i], layer_norm, lvl=lvl + 1
                )
            # Case 4: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapters:
                adapter_layer = self.adapters[adapter_block]
                context = ForwardContext.get_context()
                layer_output = adapter_layer(
                    split_hidden_states[i],
                    residual_input=split_residual[i],
                    output_gating=context.output_adapter_gating_scores,
                )
                split_hidden_states[i] = layer_output[0]
                self._store_gating_score(adapter_block, layer_output[-1])
            # Case 5: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore

        hidden_states = torch.cat(split_hidden_states, dim=1)
        return hidden_states

    def adapter_parallel(self, adapter_setup: Parallel, hidden_states, input_tensor, layer_norm, lvl=0):
        """
        For parallel execution of the adapters on the same input. This means that the input is repeated N times before
        feeding it to the adapters (where N is the number of adapters).
        """

        context = ForwardContext.get_context()
        if not context.adapters_parallelized:
            orig_batch_size = input_tensor.shape[0]
            input_tensor = input_tensor.repeat(self.config.adapters.active_setup.parallel_channels, 1, 1)
            hidden_states = hidden_states.repeat(self.config.adapters.active_setup.parallel_channels, 1, 1)
            context.adapters_parallelized = True
        else:
            # The base model should handle replication of input.
            # Therefore, we assume the (replicated) input batch to be divisible by the number of parallel channels.
            if hidden_states.shape[0] % adapter_setup.parallel_channels != 0:
                raise ValueError(
                    "The total input batch size in a Parallel adapter block must be divisible by the number of"
                    " parallel channels."
                )
            orig_batch_size = hidden_states.shape[0] // adapter_setup.parallel_channels

        # We assume all adapters have the same config
        first_adapter = self.adapters[adapter_setup.first()]
        hidden_states, _, residual = first_adapter.pre_forward(hidden_states, input_tensor, layer_norm)

        # sequentially feed different parts of the blown-up batch into different adapters
        children_hidden = []
        for i, child in enumerate(adapter_setup):
            # Case 1: We have a nested stack -> call stack method
            if isinstance(child, Stack):
                child_hidden_states, _, _ = self.adapter_stack(
                    child,
                    hidden_states[i * orig_batch_size : (i + 1) * orig_batch_size],
                    input_tensor[i * orig_batch_size : (i + 1) * orig_batch_size],
                    layer_norm,
                    lvl=lvl + 1,
                )
                children_hidden.append(child_hidden_states)
            # Case 2. We have a nested batchsplit block -> call batchsplit method
            elif isinstance(child, BatchSplit):
                child_hidden_states = self.adapter_batchsplit(
                    child,
                    hidden_states[i * orig_batch_size : (i + 1) * orig_batch_size],
                    input_tensor[i * orig_batch_size : (i + 1) * orig_batch_size],
                    layer_norm,
                    lvl=lvl + 1,
                )
                children_hidden.append(child_hidden_states)
            # Case 3: We have a single adapter which is part of this module -> forward pass
            elif child in self.adapters:
                adapter_layer = self.adapters[child]
                context = ForwardContext.get_context()
                layer_output = adapter_layer(
                    hidden_states[i * orig_batch_size : (i + 1) * orig_batch_size],
                    residual_input=residual[i * orig_batch_size : (i + 1) * orig_batch_size],
                    output_gating=context.output_adapter_gating_scores,
                )
                child_hidden_states = layer_output[0]
                self._store_gating_score(child, layer_output[-1])
                children_hidden.append(child_hidden_states)
            # Case 4: nesting other composition blocks is invalid
            elif isinstance(child, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        child.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore
            else:
                children_hidden.append(hidden_states[i * orig_batch_size : (i + 1) * orig_batch_size])

        # concatenate all outputs and return
        hidden_states = torch.cat(children_hidden, 0)
        return hidden_states, input_tensor

    def adapter_batchsplit(self, adapter_setup: BatchSplit, hidden_states, input_tensor, layer_norm, lvl=0):
        if not sum(adapter_setup.batch_sizes) == hidden_states.shape[0]:
            raise IndexError(
                "The given batch has a size of {} which is not compatible with batch_sizes {}".format(
                    hidden_states.shape[0], adapter_setup.batch_sizes
                )
            )

        first_adapter = self.adapters[adapter_setup.first()]
        hidden_states, _, residual = first_adapter.pre_forward(hidden_states, input_tensor, layer_norm)
        children_hidden = []
        for i, adapter_block in enumerate(adapter_setup):
            # compute ids of sequences thet should be passed to the ith adapter
            batch_idx = (
                sum(adapter_setup.batch_sizes[:i]),
                sum(adapter_setup.batch_sizes[: i + 1]),
            )
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                child, _, _ = self.adapter_stack(
                    adapter_block,
                    hidden_states[batch_idx[0] : batch_idx[1]],
                    input_tensor[batch_idx[0] : batch_idx[1]],
                    layer_norm,
                    lvl=lvl + 1,
                )
                children_hidden.append(child)
            # Case 2: We have a nested split -> recursively call split
            elif isinstance(adapter_block, Split):
                child = self.adapter_split(
                    adapter_block,
                    hidden_states[batch_idx[0] : batch_idx[1]],
                    input_tensor[batch_idx[0] : batch_idx[1]],
                    layer_norm,
                    lvl=lvl + 1,
                )
                children_hidden.append(child)
            # Case 3: We have a nested batch split block -> call batchsplit method
            elif isinstance(adapter_block, BatchSplit):
                child = self.adapter_batchsplit(
                    adapter_block,
                    hidden_states[batch_idx[0] : batch_idx[1]],
                    input_tensor[batch_idx[0] : batch_idx[1]],
                    layer_norm,
                    lvl=lvl + 1,
                )
                children_hidden.append(child)
            # Case 4: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapters:

                adapter_layer = self.adapters[adapter_block]
                context = ForwardContext.get_context()
                layer_output = adapter_layer(
                    hidden_states[batch_idx[0] : batch_idx[1]],
                    residual_input=residual[batch_idx[0] : batch_idx[1]],
                    output_gating=context.output_adapter_gating_scores,
                )
                children_hidden.append(layer_output[0])
                self._store_gating_score(adapter_block, layer_output[-1])
            # Case 5: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore
            else:
                children_hidden.append(hidden_states[batch_idx])

        hidden_states = torch.cat(children_hidden, 0)
        return hidden_states

    def adapter_layer_forward(self, hidden_states, input_tensor, layer_norm, task=None):
        """
        Called for each forward pass through adapters.
        """
        adapter_setup = self.get_active_setup(self.adapters)
        if adapter_setup is not None:
            input_hidden_states = hidden_states

            if isinstance(adapter_setup, Stack):
                hidden_states, _, input_tensor = self.adapter_stack(
                    adapter_setup, hidden_states, input_tensor, layer_norm, task=task
                )
            elif isinstance(adapter_setup, Fuse):
                hidden_states = self.adapter_fusion(adapter_setup, hidden_states, input_tensor, layer_norm)
            elif isinstance(adapter_setup, Split):
                hidden_states = self.adapter_split(adapter_setup, hidden_states, input_tensor, layer_norm)
            elif isinstance(adapter_setup, Parallel):
                # notice that we are overriding input tensor here to keep the same dim as hidden_states for the residual
                # in case we were blowing up the batch for parallel processing of multiple adapters for the same input
                hidden_states, input_tensor = self.adapter_parallel(
                    adapter_setup, hidden_states, input_tensor, layer_norm
                )
            elif isinstance(adapter_setup, BatchSplit):
                hidden_states = self.adapter_batchsplit(adapter_setup, hidden_states, input_tensor, layer_norm)
            else:
                raise ValueError(f"Invalid adapter setup {adapter_setup}")

            last_adapter = self.adapters[adapter_setup.last()]
            hidden_states = last_adapter.post_forward(hidden_states, input_hidden_states, input_tensor, layer_norm)

        elif layer_norm:
            hidden_states = layer_norm(hidden_states + input_tensor)
        else:
            hidden_states = hidden_states + input_tensor

        return hidden_states

    def forward(self, hidden_states, input_tensor, layer_norm):
        return self.adapter_layer_forward(hidden_states, input_tensor, layer_norm)
