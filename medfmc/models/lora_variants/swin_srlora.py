"""
BackBone: Swin-TRM
PET method: LoRA - <<LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS>>
Code: https://github.com/microsoft/LoRA
Paper: https://arxiv.org/abs/2106.09685
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
from mmcv.cnn.bricks.transformer import (FFN, AdaptivePadding, PatchEmbed,
                                         PatchMerging)
from mmcv.runner.base_module import BaseModule, ModuleList
from typing import List, Sequence
import loralib as lora
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
import warnings
from mmcv.cnn import build_norm_layer


class WindowMSALoRA(BaseModule):
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
                 lora_rank,
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
        
        self.input_dims = embed_dims


        # self.qkv = lora.MergedLinear(self.input_dims, embed_dims * 3, r=8, enable_lora=[True, False, True])
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = lora.Linear(embed_dims, embed_dims, bias=True, r=8)
        # self.proj_drop = nn.Dropout(proj_drop)
        
        self.qkv = lora.MergedLinear(self.input_dims, embed_dims * 3, r=lora_rank[0], enable_lora=[True, False, True])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = lora.Linear(embed_dims, embed_dims, bias=qkv_bias, r=lora_rank[1])
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def set_ranks(self, ranks, frozen=False):
        self.qkv.set_rank(ranks[0], frozen=frozen)
        self.proj.set_rank(ranks[1], frozen=frozen)

    def get_dimensions(self):
        dimensions = []
        dimensions.append(self.qkv.get_dimension())
        dimensions.append(self.proj.get_dimension())
        return dimensions


    def get_ranks(self):
        ranks = []   
        ranks.append(self.qkv.get_rank())
        ranks.append(self.proj.get_rank())
        return ranks

    def init_weights(self):
        super(WindowMSALoRA, self).init_weights()

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        



    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor, Optional): mask with shape of (num_windows, Wh*Ww,
                Wh*Ww), value should be between (-inf, 0].
        """
        B_, N, C = x.shape
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
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

class ShiftWindowMSALoRA(BaseModule):
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
                 lora_rank,
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
                 window_msa=WindowMSALoRA,
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
            lora_rank=lora_rank,
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
        
        self.input_dims = embed_dims
        
        # self.qkv = lora.MergedLinear(self.input_dims, embed_dims * 3, r=8, enable_lora=[True, False, True])
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = lora.Linear(embed_dims, embed_dims, bias=True, r=8)
        # self.proj_drop = nn.Dropout(proj_drop)
        
        self.qkv = lora.MergedLinear(self.input_dims, embed_dims * 3, r=lora_rank[0], enable_lora=[True, False, True])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = lora.Linear(embed_dims, embed_dims, bias=True, r=lora_rank[1])
        self.proj_drop = nn.Dropout(proj_drop)
        

    def set_ranks(self, ranks, frozen=False):
        self.qkv.set_rank(ranks[0], frozen=frozen)
        self.proj.set_rank(ranks[1], frozen=frozen)

    def get_dimensions(self):
        dimensions = []
        dimensions.append(self.qkv.get_dimension())
        dimensions.append(self.proj.get_dimension())
        return dimensions


    def get_ranks(self):
        ranks = []   
        ranks.append(self.qkv.get_rank())
        ranks.append(self.proj.get_rank())
        return ranks

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
            mask_windows = ShiftWindowMSALoRA.window_partition(
                img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        return attn_mask

class SwinBlockLoRA(BaseModule):
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
                 lora_rank,
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

        super(SwinBlockLoRA, self).__init__(init_cfg)
        self.with_cp = with_cp

        _attn_cfgs = {
            'embed_dims': embed_dims,
            'lora_rank': lora_rank,
            'num_heads': num_heads,
            'shift_size': window_size // 2 if shift else 0,
            'window_size': window_size,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'pad_small_map': pad_small_map,
            **attn_cfgs
        }
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSALoRA(**_attn_cfgs)

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
        
    def set_ranks(self, ranks, frozen=False):
        return self.attn.set_ranks(ranks, frozen=frozen)
    
    def get_dimensions(self):
        return self.attn.get_dimensions()
    
    def get_ranks(self):
        return self.attn.get_ranks()

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x

class SwinBlockSequenceLoRA(BaseModule):
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
                 lora_ranks,
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
                'lora_rank': lora_ranks[i],
                'num_heads': num_heads,
                'window_size': window_size,
                'shift': False if i % 2 == 0 else True,
                'drop_path': drop_paths[i],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                **block_cfgs[i]
            }
            block = SwinBlockLoRA(**_block_cfg)
            self.blocks.append(block)

        if downsample:
            _downsample_cfg = {
                'in_channels': embed_dims,
                'out_channels': 2 * embed_dims,
                'norm_cfg': dict(type='LN'),
                **downsample_cfg
            }
            self.downsample = PatchMerging(**_downsample_cfg)
        else:
            self.downsample = None

    def set_ranks(self, ranks, frozen=False):
        for i,block in enumerate(self.blocks):
            block.set_ranks(ranks[i], frozen=frozen)
    
    def get_dimensions(self):
        dimensions = []
        for block in self.blocks:   
            dimension = block.get_dimensions()
            dimensions.append(dimension)
        return dimensions
        
    def get_ranks(self):
        ranks = []
        for block in self.blocks:
            rank = block.get_ranks()
            ranks.append(rank)
        return ranks


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





@BACKBONES.register_module()
class SwinTransformerSR_LoRA(SwinTransformer):

    def __init__(
        self,
        lora_ranks = None
        frozen=True,
        arch='base',
        img_size=224,
        patch_size=4,
        in_channels=3,
        window_size=7,
        drop_path_rate=0.1,
        with_cp=False,
        pad_small_map=False,
        stage_cfgs=dict(),
        patch_cfg=dict(),
    ):
        #for swin-b:
        default_lora_ranks = [
            [[3,21],[16,17]],
            [[25,49],[33,43]],
            [[24,46],[37,55],[54,62],[59,63],[69,58],[61,52],[63,55],[50,60],[50,61],[51,55],[47,60],[49,65],[52,64],[54,64],[54,74],[59,61],[64,53],[57,9]],
            [[54,33],[61,13]]
        ]
        #for swin-t:
        # default_lora_ranks = [
        #     [[4,20],[14,18]],
        #     [[25,38],[34,32]],
        #     [[28,63],[38,64],[44,50],[38,51],[38,40],[39,31]],   
        #     [[53,34],[57,8]]
        # ]        
        
        #for swin-s:
        # default_lora_ranks = [
        #     [[3,16],[14,12]],
        #     [[24,37],[26,43]],
        #     [[23,47],[28,54],[41,59],[43,57],[51,47],[47,38],[43,41],[37,42],[35,46],[33,46],[34,52],[34,50],[37,59],[37,65],[39,53],[38,65],[37,65],[36,11]],
        #     [[43,18],[45,9]]
        # ]
        self.lora_ranks = lora_ranks if lora_ranks is not None else default_lora_ranks
        self.frozen = frozen
        
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
                'lora_ranks': self.lora_ranks[i],
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
            self.patch_embed = PatchEmbed(**_patch_cfg)

            stage = SwinBlockSequenceLoRA(**_stage_cfg)
            self.stages.append(stage)
            dpr = dpr[depth:]
            embed_dims.append(stage.out_channels)
            
        for name,param in self.named_parameters():
            if 'lora' in name:
                # import pdb; pdb.set_trace()
                continue
            else:
                param.requires_grad = False


    def set_ranks(self, all_ranks, frozen):
        for i, stage in enumerate(self.stages):
            stage.set_ranks(all_ranks[i], frozen=frozen)


    def get_dimensions(self):
        all_dimensions = []
        for stage in self.stages:
            all_dimensions.append(stage.get_dimensions())
        return all_dimensions

    def get_ranks(self):
        all_ranks = []
        for stage in self.stages:
            all_ranks.append(stage.get_ranks())
        return all_ranks
    

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.absolute_pos_embed, self.patch_resolution, hw_shape,
                self.interpolate_mode, self.num_extra_tokens)
        x = self.drop_after_pos(x)

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
                self.__class__ is SwinTransformerSR_LoRA:
            final_stage_num = len(self.stages) - 1
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if k.startswith('norm.') or k.startswith('backbone.norm.'):
                    convert_key = k.replace('norm.', f'norm{final_stage_num}.')
                    state_dict[convert_key] = state_dict[k]
                    del state_dict[k]
        if (version is None or version < 3) and \
                self.__class__ is SwinTransformerSR_LoRA:
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if 'attn_mask' in k:
                    del state_dict[k]

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      *args, **kwargs)