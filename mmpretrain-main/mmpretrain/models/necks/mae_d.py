# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule, ModuleList
from mmengine.utils import to_2tuple
from mmpretrain.registry import MODELS
from mmseg.models.utils.embed import PatchEmbed
from mmpretrain.models.backbones.vision_transformer import TransformerEncoderLayer
from einops import rearrange

class PatchExpanding(BaseModule):
    def __init__(self, dim, scale, flag='normal'):
        super(PatchExpanding, self).__init__()
        self.flag = flag
        self.out_channels = dim // scale
        self.out_dims = dim*scale# 维度扩展scale倍，再均分到长宽
        self.linear = nn.Linear(dim, self.out_dims)

    def forward(self, x, input_size):
        H, W = input_size
        x = self.linear(x)
        B, _, _ = x.shape
        x = rearrange(x, 'b (h w) d -> b h w d ', w=W, h=H)
        if self.flag == 'normal':
            x = x.view(B, H * 2, W * 2, -1)
            H = H * 2
            W = W * 2
        elif self.flag == 'special':
            x = x.view(B, H * 4, W * 4, -1)
            H = H * 4
            W = W * 4
        else:
            raise ValueError("There is no such PatchExpanding like this")
        x = rearrange(x, 'b h w d -> b (h w) d')
        output_size = (H, W)
        return x, output_size


class Decoder(BaseModule):
    def __init__(self,
                 embed_dims,
                 depth,
                 num_heads,
                 decoder_embed_dim,
                 mlp_ratio=4,
                 attn_drop_rate=0.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 qkv_bias=True,
                 upsample=None,
                 num_fcs=2):
        super().__init__()
        self.embed_dims = embed_dims
        self.upsample = upsample

        self.blocks = ModuleList()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]


        for i in range(depth):
            block = TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rates[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg)
            self.blocks.append(block)

    def reproduce(self, x, ids_restore):
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        return x_

    def forward(self, t_in, x, hw_shape, ids_restore):
        x_UP, up_hw_shape = self.upsample(x, hw_shape)
        # if ids_restore is not None:
        #     reproduce_out = self.reproduce(t_in, ids_restore)
        # else:
        reproduce_out = t_in
        x_in = x_UP + reproduce_out

        for block in self.blocks:
            x_in = block(x_in)

        return x_in, up_hw_shape, hw_shape

@MODELS.register_module()
class UMAEDecorder(BaseModule):
    def __init__(self,
                 in_channels=3,
                 mlp_ratio=4,
                 predict_feature_dim=48,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 mask_ratio=0.75):
        super().__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.mask_ratio = mask_ratio

        # set stochastic depth decay rule
        total_depth = sum(depths)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.Decoder = ModuleList()
        dpr_reversed = dpr[::-1]

        for i in range(num_layers-1):

            upsample = PatchExpanding(
                dim=in_channels,
                scale=2)
            in_channels = upsample.out_channels
            stage = Decoder(
                embed_dims=in_channels,
                num_heads=num_heads[-i-2],
                decoder_embed_dim=in_channels,
                mlp_ratio=mlp_ratio,
                depth=depths[-i-2],
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr_reversed[sum(depths[::-1][:i+1]):sum(depths[::-1][:i + 2])],
                upsample=upsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg)
            self.Decoder.append(stage)
        self.decoder_pred = nn.Linear(
            in_channels, predict_feature_dim, bias=True)
        self.decoder_norm = nn.LayerNorm(in_channels)

    def forward(self, encoder_outs, x,  hw_shape, idx_sequence):
        for i in range(len(self.Decoder)):
            x, hw_shape, out_hw_shape = self.Decoder[i](encoder_outs[-i - 1], x, hw_shape, idx_sequence[-i - 2])
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x
