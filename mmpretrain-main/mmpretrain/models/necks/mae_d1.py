# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList
from mmengine.utils import to_2tuple
from mmpretrain.registry import MODELS
from mmcv.cnn import ConvModule, build_upsample_layer
from mmpretrain.models.backbones.vision_transformer import TransformerEncoderLayer
from einops import rearrange
import torch.utils.checkpoint as cp
import torch.nn.functional as F


class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.conv_block1 = conv_block(
            in_channels=in_channels,
            out_channels=out_channels,
            num_convs=1,
            stride=stride,
            dilation=dilation,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dcn=None,
            plugins=None)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv_block2 = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dcn=None,
            plugins=None)

    def forward(self, skip, x):
        """Forward function."""

        x = self.upsample(x)
        x = self.conv_block1(x)

        out = torch.cat([skip, x], dim=1)
        out = self.conv_block2(out)

        return out

class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 padding=1,
                 kernel_size=3,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


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
                 num_fcs=2):
        super().__init__()
        self.embed_dims = embed_dims

        self.blocks = ModuleList()
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        self.upsample = UpConvBlock(conv_block=BasicConvBlock,
                   in_channels=embed_dims*2,
                   skip_channels=embed_dims,
                   out_channels=embed_dims,)
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

    def forward(self, t_in, x):
        x_in = self.upsample(t_in, x)
        h = x_in.shape[-2]
        w = x_in.shape[-1]
        # if ids_restore is not None:
        #     reproduce_out = self.reproduce(t_in, ids_restore)
        # else:
        x_in = rearrange(x_in, 'b c h w -> b (h w) c')
        for block in self.blocks:
            x_in = block(x_in)
        x_in = rearrange(x_in, 'b (h w) c -> b c h w', h=h, w=w)

        return x_in

@MODELS.register_module()
class UMAEDecorder1(BaseModule):
    def __init__(self,
                 in_channels,
                 mlp_ratio=4,
                 predict_feature_dim=48,
                 depths=(2, 2, 6, 2),
                 num_heads=(4, 8, 16, 32),
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
        base_channels = 1024
        for i in range(num_layers-1):
            stage = Decoder(
                embed_dims=int(base_channels * 2**(-i - 1)),
                num_heads=num_heads[-i-2],
                decoder_embed_dim=int(base_channels * 2**(-i - 1)),
                mlp_ratio=mlp_ratio,
                depth=depths[-i-2],
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr_reversed[sum(depths[::-1][:i+1]):sum(depths[::-1][:i + 2])],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg)
            self.Decoder.append(stage)
        self.decoder_pred = nn.Linear(
            in_channels, predict_feature_dim, bias=True)
        self.decoder_norm = nn.LayerNorm(in_channels)

    def forward(self, encoder_outs, x):
        for i in range(len(self.Decoder)):
            x = self.Decoder[i](encoder_outs[-i - 1], x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x
