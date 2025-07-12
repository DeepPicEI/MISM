# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList

from mmpretrain.registry import MODELS
from mmcv.cnn import ConvModule, build_upsample_layer

import torch.utils.checkpoint as cp



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
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None,
                 upsample_cfg=dict(type='InterpConv')):
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
            in_channels=in_channels,
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


@MODELS.register_module()
class UMAESwinNeck(BaseModule):
    def __init__(self,
                 in_channels,
                 encoder_stride=4,
                 depths=(2, 2, 2, 2),
                 init_cfg=None,):
        super().__init__(init_cfg=init_cfg)

        num_layers = len(depths)

        # set stochastic depth decay rule
        self.Decoder = ModuleList()
        for i in range(num_layers-1):

            stage = UpConvBlock(conv_block=BasicConvBlock,
                                in_channels=in_channels[i],
                                out_channels=in_channels[i+1],)
            self.Decoder.append(stage)
        self.Pixel_head = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[-1],
                out_channels=encoder_stride**2 * 3,
                kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )

    def forward(self, encoder_outs):
        x = encoder_outs[-1]
        for i in range(len(self.Decoder)):
            x = self.Decoder[i](encoder_outs[-i - 2], x)
        x = self.Pixel_head(x)
        return x

# if __name__ == '__main__':
#     encoder = [ torch.randn(2, 96, 56, 56), torch.randn(2, 192, 28, 28), torch.randn(2, 384, 14, 14),
#                 torch.randn(2, 768, 7, 7)]
#     net = UMAEDecorder11(in_channels=[768, 384, 192, 96])
#     out = net(encoder, encoder[-1])