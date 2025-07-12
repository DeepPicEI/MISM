# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead


class EfficientLocalizationAttention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)

        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        # 在两个维度上应用注意力
        return x * x_h * x_w



@MODELS.register_module()
class UPHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.up_ELA = EfficientLocalizationAttention(channel=self.in_channels[-1])
        self.up_bottleneck = ConvModule(
            self.in_channels[-1],
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.up_lateral_convs = nn.ModuleList()
        self.up_fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            up_l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            up_fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.up_lateral_convs.append(up_l_conv)
            self.up_fpn_convs.append(up_fpn_conv)

        self.up_fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
###########下采样支路########################

    def up_ela_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        ela_out = self.up_ELA(x)
        output = self.up_bottleneck(ela_out)

        return output


    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        up_laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.up_lateral_convs)
        ]

        up_laterals.append(self.up_ela_forward(inputs))

        # build top-down path
        up_used_backbone_levels = len(up_laterals)
        for i in range(up_used_backbone_levels - 1, 0, -1):
            prev_shape = up_laterals[i - 1].shape[2:]
            up_laterals[i - 1] = up_laterals[i - 1] + resize(
                up_laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        up_fpn_outs = [
            self.up_fpn_convs[i](up_laterals[i])
            for i in range(up_used_backbone_levels - 1)
        ]
        # append psp feature
        up_fpn_outs.append(up_laterals[-1])

        for i in range(up_used_backbone_levels - 1, 0, -1):
            up_fpn_outs[i] = resize(
                up_fpn_outs[i],
                size=up_fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        up_fpn_outs = torch.cat(up_fpn_outs, dim=1)
        up_feats = self.up_fpn_bottleneck(up_fpn_outs)

        return up_feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output




if __name__ == '__main__':
    x = [torch.randn(2, 96, 56, 56), torch.randn(2, 192, 28, 28), torch.randn(2, 384, 14, 14), torch.randn(2, 768, 7, 7)]
    net = UPHead(in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        )
    out = net(x)
    print(out)