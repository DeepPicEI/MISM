# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from mmcv.cnn import ContextBlock


import torch
import torch.nn.functional as F
import torch.nn as nn


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


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
class UPDownHead(BaseDecodeHead):
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
        self.up_ss = ScConv(op_channel=4*self.channels)
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

        self.down_ELA = EfficientLocalizationAttention(channel=self.in_channels[-1])
        self.down_ss = ScConv(op_channel=4 * self.channels)
        self.down_bottleneck = ConvModule(
            self.in_channels[-1],
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # FPN Module

        self.down_lateral_convs = nn.ModuleList()
        self.down_fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:
            down_l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            down_fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.down_lateral_convs.append(down_l_conv)
            self.down_fpn_convs.append(down_fpn_conv)

        self.down_fpn_bottleneck = ConvModule(
            int(self.channels/16),
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.PixelShuffle = nn.PixelShuffle(8)

    def up_ela_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        ela_out = self.up_ELA(x)
        output = self.up_bottleneck(ela_out)

        return output

    def down_ela_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        ela_out = self.up_ELA(x)
        output = self.down_bottleneck(ela_out)

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

#######################################下采样#################################################
        down_laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.down_lateral_convs)
        ]

        down_laterals.append(self.down_ela_forward(inputs))

        # build top-down path
        down_used_backbone_levels = len(down_laterals)
        for i in range(0, down_used_backbone_levels - 1 ):
            prev_shape = up_laterals[i - 1].shape[2:]
            down_laterals[i - 1] = down_laterals[i - 1] + resize(
                down_laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        down_fpn_outs = [
            self.down_fpn_convs[i](down_laterals[i])
            for i in range(down_used_backbone_levels - 1)
        ]
        # append psp feature
        down_fpn_outs.append(down_laterals[-1])

        for i in range(0, down_used_backbone_levels - 1):
            down_fpn_outs[i] = resize(
                down_fpn_outs[i],
                size=down_fpn_outs[-1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        down_fpn_outs = torch.cat(down_fpn_outs, dim=1)
        down_fpn_outs = self.PixelShuffle(down_fpn_outs)
        down_feats = self.down_fpn_bottleneck(down_fpn_outs)
        down_feats = resize(
                down_feats,
                size=up_feats.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        feats = down_feats + up_feats
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output




if __name__ == '__main__':
    x = [torch.randn(2, 96, 56, 56), torch.randn(2, 192, 28, 28), torch.randn(2, 384, 14, 14), torch.randn(2, 768, 7, 7)]
    net = UPDownHead(in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        )
    out = net(x)
    print(out)