# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_norm_layer
from mmcv.cnn import ConvModule

from mmengine.model import BaseModule, ModuleList
from mmengine.utils import to_2tuple
from mmpretrain.registry import MODELS
from mmpretrain.models.backbones.vision_transformer import TransformerEncoderLayer
from einops import rearrange
from .base import BaseSelfSupervisor
from typing import Dict, List, Tuple
from mmpretrain.structures import DataSample
from mmpretrain.models.backbones.resnet import ResNet
class Bottle(BaseModule):
    def __init__(self, in_channels,
                 num_heads,
                 mlp_ratio,
                 attn_drop_rate,
                 drop_rate,
                 dpr,
                 qkv_bias,
                 norm_cfg,
                 bottle_layer=2,
                 ):
        super(Bottle, self).__init__()
        self.Bottle = ModuleList()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(in_channels/2)))
        for i in range(bottle_layer):
            block = TransformerEncoderLayer(
                embed_dims=int(in_channels/2),
                num_heads=num_heads,
                feedforward_channels=int(mlp_ratio * in_channels/2),
                attn_drop_rate=attn_drop_rate,
                drop_rate=drop_rate,
                drop_path_rate=dpr,
                num_fcs=2,
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg,)
            self.Bottle.append(block)
    def reproduce(self, x, ids_restore):
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        return x_

    def select(self, x, idx):
        b, l, c = x.size()
        x_masked = torch.gather(
            x, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, c))
        return x_masked

    def forward(self, x, idx, ids_restore):
        if ids_restore is not None:
            t_out = self.select(x, idx)
            for i in range(len(self.Bottle)):
                t_out = self.Bottle[i](t_out)
            t_out = self.reproduce(t_out, ids_restore)
            return t_out
        else:
            for i in range(len(self.Bottle)):
                x = self.Bottle[i](x)
            return x

class Encoder(BaseModule):
    def __init__(self, embed_dims,
                 depth,
                 num_heads,
                 mlp_ratio=4,
                 attn_drop_rate=0.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 qkv_bias=True,
                 downsample=None,
                 with_cp=False,
                 num_fcs=2):
        super().__init__()
        self.downsample = downsample
        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(embed_dims/2)))
        self.blocks = ModuleList()
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]


        for i in range(depth):
            block = TransformerEncoderLayer(
                    embed_dims=int(embed_dims/2),
                    num_heads=num_heads,
                    feedforward_channels=int(mlp_ratio * (embed_dims/2)),
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rates[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,)
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

    def select(self, x, idx):
        b, l, c = x.size()
        x_masked = torch.gather(
            x, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, c))
        return x_masked

    def forward(self, x, idx, ids_restore):
        if idx is not None:
            t_out = self.select(x, idx)
            for block in self.blocks:
                t_out = block(t_out)
            t_out = self.reproduce(t_out, ids_restore)
        else:
            t_out = x
            for block in self.blocks:
                t_out = block(t_out)
        return t_out


@MODELS.register_module()
class UMAEencorder1(BaseModule):
    def __init__(self,
                 pretrain_img_size=224,
                 embed_dims=256,
                 patch_size=4,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(4, 8, 16, 32),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None,
                 mask_ratio=0.75):
        self.frozen_stages = frozen_stages

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super().__init__(init_cfg=init_cfg)
        self.resnet = ResNet(depth=50, num_stages=4, out_indices=(0, 1, 2, 3))
        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.mask_ratio = mask_ratio

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'


        # set stochastic depth decay rule
        total_depth = sum(depths)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]
        in_channels = [256, 512, 1024, 2048]
        self.downdim = ModuleList()
        for i, channels in enumerate(in_channels):
            self.downdim.append(ConvModule(
                    in_channels=channels,
                    out_channels=int(channels/2),
                    kernel_size=1,
                    stride= 1,
                    dilation=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')))

        self.Encoder = ModuleList()

        for i in range(num_layers-1):
            stage = Encoder(
                embed_dims=in_channels[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                depth=depths[i],
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp)
            self.Encoder.append(stage)


        self.Bottle = Bottle(
            in_channels=in_channels[-1],
            num_heads=num_heads[-1],
            mlp_ratio=mlp_ratio,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
            dpr=dpr[-1],
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            bottle_layer=2)


        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        # for i in out_indices:
        #     layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
        #     layer_name = f'norm{i}'
        #     self.add_module(layer_name, layer)

    def random_masking(self,
            x: torch.Tensor,
            mask_ratio: float = 0.75
    ):
        """Generate the mask for MAE Pre-training.

        Args:
            x (torch.Tensor): Image with data augmentation applied, which is
                of shape B x L x C.
            mask_ratio (float): The mask ratio of total patches.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked image, mask
            and the ids to restore original image.

            - ``x_masked`` (torch.Tensor): masked image.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        B, C, H, W = x.shape  # batch, length, dim
        noise = torch.rand((B, H, W), device=x.device)
        idx_sequence = []
        mask_out = []
        idx = []
        for i in range(4):
            new_height = int(H / (2 ** i))
            new_width = int(W / (2 ** i))
            noise = F.interpolate(noise.unsqueeze(1), size=(new_height, new_width),
                                  mode='bilinear',
                                  align_corners=False)
            noise = rearrange(noise, 'b c h w -> (b c) (h w)')
            len_keep = int(new_height * new_width * (1 - mask_ratio))
            ids_shuffle = torch.argsort(
                noise, dim=1)  # ascend: small is keep, large is remove
            ids_keep = ids_shuffle[:, :len_keep]
            idx.append(ids_keep)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            idx_sequence.append(ids_restore)
            noise = rearrange(noise, 'b (h w) -> b h w', h=new_height, w=new_width)
            # keep the first subset

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, new_height * new_width], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
            mask_out.append(mask)
        return mask_out, idx, idx_sequence

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.resnet.eval()
            for param in self.resnet.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if i < 4:
                m = self.Encoder[i - 1]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
            else:
                m = self.Bottle
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward_mask(self, x_in):
        x_in = self.resnet(x_in)
        shape = [(i.shape[-1], i.shape[-2]) for i in x_in]
        # hw_shape = (x[0].shape[-2], x[0].shape[-2])
        x=[]
        for i in range(len(self.downdim)):
            x_out = self.downdim[i](x_in[i])
            x.append(x_out)

        mask, idx, idx_sequence = self.random_masking(x[0], mask_ratio=self.mask_ratio)
        x = [rearrange(i, 'b c h w -> b (h w) c') for i in x]
        encoder_outs = []
        for i in range(len(self.Encoder)):
            out = self.Encoder[i](x[i], idx[i], ids_restore=idx_sequence[i])
            encoder_outs.append(out)

        x = self.Bottle(x[-1], idx=idx[-1], ids_restore=idx_sequence[-1])

        norm_outs = []
        for i in self.out_indices:

            if i == self.out_indices[-1]:
                x = rearrange(x, 'b (h w) c -> b c h w', h=shape[i][0], w=shape[i][0])
            else:
                norm_out = rearrange(encoder_outs[i], 'b (h w) c -> b c h w', h=shape[i][0], w=shape[i][0])
                norm_outs.append(norm_out)

        return norm_outs, x, mask[0]

    def forward_infer(self, x_in):
        x_in = self.resnet(x_in)
        shape = [(i.shape[-1], i.shape[-2]) for i in x_in]
        x = []
        for i in range(len(self.downdim)):
            x_out = self.downdim[i](x_in[i])
            x.append(x_out)

        encoder_outs = []
        for i in range(len(self.Encoder)):
            out, = self.Encoder[i](x, idx=None, ids_restore=None)
            encoder_outs.append(out)

        x = self.Bottle(x, idx=None, ids_restore=None)
        encoder_outs.append(x)
        norm_outs = []
        for i in self.out_indices:

            if i == self.out_indices[-1]:
                x = rearrange(x, 'b (h w) c -> b c h w', h=shape[i][0], w=shape[i][1])
            else:
                norm_out = rearrange(encoder_outs[i], 'b (h w) c -> b c h w', h=shape[i][0], w=shape[i][1])
                norm_outs.append(norm_out)

        return norm_outs

    def forward(self, x):
        if self.mask_ratio is not None:
            return self.forward_mask(x)
        else:
            return self.forward_infer(x)


@MODELS.register_module()
class UMAEBN1(BaseSelfSupervisor):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
    <https://arxiv.org/abs/2111.06377>`_.
    """

    def extract_feat(self, inputs: torch.Tensor):
        return self.backbone(inputs, mask=None)

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (torch.Tensor): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.
        encoder_outs, x, mask = self.backbone(inputs)
        pred = self.neck(encoder_outs, x)
        loss = self.head.loss(pred, inputs, mask)
        losses = dict(loss=loss)
        return losses

# if __name__ == '__main__':
#     x = torch.randn(4, 3, 224, 224)
#     model = SwinTransformer()
#     out = model(x)
#     print(out)