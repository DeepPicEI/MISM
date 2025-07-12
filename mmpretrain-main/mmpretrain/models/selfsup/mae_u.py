# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_norm_layer

from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner import CheckpointLoader
from mmengine.utils import to_2tuple

from mmpretrain.registry import MODELS
from mmseg.models.utils.embed import PatchEmbed, PatchMerging
from mmpretrain.models.backbones.vision_transformer import TransformerEncoderLayer
from einops import rearrange
from .base import BaseSelfSupervisor
from typing import Dict, List, Tuple
from mmpretrain.structures import DataSample

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
        # self.partition = Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_height, p2=patch_width)
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


class Bottle(BaseModule):
    def __init__(self, in_channels,
                 num_heads,
                 mlp_ratio,
                 attn_drop_rate,
                 drop_rate,
                 dpr,
                 qkv_bias,
                 norm_cfg,
                 decoder_embed_dim,
                 bottle_layer=2,
                 ):
        super(Bottle, self).__init__()
        self.Bottle = ModuleList()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        for i in range(bottle_layer):
            block = TransformerEncoderLayer(
                embed_dims=in_channels,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * in_channels,
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
            t_in = self.select(x, idx)
            for i in range(len(self.Bottle)):
                t_in = self.Bottle[i](t_in)
            t_out = self.reproduce(t_in, ids_restore)
            return t_out
        else:
            t_out = self.Bottle(x)
        return t_out

class Decoder(BaseModule):

    def __init__(self, embed_dims,
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
        if ids_restore is not None:
            reproduce_out = self.reproduce(t_in, ids_restore)
        else:
            reproduce_out = t_in
        x_in = x_UP + reproduce_out

        for block in self.blocks:
            t_out = block(x_in)

        return t_out, up_hw_shape, hw_shape
        # else:
        #     return t_out, hw_shape, t_out, hw_shape


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
        self.embed_dims = embed_dims
        self.downsample = downsample
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

        self.blocks = ModuleList()
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
                    norm_cfg=norm_cfg,)
            self.blocks.append(block)

    def select(self, x, idx):
        b, l, c = x.size()
        x_masked = torch.gather(
            x, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, c))
        return x_masked

    def forward(self, x, idx, hw_shape):
        if idx is not None:
            t_in = self.select(x, idx)
        else:
            t_in = x
        for block in self.blocks:
            t_out = block(t_in)
        x_down, down_hw_shape = self.downsample(x, hw_shape)
        return x_down, down_hw_shape, t_out, hw_shape


@MODELS.register_module()
class UMAEVIT(BaseModule):
    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 mlp_ratio=4,
                 predict_feature_dim=48,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 patch_norm=True,
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

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.mask_ratio = mask_ratio

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.Encoder = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers-1):
            downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            stage = Encoder(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                depth=depths[i],
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp)
            self.Encoder.append(stage)

            in_channels = downsample.out_channels

        self.Bottle = Bottle(
            in_channels=in_channels,
            num_heads=num_heads[-1],
            mlp_ratio=mlp_ratio,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
            dpr=dpr[-1],
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            decoder_embed_dim=in_channels,
            bottle_layer=2)

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
            # else:
            #     stage = PatchExpanding(
            #         flag='special',
            #         dim=in_channels,
            #         scale=4)
            self.Decoder.append(stage)
        self.decoder_pred = nn.Linear(
            in_channels, predict_feature_dim, bias=True)
        self.decoder_norm = nn.LayerNorm(in_channels)


        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def random_masking(self,
            x: torch.Tensor,
            shape,
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
        B, _, _ = x.shape  # batch, length, dim
        H, W = shape
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
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    print_log('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    print_log(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, strict=False)

    def forward_mask(self, x):
        x, hw_shape = self.patch_embed(x)
        mask, idx, idx_sequence = self.random_masking(x, hw_shape, mask_ratio=self.mask_ratio)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        encoder_outs = []
        for i in range(len(self.Encoder)):
            x, hw_shape, out, out_hw_shape = self.Encoder[i](x, idx[i], hw_shape)
            encoder_outs.append(out)

        x = self.Bottle(x, idx=idx[-1], ids_restore=idx_sequence[-1])
        decoder_out = []
        for i in range(len(self.Decoder)):
            x, hw_shape, out_hw_shape = self.Decoder[i](encoder_outs[-i - 1], x, hw_shape, idx_sequence[-i - 2])
            decoder_out.append(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return (x, mask[0], idx_sequence)

    def forward_infer(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        encoder_outs = []
        for i in range(len(self.Encoder)):
            x, hw_shape, out, out_hw_shape = self.Encoder[i](x, hw_shape, idx=None)
            encoder_outs.append(out)

        x = self.Bottle(x, idx=None, ids_restore=None)

        decoder_out = []
        for i in range(len(self.Decoder)):
            x, hw_shape, out_hw_shape = self.Decoder[i](encoder_outs[-i - 1], x, hw_shape, idx_sequence=None)
            decoder_out.append(x)
        return decoder_out

    def forward(self, x):
        if self.mask_ratio is not None:
            return self.forward_mask(x)
        else:
            return self.forward_infer(x)


@MODELS.register_module()
class UMAE(BaseSelfSupervisor):
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
        pred, mask, ids_restore = self.backbone(inputs)
        # pred = self.neck(latent, ids_restore)
        loss = self.head.loss(pred, inputs, mask)
        losses = dict(loss=loss)
        return losses

# if __name__ == '__main__':
#     x = torch.randn(4, 3, 224, 224)
#     model = SwinTransformer()
#     out = model(x)
#     print(out)