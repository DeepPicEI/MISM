_base_ = [
    '../_base_/datasets/clip.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='ClipSam',
    backbone=dict(
        type='AgentSwinTransformer',
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=80,
        embed_dim=128,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        window_size=56,
        mlp_ratio=4,
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.5,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        agent_num=[9, 16, 49, 49],
        downstream_agent_shapes=[(9, 9), (12, 12), (14, 14), (7, 7)],
        kernel_size=3,
        attn_type='AABB',
        scale=-0.5),
    neck=dict(
        type='CLIPSamNeck', in_channels=1024, out_channels=256),
    head=dict(
        type='CLIPSamHead',
        loss=dict(type='PixelReconstructionLoss', criterion='L2', channel=3)))

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4 * 2048 / 512,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=5.0),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=5e-7 / 1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='MultiStepLR',
        milestones=[700],
        by_epoch=True,
        begin=10,
        end=800,
        convert_to_iter_based=True)
]

# runtime
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=2048)