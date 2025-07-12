_base_ = [
    '../_base_/datasets/clip.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='ClipSam',
    backbone=dict(
        type='RMT_Swin',
        embed_dims=[128, 256, 512, 1024],
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        init_values=[2, 2, 2, 2],
        heads_ranges=[5, 5, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.4,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        out_indices = (0, 1, 2, 3)),
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