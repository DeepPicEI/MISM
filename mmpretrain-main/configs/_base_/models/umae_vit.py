# model settings
model = dict(
    type='UMAEBN1',
    backbone=dict(type='UMAEencorder1', patch_size=4, mask_ratio=0.75),
    neck=dict(
        type='UMAEDecorder1',
        in_channels=128,
        mlp_ratio=4,
    ),
    head=dict(
        type='MAEPretrainHead',
        norm_pix=True,
        patch_size=4,
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    init_cfg=[
        dict(type='Xavier', layer='Linear', distribution='uniform'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])
