# model settings
model = dict(
    type='UMAEFPN',
    backbone=dict(type='UMAVITFPN', patch_size=4, mask_ratio=0.75),
    neck=dict(
        type='UMAEFPNNeck',
        in_channels=768,
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
