# model settings
model = dict(
    type='UMAE',
    backbone=dict(type='UMAEVIT', patch_size=4, mask_ratio=0.75),
    neck=None,
    head=dict(
        type='MAEPretrainHead',
        norm_pix=True,
        patch_size=4,
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    init_cfg=[
        dict(type='Xavier', layer='Linear', distribution='uniform'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])
