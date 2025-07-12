# dataset settings
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)


# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomResizedCrop', scale=224),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(type='PackInputs'),
# ]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, crop_ratio_range=(0.67, 1.0)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='SimMIMMaskGenerator',
        input_size=224,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='C:/Users/huang/Desktop/data/voc/gzVOCdevkit/VOC2012',
        with_label=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
