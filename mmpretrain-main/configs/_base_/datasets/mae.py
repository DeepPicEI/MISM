# dataset settings
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[39.91576, 39.91576, 39.91576],
    std=[69.24041, 69.24041, 69.24041],
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

# train_dataloader = dict(
#     batch_size=4,
#     num_workers=5,
#     dataset=dict(
#         type=dataset_type,
#         data_root='C:/Users/huang/Desktop/data/voc/jzVOCdevkit/VOC2012',
#         split='train',
#         pipeline=train_pipeline),
#     sampler=dict(type='DefaultSampler', shuffle=True),
# )
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    # 训练数据集配置
    dataset=dict(
        type='CustomDataset',
        data_prefix='C:/Users/huang/Desktop/data/jizhu/images/train',
        with_label=False,  # 对于无监督任务，使用 False
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),)