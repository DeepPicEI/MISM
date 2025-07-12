# dataset settings
dataset_type = 'VOC'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='C:/student_learn_data/python/deep-learning-for-image-processing-master/pytorch_object_detection/faster_rcnn/VOCdevkit/VOC2012',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
