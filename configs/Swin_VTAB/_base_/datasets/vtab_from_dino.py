# dataset settings
import mmcls.datasets.pipelines
dataset_type = 'VTAB1k'
img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], 
    mean=[125.3, 123.0, 113.9], std=[63.0, 62.1, 66.7],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=128,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Cutout',shape=128*0.2),
    # dict(type='Brightness', magnitude=0.2),
    # dict(type='Contrast', magnitude=0.2),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=128, backend='pillow', interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/vtab-1k_cifar',
        ann_file='data/vtab-1k_cifar/train800val200.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
         data_prefix='data/vtab-1k_cifar',
        ann_file='data/vtab-1k_cifar/test.txt',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,  
        data_prefix='data/vtab-1k_cifar',
        ann_file='data/vtab-1k_cifar/test.txt',
        pipeline=test_pipeline,
        test_mode=True))


evaluation = dict(
    interval=1,
    metric='accuracy',
    metric_options={'topk': (1, 5)},  
    save_best='auto')