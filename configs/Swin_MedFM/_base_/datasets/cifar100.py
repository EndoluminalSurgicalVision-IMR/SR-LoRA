#dataset settings
dataset_type = 'CIFAR100Fewshot'
img_norm_cfg = dict(
    mean=[125.307, 122.95, 113.865],  # CIFAR100 dataset mean
    std=[62.9932, 62.0887, 66.7048],  # CIFAR100 dataset std
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),  # Assumes images are pre-loaded as numpy arrays
    dict(
        type='RandomResizedCrop',
        size=32,  # CIFAR100 images are 32x32
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # Assumes images are pre-loaded as numpy arrays
    dict(type='Resize', size=32, backend='pillow', interpolation='bicubic'),  # CIFAR100 images are 32x32
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/VTAB-1k/CIFAR100/images', 
        ann_file='data/VTAB-1k/CIFAR100/cifar100_1-shot_train_exp1.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/VTAB-1k/CIFAR100/images',  
        ann_file='data/VTAB-1k/CIFAR100/cifar100_1-shot_val_exp1.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='data/VTAB-1k/CIFAR100/images',  
        ann_file='data/VTAB-1k/CIFAR100/cifar100_testWithLabel.txt',
        pipeline=test_pipeline))
evaluation = dict(
    interval=1,
    metric='accuracy',
    metric_options={'topk': (1, 5)},  
    save_best='auto')

# #dataset settings
# dataset_type = 'CIFAR100'
# img_norm_cfg = dict(
#     mean=[129.304, 124.070, 112.434],
#     std=[68.170, 65.392, 70.418],
#     to_rgb=False)
# train_pipeline = [
#     dict(type='RandomCrop', size=32, padding=4),
#     dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'])
# ]
# test_pipeline = [
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'])
# ]
# data = dict(
#     samples_per_gpu=16,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         data_prefix='data/cifar100',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         data_prefix='data/cifar100',
#         pipeline=test_pipeline,
#         test_mode=True),
#     test=dict(
#         type=dataset_type,
#         data_prefix='data/cifar100',
#         pipeline=test_pipeline,
#         test_mode=True))