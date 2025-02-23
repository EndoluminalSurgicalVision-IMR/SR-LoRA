_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/datasets/vtab.py',  
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py',
]

lr = 1e-3
dataset = 'VTAB1k'  
sub_dataset = 'sun397'
run_name = f'in21k-vitmelora_bs4_lr{lr}_{dataset}_{sub_dataset}'

model = dict(
    type='ImageClassifier',
   backbone=dict(type='VitMELoRA', lora_rank=8, num_lora_matrices=4),
    head=dict(
        type='LinearClsHead',
        num_classes=397, 
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset,
        data_prefix=f'data/vtab-1k/{sub_dataset}',
        ann_file=f'data/vtab-1k/{sub_dataset}/train800val200.txt'),
    val=dict(
        type=dataset,
         data_prefix=f'data/vtab-1k/{sub_dataset}',
        ann_file=f'data/vtab-1k/{sub_dataset}/test.txt',
        test_mode=True),
    test=dict(
        type=dataset,  
        data_prefix=f'data/vtab-1k/{sub_dataset}',
        ann_file=f'data/vtab-1k/{sub_dataset}/test.txt',
        test_mode=True))


test_evaluator = dict(type='Accuracy', topk=(1, 5, ))
val_evaluator = dict(type='Accuracy', topk=(1, 5, ))

optimizer = dict(lr=lr, weight_decay=0.05)

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20,
    warmup_by_epoch=False)

log_config = dict(
    interval=5, hooks=[
        dict(type='TextLoggerHook'),
    ])

load_from = 'work_dirs/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
work_dir = f'work_dirs/vit_lora-mevtab-few-shot/{run_name}'

runner = dict(type='EpochBasedRunner', max_epochs=20)