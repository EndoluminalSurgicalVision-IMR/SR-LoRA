_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/datasets/vtab_no_aug.py',  
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

lr = 1e-3
dataset = 'VTAB1k'  
sub_dataset = 'clevr_count'
run_name = f'in21k-swinsrlora_bs4_lr{lr}_{dataset}_{sub_dataset}'

model = dict(
    type='ImageClassifier',
     backbone=dict(type='SwinTransformerSR_LoRA',arch='base',img_size=384,stage_cfgs=dict(block_cfgs=dict(window_size=12))),
    head=dict(
        type='LinearClsHead',
        num_classes=8, 
        in_channels=1024,
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

custom_hooks = [
    dict(type='swin_srloraHook')
]

load_from = 'work_dirs/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth'
work_dir = f'work_dirs/swin_srlora-vtab-few-shot/{run_name}'

runner = dict(type='EpochBasedRunner', max_epochs=20)