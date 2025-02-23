_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/datasets/colon.py',
    '../_base_/schedules/imagenet_dense.py',
    '../_base_/default_runtime.py',
]

lr = 1e-3
n = 1
vpl = 1
dataset = 'colon'
nshot = 1
run_name = f'in21k-vitvpt_bs4_lr{lr}_{nshot}-shot_{dataset}'

model = dict(
    type='ImageClassifier',
    backbone=dict(type='PromptedVisionTransformer', prompt_length=1),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
data = dict(
    samples_per_gpu=4,  # use 2 gpus, total 128
    train=dict(
        ann_file=f'data_backup/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp5.txt'),
    val=dict(ann_file=f'data_backup/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp5.txt'),
    test=dict(ann_file=f'data/MedFMC/{dataset}/val_WithLabel.txt'))

optimizer = dict(lr=lr)

log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

# load_from = 'work_dirs/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth'

load_from = 'work_dirs/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'

work_dir = f'work_dirs/vit_vpt/{run_name}'

runner = dict(type='EpochBasedRunner', max_epochs=20)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
