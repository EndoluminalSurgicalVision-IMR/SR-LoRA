_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/datasets/endoscopy.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py',
]

lr = 1e-5
n = 1
# vpl = 5
dataset = 'endo'
nshot = 1
run_name = f'in21k-vitadapter_bs4_lr{lr}_{nshot}-shot_{dataset}'

model = dict(
    type='ImageClassifier',
    backbone=dict(type='VitAdapter'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=4,
        in_channels=768,
    ))
# data = dict(
#     samples_per_gpu=4,  # use 2 gpus, total 128
#     train=dict(
#         ann_file=f'data/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp5.txt'),
#     val=dict(        
#         data_prefix='/lustre/home/acct-eeyj/eeyj-tuen/yyc_workspace/MedFMC_val/chest/images',
#         ann_file='/lustre/home/acct-eeyj/eeyj-tuen/yyc_workspace/MedFMC_val/chest/validation.txt',),
#     test=dict(
#         data_prefix='/lustre/home/acct-eeyj/eeyj-tuen/yyc_workspace/MedFMC_val/chest/images',
#         ann_file='/lustre/home/acct-eeyj/eeyj-tuen/yyc_workspace/MedFMC_val/chest/validation.txt',
#     ))

data = dict(
    samples_per_gpu=4,  # use 2 gpus, total 128
    train=dict(
        ann_file=f'data_backup/MedFMC/{dataset}/{dataset}_{nshot}-shot_train_exp5.txt'),
    val=dict(ann_file=f'data_backup/MedFMC/{dataset}/{dataset}_{nshot}-shot_val_exp5.txt'),
    test=dict(ann_file=f'data/MedFMC/{dataset}/val_WithLabel.txt'))


optimizer = dict(lr=lr)

# learning policy
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

# load_from = 'pretrain/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
# work_dir = f'work_dirs/vit_adapter/{run_name}'

load_from = 'work_dirs/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
work_dir = f'work_dirs/vit_adapter/{run_name}'

evaluation = dict(interval=5, metric='mAP', save_best='auto')
runner = dict(type='EpochBasedRunner', max_epochs=20)
