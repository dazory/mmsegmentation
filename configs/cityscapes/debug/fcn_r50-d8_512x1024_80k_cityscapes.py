root = '/ws/external/configs'
_base_ = [
    f'{root}/_base_/models/fcn_r50-d8.py', f'{root}/_base_/datasets/cityscapes.py',
    f'{root}/_base_/default_runtime.py', f'{root}/_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained=None)


###############
### DATASET ###
###############
custom_imports = dict(imports=['mmseg.datasets.pipelines.augmix'], allow_failed_imports=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    ### AugMix ###
    dict(type='AugMix', no_jsd=False, aug_list='wotrans', **img_norm_cfg),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline),
    test=dict(
        data_root='/ws/data/cityscapes-c/'))
