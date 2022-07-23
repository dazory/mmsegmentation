root = '/ws/external/configs'
_base_ = [
    f'{root}/_base_/models/fcn_r50-d8.py', f'{root}/_base_/datasets/cityscapes.py',
    f'{root}/_base_/default_runtime.py', f'{root}/_base_/schedules/schedule_80k.py'
]
'''
[NAMING] (updated 22.06.09)
  data pipeline:    [original, augmix(copy, wotrans)]
  loss:             [none, plus]
  decode_head:      [decode_head(none, jsd, ...)]
  auxiliary_head:   [auxiliary_head(none, jsd, ...)]
  parameters:       [e1, lw(e.g.,1e-4), wr(true, false)]
                     > e1: 1 epoch
                     > lw: lambda weight
                     > wr: weight reduce  
[OPTIONS] (updated 22.06.09)
  model
  * decode_head/loss_decode.additional_loss,
    auxiliary_head/loss_decode.additional_loss
    : [None, 'jsd', 'jsdy', 'jsdv1_1', 'jsdv2', ...]
  * train_cfg.wandb.log.features_list 
    : [None, "rpn_head.rpn_cls", "neck.fpn_convs.0.conv", "neck.fpn_convs.1.conv", "neck.fpn_convs.2.conv", "neck.fpn_convs.3.conv"]
  * train_cfg.wandb.log.vars
    : ['log_vars'] 
'''

#############
### MODEL ###
#############
model = dict(
    ### NOTICE: pretrained = None IF DEBUG, ELSE 'open-mmlab://resnet50_v1c'
    pretrained=None, # 'open-mmlab://resnet50_v1c'
    decode_head=dict(
        loss_decode=dict(
            type='CrossEntropyLossPlus', use_sigmoid=False, loss_weight=1.0
            , additional_loss='jsdv1_3_1', lambda_weight=0.0001, wandb_name='dechead'
            )),
    auxiliary_head=dict(
        loss_decode=dict(
            type='CrossEntropyLossPlus', use_sigmoid=False, loss_weight=0.4
            , additional_loss='None', lambda_weight=0.0001, wandb_name='auxhead'
            )),
)



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
    dict(type='Collect', keys=['img', 'img2', 'img3', 'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline),
    test=dict(
        data_root='/ws/data/cityscapes-c/'))
