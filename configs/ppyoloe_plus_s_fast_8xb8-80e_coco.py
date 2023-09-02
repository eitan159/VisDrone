data_root = '/cortex/data/images/VisDrone/'
dataset_type = 'VisDrone'
num_classes = 8
deepen_factor = 0.33
widen_factor = 0.5
base_lr = 0.001


train_batch_size_per_gpu = 8
train_num_workers = 8
val_batch_size_per_gpu = 1
val_num_workers = 2

load_from = './pretrained_models/ppyoloe_plus_s_fast_8xb8-80e_coco_20230101_154052-9fee7619.pth'
save_epoch_intervals = 5
max_epochs = 30
work_dir = './exp/'

_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
backend_args = None
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
]
default_hooks = dict(
    checkpoint=dict(
        interval=save_epoch_intervals, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(
        min_lr_ratio=0.0,
        start_factor=0.0,
        total_epochs=96,
        type='PPYOLOEParamSchedulerHook',
        warmup_epochs=5,
        warmup_min_iter=1000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        attention_cfg=dict(
            act_cfg=dict(type='HSigmoid'), type='EffectiveSELayer'),
        block_cfg=dict(
            shortcut=True, type='PPYOLOEBasicBlock', use_alpha=True),
        deepen_factor=0.33,
        norm_cfg=dict(eps=1e-05, momentum=0.1, type='BN'),
        type='PPYOLOECSPResNet',
        use_large_stem=True,
        widen_factor=0.5),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                192,
                384,
                768,
            ],
            norm_cfg=dict(eps=1e-05, momentum=0.1, type='BN'),
            num_base_priors=1,
            num_classes=num_classes,
            reg_max=16,
            type='PPYOLOEHeadModule',
            widen_factor=0.5),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='giou',
            loss_weight=2.5,
            reduction='mean',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0,
            reduction='sum',
            type='mmdet.VarifocalLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.125,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='PPYOLOEHead'),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                interval=1,
                keep_ratio=False,
                random_interp=True,
                random_size_range=(
                    320,
                    800,
                ),
                size_divisor=32,
                type='PPYOLOEBatchRandomResize'),
        ],
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        pad_size_divisor=32,
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='PPYOLOEDetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        block_cfg=dict(
            shortcut=False, type='PPYOLOEBasicBlock', use_alpha=False),
        deepen_factor=0.33,
        drop_block_cfg=None,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=1e-05, momentum=0.1, type='BN'),
        num_blocks_per_layer=3,
        num_csplayer=1,
        out_channels=[
            192,
            384,
            768,
        ],
        type='PPYOLOECSPPAFPN',
        use_spp=True,
        widen_factor=0.5),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=1000,
        score_thr=0.01),
    train_cfg=dict(
        assigner=dict(
            alpha=1,
            beta=6,
            eps=1e-09,
            num_classes=num_classes,
            topk=13,
            type='BatchTaskAlignedAssigner'),
        initial_assigner=dict(
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            num_classes=num_classes,
            topk=9,
            type='BatchATSSAssigner'),
        initial_epoch=30),
    type='YOLODetector')

optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr,
        momentum=0.9,
        nesterov=False,
        type='SGD',
        weight_decay=0.0005),
    paramwise_cfg=dict(norm_decay_mult=0.0),
    type='OptimWrapper')

param_scheduler = None
persistent_workers = True
resume = False
strides = [
    8,
    16,
    32,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        ann_file='./visdrone_test.json',
        data_prefix=dict(img='test/images'),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                height=640,
                interpolation='bicubic',
                keep_ratio=False,
                type='mmdet.FixShapeResize',
                width=640),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=val_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='./visdrone_test.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        height=640,
        interpolation='bicubic',
        keep_ratio=False,
        type='mmdet.FixShapeResize',
        width=640),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]

train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolov5_collate', use_ms_training=True),
    dataset=dict(
        ann_file='./visdrone_train.json',
        data_prefix=dict(img='train/images'),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PPYOLOERandomDistort'),
            dict(mean=(
                103.53,
                116.28,
                123.675,
            ), type='mmdet.Expand'),
            dict(type='PPYOLOERandomCrop'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='VisDrone'),
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PPYOLOERandomDistort'),
    dict(mean=(
        103.53,
        116.28,
        123.675,
    ), type='mmdet.Expand'),
    dict(type='PPYOLOERandomCrop'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        ann_file='./visdrone_val.json',
        data_prefix=dict(img='val/images'),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                height=640,
                interpolation='bicubic',
                keep_ratio=False,
                type='mmdet.FixShapeResize',
                width=640),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='VisDrone'),
    drop_last=False,
    num_workers=val_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='./visdrone_val.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')

visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(init_kwargs=dict(project='visdrone_s'), type='WandbVisBackend'),
    ])