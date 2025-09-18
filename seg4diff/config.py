# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_seg4diff_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # data config
    # select the dataset mapper
    cfg.INPUT.STRONG_AUG = False
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"

    cfg.DATASETS.VAL_ALL = ("coco_2017_val_all_stuff_sem_seg",)
    cfg.DATASETS.CAPTION_DIR = None

    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32


    cfg.MODEL.CLIP_PIXEL_MEAN = [122.7709383, 116.7460125, 104.09373615]
    cfg.MODEL.CLIP_PIXEL_STD = [68.5005327, 66.6321579, 70.3231630]
    # three styles for clip classification, crop, mask, cropmask
    
    cfg.MODEL.GT_ONLY_PROMPT = False
    cfg.MODEL.CLIP_FILTERING = False
    cfg.MODEL.WITH_SOS_EOS = False
    
    # zero shot config
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = "datasets/coco.json"
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = "datasets/voc21.json"
    cfg.MODEL.SEM_SEG_HEAD.MEAN_HEADS = False
    cfg.MODEL.SEM_SEG_HEAD.CLASS_WEIGHT = 0.0
    cfg.MODEL.SEM_SEG_HEAD.MASK_WEIGHT = 0.02
    cfg.MODEL.SEM_SEG_HEAD.DICE_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.COST_THRESH = None
    
    cfg.MODEL.BACKGROUND_NAME = "background"
    cfg.MODEL.BACKBONE.NUM_TOKENS = 171
    cfg.MODEL.BACKBONE.FP16 = False
    cfg.MODEL.BACKBONE.SOFTMAX = True
    cfg.MODEL.BACKBONE.SOFTMAX_TEMPS = [1.0, 1.0, 1.0, 1.0]
    cfg.MODEL.BACKBONE.SEED = None
    
    cfg.MODEL.BACKBONE.ATTENTION_LAYERS = [i for i in range(24)]
    cfg.MODEL.BACKBONE.USE_LORA = False
    cfg.MODEL.BACKBONE.LORA_LAYERS = None
    cfg.MODEL.BACKBONE.LORA_BLOCKS = None
    cfg.MODEL.BACKBONE.LORA_RANK = 16
    cfg.MODEL.BACKBONE.LORA_WEIGHTS = None
    cfg.MODEL.BACKBONE.USE_LEARNABLE_TOKENS = False
    cfg.MODEL.BACKBONE.KEEP_HEAD = False
    cfg.MODEL.BACKBONE.MAX_HEAD = False
    cfg.MODEL.BACKBONE.SOFTMAX_I2T_ONLY = False

    # optimization settings
    cfg.MODEL.BACKBONE.ENABLE_CPU_OFFLOAD = False
    cfg.MODEL.BACKBONE.ENABLE_TILING = False
    cfg.MODEL.BACKBONE.ENABLE_SLICING = False
    cfg.MODEL.BACKBONE.ATTENTION_HEAD_REDUCTION = None
    cfg.MODEL.BACKBONE.GRADIENT_CHECKPOINTING = False

    cfg.MODEL.W = 0.1
    cfg.MODEL.TEMPERATURE = 0.07
    cfg.MODEL.OUTPUT_POWER = 2.0
    cfg.MODEL.NORM_BEFORE_MERGE = True
    cfg.MODEL.NORM_AFTER_MERGE = True
    cfg.MODEL.SIGMOID_AFTER_MERGE = False
    cfg.MODEL.HARD_ASSIGNMENT = True
    cfg.MODEL.NEGATIVE_BG = False
    cfg.MODEL.EVAL_MODE = None
    cfg.MODEL.NUM_INFERENCE_STEPS = 28
    cfg.MODEL.NOISE_STEPS = 14
    cfg.MODEL.KL_THRESHOLD = 0.001
    cfg.MODEL.SCALE_SHIFT_AFTER_MERGE = False

    cfg.MODEL.MASK_LOSS_WEIGHT = 0.1
    cfg.MODEL.DICE_LOSS_WEIGHT = 0.01
    cfg.MODEL.FM_WEIGHT = 1.
    cfg.MODEL.USE_ATTN_MLP = False
    cfg.MODEL.SHORT_CAPTION = False
    cfg.MODEL.CAPTION_DROPOUT = 0.0
    cfg.MODEL.MAX_MASK = 32

    cfg.MODEL.GENERATE_VAL_IMAGES = True
    cfg.MODEL.VISUALIZE_ATTENTION = False
    cfg.MODEL.VISUALIZE_VAL_ATTENTION = False
    cfg.MODEL.USE_SEM_SEG_HEAD = False

    cfg.MODEL.LAYER=None
    cfg.MODEL.HEAD=None

    cfg.MODEL.EVAL_UNSUP = False

    cfg.LOG_NAME = None