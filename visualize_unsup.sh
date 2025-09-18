#!/bin/bash

OUTPUT_DIR=$1

mkdir -p $OUTPUT_DIR

# # COCO-Stuff
# python demo/demo.py \
#  --config-file ./configs/sd3_eval_unsup.yaml \
#  --input /mnt/data3/catseg_dataset/coco/val2017/*.jpg \
#  --output $OUTPUT_DIR \
#  --opts \
#     MODEL.GT_ONLY_PROMPT True \
#     DATASETS.TEST \(\"coco_2017_test_stuff_all_sem_seg\"\,\) \
#     MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/coco.json" \
#     MODEL.KL_THRESHOLD 0.03
    

# python demo/demo.py \
#  --config-file ./configs/sd3_eval_unsup.yaml \
#  --input /home/cvlab15/project/chyun/cat3dog1.png \
#  --output $OUTPUT_DIR \
#  --opts \
#     MODEL.GT_ONLY_PROMPT False \
#     DATASETS.TEST \(\"coco_2017_test_stuff_all_sem_seg\"\,\) \
#     MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/coco.json"


# python demo/demo.py \
#  --config-file ./configs/sd3_eval_weakly.yaml \
#  --input /home/cvlab15/project/chyun/*.png \
#  --output $OUTPUT_DIR \
#  --opts \
#     MODEL.GT_ONLY_PROMPT True \
#     DATASETS.TEST \(\"coco_2017_test_stuff_all_sem_seg\"\,\) \
#     MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/coco.json"

# # ADE20K-150
# python demo/demo.py \
#  --config-file ./configs/sd3_eval_unsup.yaml \
#  --input /mnt/data3/catseg_dataset/ADEChallengeData2016/images/validation/*.jpg \
#  --output $OUTPUT_DIR \
#  --opts \
#     MODEL.GT_ONLY_PROMPT True \
#     DATASETS.TEST \(\"ade20k_150_test_sem_seg\"\,\) \
#     MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade150.json"

# # ADE20K-847
# python demo/demo.py \
#  --config-file ./configs/ca_mask.yaml \
#  --input /media/dataset1/catseg_dataset/ADEChallengeData2016/images/validation/*.jpg \
#  --output $OUTPUT_DIR \
#  --opts \
#     MODEL.GT_ONLY_PROMPT True \
#     DATASETS.TEST \(\"ade20k_full_sem_seg_freq_val_all\"\,\) \
#     MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade847.json" \
#     MODEL.WEIGHTS None

# # Pascal VOC
# python demo/demo.py \
#  --config-file ./configs/ca_mask.yaml \
#  --input /media/dataset1/catseg_dataset/VOCdevkit/VOC2012/JPEGImages/*.jpg \
#  --output $OUTPUT_DIR \
#  --opts \
#     MODEL.GT_ONLY_PROMPT True \
#     DATASETS.TEST \(\"voc_2012_test_sem_seg\"\,\) \
#     MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/voc20.json" \
#     MODEL.WEIGHTS None

# Pascal VOC-b
python demo/demo.py \
 --config-file ./configs/sd3_eval_unsup.yaml \
 --input /mnt/data3/catseg_dataset/VOCdevkit/VOC2012/JPEGImages/*.jpg \
 --output $OUTPUT_DIR \
 --opts \
    MODEL.GT_ONLY_PROMPT True \
    DATASETS.TEST \(\"voc_2012_test_background_sem_seg\"\,\) \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/voc21.json" \
    MODEL.WEIGHTS None

# # Pascal Context 59
# python demo/demo.py \
#  --config-file ./configs/sd3_eval_unsup.yaml \
#  --input /mnt/data3/catseg_dataset/VOCdevkit/VOC2010/JPEGImages/*.jpg \
#  --output $OUTPUT_DIR \
#  --opts \
#     MODEL.GT_ONLY_PROMPT True \
#     DATASETS.TEST \(\"context_59_test_sem_seg\"\,\) \
#     MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/pc59.json" \
#     MODEL.WEIGHTS None

# python demo/demo.py \
#  --config-file ./configs/sd3_eval_weakly.yaml \
#  --input /mnt/data3/catseg_dataset/VOCdevkit/VOC2010/JPEGImages/2008_000501.jpg \
#  --output $OUTPUT_DIR \
#  --opts \
#     MODEL.GT_ONLY_PROMPT True \
#     DATASETS.TEST \(\"context_59_test_sem_seg\"\,\) \
#     MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/pc59.json" \
#     MODEL.WEIGHTS None

# # Pascal Context 459
# python demo/demo.py \
#  --config-file ./configs/ca_mask.yaml \
#  --input /media/dataset1/catseg_dataset/VOCdevkit/VOC2010/JPEGImages/*.jpg \
#  --output $OUTPUT_DIR \
#  --opts \
#     MODEL.GT_ONLY_PROMPT True \
#     DATASETS.TEST \(\"context_459_test_sem_seg\"\,\) \
#     MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/pc459.json" \
#     MODEL.WEIGHTS None