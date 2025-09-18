#!/bin/sh

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}

#Pascal VOC
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/voc20.json" \
 DATASETS.TEST \(\"voc_2012_train_sem_seg\"\,\) \
 $opts
#  MODEL.WEIGHTS None \

# COCO-Object w/o bg
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/coco_object.json" \
 DATASETS.TEST \(\"coco_2017_test_object_wo_bg_stuff_all_sem_seg\"\,\) \
 $opts
#  MODEL.WEIGHTS None \

# ADE
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade150.json" \
 DATASETS.TEST \(\"ade20k_150_test_sem_seg\"\,\) \
 $opts
#  MODEL.WEIGHTS None \

#Pascal Context 59 w/o background
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON  "datasets/pc59.json" \
 DATASETS.TEST \(\"context_59_test_sem_seg\"\,\) \
 MODEL.WEIGHTS None \
 $opts

 #Cityscapes
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/cityscapes.json" \
 DATASETS.TEST \(\"cityscapes_fine_sem_seg_val\"\,\) \
 MODEL.WEIGHTS None \
 $opts

cat $output/eval/log.txt | grep copypaste