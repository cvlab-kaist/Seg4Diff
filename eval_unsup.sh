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
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/voc21.json" \
 DATASETS.TEST \(\"voc_2012_train_background_sem_seg\"\,\) \
 MODEL.WEIGHTS None \
 $opts

#Pascal Context 59 w/ background
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON  "datasets/pc59.json" \
 DATASETS.TEST \(\"context_59_test_background_sem_seg\"\,\) \
 MODEL.WEIGHTS None \
 $opts

# COCO-Object with bg - test
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/coco_object.json" \
 DATASETS.TEST \(\"coco_2017_test_object_stuff_all_sem_seg\"\,\) \
 MODEL.WEIGHTS None \
 $opts

# COCO-Stuff-27
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/coco_stuff_27.json" \
 DATASETS.TEST \(\"coco_2017_test_27_stuff_all_sem_seg\"\,\) \
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

# ADE
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade150.json" \
 DATASETS.TEST \(\"ade20k_150_test_sem_seg\"\,\) \
 MODEL.WEIGHTS None \
 $opts

cat $output/eval/log.txt | grep copypaste