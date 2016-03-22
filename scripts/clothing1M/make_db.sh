#!/usr/bin/env bash
# Make the auxiliary files and databases for the clothing1M dataset

cd $(dirname ${BASH_SOURCE[0]})/../../

CAFFE=external/caffe
EXP=external/exp

DATA_ROOT=$EXP/datasets/clothing1M
OUTPUT_DIR=$EXP/db/clothing1M
SNAPSHOTS_DIR=$EXP/snapshots/clothing1M
LOGS_DIR=logs/clothing1M

# setup output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p ${SNAPSHOTS_DIR}
mkdir -p ${LOGS_DIR}

# make auxiliary files
echo "Making auxiliary files"
python2 data/clothing1M/make_aux_data.py ${DATA_ROOT} ${OUTPUT_DIR}

# make databases
make_imageset() {
    $CAFFE/build/tools/convert_imageset \
        -backend lmdb -resize_height 256 -resize_width 256 \
        ${DATA_ROOT}/ ${OUTPUT_DIR}/$1.txt ${OUTPUT_DIR}/$1
}

make_labelset() {
    $CAFFE/build/tools/convert_labelset \
        -backend lmdb ${OUTPUT_DIR}/$1.txt ${OUTPUT_DIR}/$1
}

echo "Making clean train/val/test"
make_imageset clean_train
make_imageset clean_val
make_imageset clean_test

echo "Making ntype train/val/test"
make_imageset ntype_train
make_imageset ntype_val
make_imageset ntype_test

echo "Making mixed train"
make_imageset mixed_train_images
make_labelset mixed_train_label_clean
make_labelset mixed_train_label_noisy
make_labelset mixed_train_label_ntype

# create mean file
echo "Making mean file"
$CAFFE/build/tools/compute_image_mean \
    -backend lmdb \
    ${OUTPUT_DIR}/clean_train ${OUTPUT_DIR}/clothing_mean.binaryproto