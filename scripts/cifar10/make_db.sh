#!/usr/bin/env bash
# Make the auxiliary files and databases for the cifar10 dataset

cd $(dirname ${BASH_SOURCE[0]})/../../

CAFFE=external/caffe
EXP=external/exp

if [[ $# -ne 0 ]] && [[ $# -ne 1 ]]; then
    echo "Usage: $(basename $0) [noise_level=0.5]"
    echo "    noise_level     Level of label noise to be generated"
    exit
fi

if [[ $# -eq 1 ]]; then
  NOISE_LEVEL=$1
else
  NOISE_LEVEL=0.5
fi

DATA_ROOT=$EXP/datasets/cifar10
OUTPUT_DIR=$EXP/db/cifar10
SNAPSHOTS_DIR=$EXP/snapshots/cifar10
LOGS_DIR=logs/cifar10

# setup output directory
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${SNAPSHOTS_DIR}
mkdir -p ${LOGS_DIR}

# make auxiliary files
echo "Making auxiliary files"
python2 data/cifar10/convert_raw_dataset.py ${DATA_ROOT} ${OUTPUT_DIR}
python2 data/cifar10/generate_noisy_labels.py ${OUTPUT_DIR} --level ${NOISE_LEVEL}
python2 data/cifar10/estimate_matrix_c.py ${OUTPUT_DIR}
python2 tools/convert_to_blobproto.py \
    ${OUTPUT_DIR}/matrix_q.pkl ${OUTPUT_DIR}/true_matrix_q.binaryproto
python2 tools/convert_to_blobproto.py \
    ${OUTPUT_DIR}/matrix_c.pkl ${OUTPUT_DIR}/matrix_c.binaryproto

# make databases
make_imageset() {
    $CAFFE/build/tools/convert_imageset \
        -backend lmdb -resize_height 32 -resize_width 32 \
        ${OUTPUT_DIR}/ ${OUTPUT_DIR}/$1.txt ${OUTPUT_DIR}/$1
}

make_labelset() {
    $CAFFE/build/tools/convert_labelset \
        -backend lmdb ${OUTPUT_DIR}/$1.txt ${OUTPUT_DIR}/$1
}

make_imageset 'test'
make_imageset 'clean_train'
make_imageset 'ntype_train'
make_imageset 'ntype_test'
make_imageset 'mixed_train'
make_imageset 'mixed_train_images'

make_labelset 'mixed_train_label_clean'
make_labelset 'mixed_train_label_noisy'
make_labelset 'mixed_train_label_ntype'

$CAFFE/build/tools/compute_image_mean \
    -backend lmdb \
    ${OUTPUT_DIR}/mixed_train_images ${OUTPUT_DIR}/cifar10_mean.binaryproto