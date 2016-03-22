#!/usr/bin/env bash

cd $(dirname ${BASH_SOURCE[0]})/../../

CAFFE=external/caffe
EXP=external/exp

GLOG_logtostderr=1 $CAFFE/build/tools/caffe train \
    -solver models/clothing1M/clean_solver.prototxt \
    -weights $EXP/snapshots/bvlc_reference_caffenet.caffemodel \
    -gpu 0 \
    2>&1 | tee logs/clothing1M/clean.log