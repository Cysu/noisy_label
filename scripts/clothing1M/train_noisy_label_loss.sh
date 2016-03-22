#!/usr/bin/env bash

cd $(dirname ${BASH_SOURCE[0]})/../../

CAFFE=external/caffe
EXP=external/exp

GLOG_logtostderr=1 $CAFFE/build/tools/caffe train \
    -solver models/clothing1M/noisy_label_loss_solver.stage1.prototxt \
    -weights $EXP/snapshots/clothing1M/noisy_label_loss_iter_0.caffemodel \
    -gpu 0 \
    2>&1 | tee logs/clothing1M/noisy_label_loss.stage1.log

GLOG_logtostderr=1 $CAFFE/build/tools/caffe train \
    -solver models/clothing1M/noisy_label_loss_solver.stage2.prototxt \
    -snapshot $EXP/snapshots/clothing1M/noisy_label_loss_iter_100000.solverstate \
    -gpu 0 \
    2>&1 | tee logs/clothing1M/noisy_label_loss.stage2.log