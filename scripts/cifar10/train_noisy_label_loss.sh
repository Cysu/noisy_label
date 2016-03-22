#!/usr/bin/env bash

cd $(dirname ${BASH_SOURCE[0]})/../../

CAFFE=external/caffe
EXP=external/exp

GLOG_logtostderr=1 $CAFFE/build/tools/caffe train \
    -solver models/cifar10/noisy_label_loss_solver.prototxt \
    -weights $EXP/snapshots/cifar10/noisy_label_loss_iter_0.caffemodel \
    -gpu 0 \
    2>&1 | tee logs/cifar10/noisy_label_loss.log