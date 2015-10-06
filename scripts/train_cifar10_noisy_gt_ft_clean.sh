#!/usr/bin/env bash

cd $(dirname ${BASH_SOURCE[0]})/../

CAFFE=external/caffe
EXP=external/exp

GLOG_logtostderr=1 $CAFFE/build/tools/caffe train \
    -solver models/cifar10_noisy_gt_ft_clean_solver.prototxt \
    -weights $EXP/models/cifar10_clean_iter_5000.caffemodel \
    -gpu 0 \
    2>&1 | tee logs/cifar10_noisy_gt_ft_clean.log