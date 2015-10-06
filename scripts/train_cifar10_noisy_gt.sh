#!/usr/bin/env bash

cd $(dirname ${BASH_SOURCE[0]})/../

CAFFE=external/caffe
EXP=external/exp

GLOG_logtostderr=1 $CAFFE/build/tools/caffe train \
    -solver models/cifar10_noisy_gt_solver.prototxt -gpu 0 \
    2>&1 | tee logs/cifar10_noisy_gt.log