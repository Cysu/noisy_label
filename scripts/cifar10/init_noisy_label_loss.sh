#!/usr/bin/env bash

cd $(dirname ${BASH_SOURCE[0]})/../../

EXP=external/exp

rm $EXP/snapshots/cifar10/noisy_label_loss_iter_0.caffemodel

echo "Copy cifar10 clean pretrained parameters"
python2 tools/copy_params.py \
    --source-model models/cifar10/clean_trainval.prototxt \
    --source-weights $EXP/snapshots/cifar10/clean_iter_5000.caffemodel \
    --target-model models/cifar10/noisy_label_loss_trainval.prototxt \
    --target-weights $EXP/snapshots/cifar10/noisy_label_loss_iter_0.caffemodel \
    --layers-dict "{name: name + '_clean' for name in ['conv1', 'conv2', 'conv3', 'ip1', 'ip2']}"

echo "Copy cifar10 ntype pretrained parameters"
python2 tools/copy_params.py \
    --source-model models/cifar10/ntype_trainval.prototxt \
    --source-weights $EXP/snapshots/cifar10/ntype_iter_1500.caffemodel \
    --target-model models/cifar10/noisy_label_loss_trainval.prototxt \
    --target-weights $EXP/snapshots/cifar10/noisy_label_loss_iter_0.caffemodel \
    --layers-dict "{name: name + '_ntype' for name in ['conv1', 'conv2', 'conv3']}"