#!/usr/bin/env bash

cd $(dirname ${BASH_SOURCE[0]})/../

EXP=external/exp

rm $EXP/models/cifar10_noisy_label_loss_iter_0.caffemodel

echo "Copy cifar10 clean pretrained parameters"
python2 tools/copy_params.py \
    --source-model models/cifar10_clean_trainval.prototxt \
    --source-weights $EXP/models/cifar10_clean_iter_5000.caffemodel \
    --target-model models/cifar10_noisy_label_loss_trainval.prototxt \
    --target-weights $EXP/models/cifar10_noisy_label_loss_iter_0.caffemodel \
    --layers-dict "{name: name + '_clean' for name in ['conv1', 'conv2', 'conv3', 'ip1', 'ip2']}"

echo "Copy cifar10 ntype pretrained parameters"
python2 tools/copy_params.py \
    --source-model models/cifar10_ntype_trainval.prototxt \
    --source-weights $EXP/models/cifar10_ntype_iter_1500.caffemodel \
    --target-model models/cifar10_noisy_label_loss_trainval.prototxt \
    --target-weights $EXP/models/cifar10_noisy_label_loss_iter_0.caffemodel \
    --layers-dict "{name: name + '_ntype' for name in ['conv1', 'conv2', 'conv3']}"