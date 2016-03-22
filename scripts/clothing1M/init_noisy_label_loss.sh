#!/usr/bin/env bash

cd $(dirname ${BASH_SOURCE[0]})/../../

EXP=external/exp

rm $EXP/snapshots/clothing1M/noisy_label_loss_iter_0.caffemodel

echo "Copy clothing1M clean pretrained parameters"
python2 tools/copy_params.py \
    --source-model models/clothing1M/clean_trainval.prototxt \
    --source-weights $EXP/snapshots/clothing1M/clean_iter_15000.caffemodel \
    --target-model models/clothing1M/noisy_label_loss_trainval.stage1.prototxt \
    --target-weights $EXP/snapshots/clothing1M/noisy_label_loss_iter_0.caffemodel \
    --layers-dict "{name: name + '_clean' for name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']}"

echo "Copy clothing1M ntype pretrained parameters"
python2 tools/copy_params.py \
    --source-model models/clothing1M/ntype_trainval.prototxt \
    --source-weights $EXP/snapshots/clothing1M/ntype_iter_5000.caffemodel \
    --target-model models/clothing1M/noisy_label_loss_trainval.stage1.prototxt \
    --target-weights $EXP/snapshots/clothing1M/noisy_label_loss_iter_0.caffemodel \
    --layers-dict "{name: name + '_ntype' for name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']}"