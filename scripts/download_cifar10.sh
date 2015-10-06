#!/usr/bin/env bash

cd $(dirname ${BASH_SOURCE[0]})/../

EXP=external/exp

mkdir -p $EXP/datasets/cifar10
cd $EXP/datasets/cifar10

wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xf cifar-10-python.tar.gz
mv cifar-10-batches-py/* .
rm -rf cifar-10-batches-py
rm cifar-10-python.tar.gz