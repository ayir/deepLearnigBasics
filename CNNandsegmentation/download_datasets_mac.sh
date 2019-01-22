#!/usr/bin/env bash

REL_SCRIPT_DIR=$(dirname "$0")
INITIAL_DIR=$(pwd)
cd $REL_SCRIPT_DIR
ABS_SCRIPT_DIR=$(pwd)

cd datasets
rm -r cifar10_train.p

# Get CIFAR10
curl http://filecremers3.informatik.tu-muenchen.de/~dl4cv/cifar10_train.zip -o cifar10_train.zip
tar -xzvf cifar10_train.zip
rm cifar10_train.zip

# Get segmentation dataset
curl http://filecremers3.informatik.tu-muenchen.de/~dl4cv/segmentation_data.zip -o segmentation_data.zip
unzip segmentation_data.zip
rm segmentation_data.zip

# Get segmentation dataset test
curl http://filecremers3.informatik.tu-muenchen.de/~dl4cv/segmentation_data_test.zip -o segmentation_data_test.zip
unzip segmentation_data_test.zip
rm segmentation_data_test.zip

cd $INITIAL_DIR
