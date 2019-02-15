#!/usr/bin/env bash

REL_SCRIPT_DIR=$(dirname "$0")
INITIAL_DIR=$(pwd)
cd $REL_SCRIPT_DIR
ABS_SCRIPT_DIR=$(pwd)

cd datasets

# Get mnist data
curl http://filecremers3.informatik.tu-muenchen.de/~dl4cv/mnist_train.zip -o mnist_train.zip
unzip mnist_train.zip
rm mnist_train.zip


# Get keypoints data
curl http://filecremers3.informatik.tu-muenchen.de/~dl4cv/training.zip -o training.zip
unzip training.zip
rm training.zip


cd $INITIAL_DIR
