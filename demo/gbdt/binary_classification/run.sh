#!/bin/bash

# make sure you are in path "ytk-learn"
# cd ../../..

sh demo/gbdt/binary_classification/libsvm_convert_2_ytklearn.sh
sh demo/gbdt/binary_classification/local_optimizer.sh
sh demo/gbdt/binary_classification/predict.sh
