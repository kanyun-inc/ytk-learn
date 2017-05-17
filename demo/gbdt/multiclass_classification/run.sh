#!/bin/bash

# make sure you are in path "ytk-learn"
# cd ../../..

sh demo/gbdt/multiclass_classification/libsvm_convert_2_ytklearn.sh
sh demo/gbdt/multiclass_classification/local_optimizer.sh
sh demo/gbdt/multiclass_classification/predict.sh
