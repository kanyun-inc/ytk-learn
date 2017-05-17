#!/usr/bin/env bash

# make sure you are in path "ytk-learn"
# cd ../../..

model_name=gbdt
config_path="demo/gbdt/binary_classification/local_gbdt.conf"
file_dir="demo/gbdt/binary_classification/agaricus.test.ytklearn"

# transform
transform="false"
transform_script_path="bin/transform.py"

# PREDICT_RESULT_ONLY, LABEL_AND_PREDICT, PREDICT_AS_FEATURE, TREE_LEAF_AS_FEATURE
resultSaveMode="PREDICT_RESULT_ONLY"
resultFileSuffix="_"${model_name}"_"${resultSaveMode}

max_error_tol=0
eval_metric="auc,confusion_matrix"
#value or leafid
predict_type="value"

nohup java -server -Xmx1000m -XX:-OmitStackTraceInFastThrow -classpath .:lib/*:config -Dlog4j.configuration=file:config/log4j.properties com.fenbi.ytklearn.predictor.Predicts  \
    "${config_path}" "${model_name}" "${file_dir}" "${transform}" "${transform_script_path}" "${resultSaveMode}" "${resultFileSuffix}" "${max_error_tol}" "${eval_metric}" "${predict_type}" >> log/info.log 2>&1 &
wait