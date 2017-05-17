#!/usr/bin/env bash

echo "########################################################################"
echo "######          offline predicting "
echo "######  usage: cd ytk-learn & sh bin/predict.sh"
echo "######  attention : '???' means the value must be filled by user himself"
echo "#########################################################################"

# model name(linear, fm, ffm, gbdt, gbmlr, gbsdt, gbhmlr, gbhsdt, multiclass_linear)
model_name=???

config_path="config/model/${model_name}.conf"

# data file for predicting
file_name=???

# train/test line python transform switch & script
transform="false"
transform_script_path="bin/transform.py"

# result save mode: PREDICT_RESULT_ONLY, LABEL_AND_PREDICT, PREDICT_AS_FEATURE
resultSaveMode="PREDICT_RESULT_ONLY"
resultFileSuffix="_"${model_name}"_"${resultSaveMode}

# max error data format tolerate number
max_error_tol=100
eval_metric="auc,mae"
#value or leafid
predict_type="value"

nohup java -server -Xmx1000m -XX:-OmitStackTraceInFastThrow -classpath .:lib/*:config -Dlog4j.configuration=file:config/log4j.properties com.fenbi.ytklearn.predictor.Predicts  \
    "${config_path}" "${model_name}" "${file_name}" "${transform}" "${transform_script_path}" "${resultSaveMode}" "${resultFileSuffix}" "${max_error_tol}" "${eval_metric}" "${predict_type}" >> log/info.log 2>&1 &
