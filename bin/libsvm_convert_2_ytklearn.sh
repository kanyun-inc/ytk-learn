#!/usr/bin/env bash
# binary_classification@label1,label2, multi_classification@label1,label2,..., regression
mode="???"
x_delim="###"
y_delim=","
features_delim=","
feature_name_val_delim=":"

fs_scheme="???"
libsvm_data_path="???"
ytklearn_data_path="???"

nohup java -server -Xmx1000m -XX:-OmitStackTraceInFastThrow -classpath .:lib/*:config -Dlog4j.configuration=file:config/log4j.properties com.fenbi.ytklearn.utils.LibsvmConvertTool  \
    "${mode}" "${x_delim}" "${y_delim}" "${features_delim}" "${feature_name_val_delim}" "${fs_scheme}" "${libsvm_data_path}" "${ytklearn_data_path}"  >> log/info.log 2>&1 &
