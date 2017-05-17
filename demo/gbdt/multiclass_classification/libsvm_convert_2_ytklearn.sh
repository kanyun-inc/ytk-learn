#!/usr/bin/env bash
# binary_classification@label1,label2, multi_classification@label1,label2,..., regression

# make sure you are in path "ytk-learn"
# cd ../../..

mode="multi_classification@0,1,2,3,4,5"
x_delim="###"
y_delim=","
features_delim=","
feature_name_val_delim=":"

fs_scheme="local"
libsvm_data_path=("demo/data/libsvm/dermatology.train.libsvm" "demo/data/libsvm/dermatology.test.libsvm")
relative_path="demo/gbdt/multiclass_classification"
ytklearn_data_path=("${relative_path}/dermatology.train.ytklearn" "${relative_path}/dermatology.test.ytklearn")

for i in "${!libsvm_data_path[@]}"; do
	cur_libsvm_data_path=${libsvm_data_path[$i]}
	cur_ytklearn_data_path=${ytklearn_data_path[$i]}
	java -server -Xmx1000m -XX:-OmitStackTraceInFastThrow -classpath .:lib/*:config -Dlog4j.configuration=file:config/log4j.properties com.fenbi.ytklearn.utils.LibsvmConvertTool  \
    "${mode}" "${x_delim}" "${y_delim}" "${features_delim}" "${feature_name_val_delim}" "${fs_scheme}" "${cur_libsvm_data_path}" "${cur_ytklearn_data_path}"  >> log/info.log 2>&1;
done

if [ -s "${relative_path}/dermatology.train.ytklearn" ] && [ -s "${relative_path}/dermatology.test.ytklearn" ]; then
	echo "convert data format libsvm to ytk-learn complete!"
else
	echo "task failed, see details in log/info"
fi
