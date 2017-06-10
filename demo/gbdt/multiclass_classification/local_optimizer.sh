#!/usr/bin/env bash

# make sure you are in path "ytk-learn"
# cd ../../..

# thread number
thread_num=1

# use current machine as master, if an ConnectException occurs in slave.log，
# try to set master_host=127.0.0.1
master_host=$(hostname)

# if you run more than one training tasks on the same host at the same time,
# different tasks must have different ports!
master_port=65534
echo "master host:${master_host}, master port:${master_port}"

# model name
model_name=gbdt
echo "model name:${model_name}"

# config
properties_path="demo/gbdt/multiclass_classification/local_gbdt.conf"
echo "config:${properties_path}"

# transform
transform="false"
transform_script_path="bin/transform.py"

echo "kill old task..."
kill $(cat master_${master_port}.pid)
kill $(cat slave_${master_port}.pid)

echo "start master..."
nohup java -server -Xmx2000m -classpath .:lib/*:config -Dlog4j.configuration=file:config/log4j_master.properties com.fenbi.mp4j.comm.CommMaster 1 "${master_port}" >> log/master.log 2>&1 & echo $! > master_${master_port}.pid

echo "start slave..."
nohup java -server -Xmx60000m -XX:-OmitStackTraceInFastThrow -classpath .:lib/*:config -Dlog4j.configuration=file:config/log4j_slave.properties com.fenbi.ytklearn.worker.LocalTrainWorker  \
    "${model_name}" "${properties_path}" "${transform_script_path}" "${transform}" user "${master_host}" "${master_port}" "${thread_num}" >> log/slave.log 2>&1 & echo $! > slave_${master_port}.pid
wait
