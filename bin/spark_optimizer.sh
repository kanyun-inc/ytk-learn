#!/usr/bin/env bash
echo "########################################################################"
echo "######          spark cluster training script"
echo "######  usage: cd ytk-learn & sh bin/spark_optimizer.sh & "
echo "######  attention : '???' means the value must be filled by user himself"
echo "#########################################################################"

# used for ssh login or create kill script
login_user=???

# used for distinguishing different job name
user=ytk-learn

# spark executor numbers
slave_num=4

# thread num in each slave
thread_num=4
echo "slave num:${slave_num}, thread num:${thread_num}"

# use current machine as master
master_host=$(hostname)

# if you run more than one training tasks on the same host at the same time,
# different tasks must have different ports!
master_port=???
echo "master host:${master_host}, master port:${master_port}"

# model name
model_name=???
echo "model name:${model_name}"

# model name(linear, fm, ffm, gbdt, gbmlr, gbsdt, gbhmlr, gbhsdt, multiclass_linear)
properties_path="config/model/${model_name}.conf"

properties="${model_name}.conf"
echo "config path:${properties_path}, config name:${properties}"

# train/test line python transform switch & script
transform="false"
transform_script_path="bin/transform.py"
transform_script="transform.py"

# spark cluster queue name
spark_queue=???
echo "spark queue:${spark_queue}"

# spark training restart times
max_spark_restart=1
echo "max spark retart time:${max_spark_restart}"

# spark executor memory
spark_executor_memory=20g
echo "spark executor memory:${spark_executor_memory}"

# start master & spark
for (( i = 1, error_code = 201; error_code != 0 && i <= ${max_spark_restart}; i++ ))
do
    echo "kill spark application"
    python bin/yarn_job_kill.py log/yarn_${master_port}.log

    echo "restart $i times!"

    echo "kill old master"
    kill $(cat master_${master_port}.pid)
    sh "kill_"${master_port}".sh"

    # start ytk-mp4j master, application pid will be saved in master_${master_port}.pid file,
    # default max memory is 1000m, master log saved in log/master.log
    # master doesn't run in spark
    echo "start master..."
    nohup java -server -Xmx1000m -classpath .:lib/*:config -Dlog4j.configuration=file:config/log4j_master.properties com.fenbi.mp4j.comm.CommMaster ${slave_num} ${master_port} >> log/master.log 2>&1 & echo $! > master_${master_port}.pid

    # start spark training worker(only support YARN)
    # if you want to know whether or not your spark job running successfully, see log/yarn_${master_port}.log
    echo "start spark..."
    nohup spark-submit --name "${user} ${model_name} training on spark" --driver-cores 2 --executor-cores  $[${thread_num}+1] \
     --num-executors ${slave_num} --driver-memory 2g --executor-memory ${spark_executor_memory} --master yarn-cluster \
     --supervise --queue ${spark_queue} --files "${properties_path},${transform_script_path}" --class \
     com.fenbi.ytklearn.worker.SparkTrainWorker lib/ytk-learn.jar \
     "${model_name}" "${properties_path}" "${properties}" "${transform_script}" "${transform}" "${login_user}" "${master_host}" "${master_port}"  \
     "${slave_num}" "${thread_num}" >> log/yarn_${master_port}.log
    error_code=$?
    echo "error_code:${error_code}"
    perror $error_code

done