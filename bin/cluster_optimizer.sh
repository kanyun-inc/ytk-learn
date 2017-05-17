#!/usr/bin/env bash
echo "########################################################################"
echo "######          multi process training script"
echo "######  usage: cd ytk-learn & sh bin/cluster_optimizer.sh"
echo "######  attention : '???' means the value must be filled by user himself"
echo "#########################################################################"

# used for ssh login or create kill script
login_user=???

# process number
slave_num=???

# thread num in each process
thread_num=???
echo "slave num:${slave_num}, thread num:${thread_num}"

# use current machine as master
master_host=$(hostname)

# if you run more than one training tasks on the same host at the same time,
# different tasks must have different ports!
master_port=???
echo "master host:${master_host}, master port:${master_port}"

# model name(linear, fm, ffm, gbdt, gbmlr, gbsdt, gbhmlr, gbhsdt, multiclass_linear)
model_name=linear
echo "model name:${model_name}"

# hosts of all slaves(processes), separated by space
slave_hosts=(??? ??? ...)

# executable files location of every slave
slave_main_path="???"
echo "slave hosts:${slave_hosts}"
echo "slave main path:${slave_main_path}"

# config
properties_path="config/model/${model_name}.conf"
echo "config:${properties_path}"

# train/test line python transform switch & script
transform="false"
transform_script_path="bin/transform.py"

# kill old task
echo "kill old task..."
kill $(cat master_${master_port}.pid)
sh "kill_"${master_port}".sh"

# start ytk-mp4j master, application pid will be saved in master_${master_port}.pid file,
# default max memory is 1000m, master log saved in log/master.log
echo "start master..."
nohup java -server -Xmx1000m -classpath .:lib/*:config -Dlog4j.configuration=file:config/log4j_master.properties com.fenbi.mp4j.comm.CommMaster "${slave_num}" "${master_port}" >> log/master.log 2>&1 & echo $! > master_${master_port}.pid

# copy lib and config to all slaves
echo "copy lib and config..."
for slave_host in ${slave_hosts[@]}
do
    ssh ${login_user}@${slave_host} "mkdir "${slave_main_path}
    ssh ${login_user}@${slave_host} "mkdir "${slave_main_path}"/log"
    ssh ${login_user}@${slave_host} "cd "${slave_main_path}"; kill $(cat slave_"${master_port}".pid)"
    scp -r lib ${login_user}@${slave_host}":"${slave_main_path}
    scp -r config ${login_user}@${slave_host}":"${slave_main_path}
    scp -r bin ${login_user}@${slave_host}":"${slave_main_path}
done

# start local train workers, application pid will be saved in slave_${master_port}.pid,
# defaul max memory is 60000m, slave log slaved in log/slave.log
cmd="cd ${slave_main_path};nohup java -server -Xmx60000m -XX:-OmitStackTraceInFastThrow -classpath .:lib/*:config -Dlog4j.configuration=file:config/log4j_slave.properties com.fenbi.ytklearn.worker.LocalTrainWorker ${model_name} ${properties_path} ${transform_script_path} ${transform} ${login_user} ${master_host} ${master_port} ${thread_num} >> log/slave.log 2>&1 & echo \$! > slave_${master_port}.pid"
echo "cmd:${cmd}"
echo "start slavers..."
for slave_host in ${slave_hosts[@]}
do
    ssh ${login_user}@${slave_host} "${cmd}"
done
