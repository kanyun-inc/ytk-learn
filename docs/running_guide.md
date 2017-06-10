

### Building Ytk-learn

Ytk-learn is built using Apache Maven, run:

```shell
sh tool/package.sh
```

```ytk-learn.zip``` package is created in the "target" directory. You can put it anywhere you want to run ,and run ```unzip ytk-learn.zip```, then you will see the following lists of directorys below in ytk-learn:
- ```bin```: running scripts including training scrips, data format converting scripts, offline prediction script.
- ```config```: configuration files including log4j and model configurations.
- ```log```: when you run some scripts, corresponding logs will be generated in this directory. 
- ```demo```: several demos of each model in this directory.

[ytk-learn.zip](https://github.com/yuantiku/ytk-learn/releases) is avaliable for downloading. 

### Training Platform and Corresponding Running Scripts

- single machine
- commom cluster
- spark cluster
- hadoop cluster
- easily extended to other computation platforms

|                | Linux                                    | Mac OS                                   | Windows(cygwin)                          |
| :------------- | :--------------------------------------- | :--------------------------------------- | :--------------------------------------- |
| single machine | [local_optimizer.sh](../bin/local_optimizer.sh) | [local_optimizer.sh](../bin/local_optimizer.sh) | [win_local_optimizer.bat](../bin/win_local_optimizer.bat)/[cygwin_local_optimizer.sh](../bin/cygwin_local_optimizer.sh) |
| common cluster | [cluster_optimizer.sh](../bin/cluster_optimizer.sh) | no                                       | no                                       |
| spark cluster  | [spark_optimizer.sh](../bin/spark_optimizer.sh) | no                                       | no                                       |
| hadoop         | [hadoop_optimizer.sh](../bin/hadoop_optimizer.sh) | no                                       | no                                       |

Ytk-learn uses master-slave communication mode based on ytk-mp4j. Master node is responsible for the coordination of slave nodes. Slave nodes are real workers. If you want to run more than one training task in the same host at the same time, different tasks must have different master ports.

The properties of thread number, master host, master port, slave hosts, model name, configuration path, data transformation and runing commands can be set in these scripts. 

Double click the [win_local_optimizer.bat](../bin/win_local_optimizer.bat) to start training on windows.

### Storage Sytem

- local disk
- hdfs
- easily extended to other storage systems



|                  | single | cluster | spark cluster | hadoop cluster |
| ---------------- | :----: | :-----: | :-----------: | :------------: |
| local filesystem |  yes   |   yes   |      no       |       no       |
| hdfs filesystem  |  yes   |   yes   |      yes      |      yes       |



### Train/Test Data Splitting Manner

Most optimization methods use data parallel in cluster training, then ways of reasonably splitting train/test data are very important. Fortunately,  in Spark/Hadoop cluster, the train data has been splitted uniformly in each executor/reducer,  but in common cluster,  the way of splitting the data must be assigned. If you have not assigned your data(assigned : false), there are two options to choose, "lines_avg" and "files_vg".

### Configuration

The [configurations](../config/model) for our models mainly consist of four parts: data, model, feature and optimization. 

- Data: data-related configuration, such as training data path, testing data path and [data format](data_format.md).


- Model: model-related configuration, such as model path, user-provided feature dict path and whether to continue training.
- Feature: feature-processing-related configuration. 
- Optimization: training-related configuration, such as hyper parameters.

### Logs

Logs in ytk-learn are very useful. You can monitor task procedure, see importance information such as evaluation results and find detailed error information when the program is not running as you expected.

After starting up training,  you can use ```tail -f log/master.log``` to watch process,  most errors and exceptions are printed in this log file. If training is blocked or nothing about error or exception can be found in ```master.log```,  you must to check ```slave.log``` or ```slave_error.log```, if there is a ```ConnectionException```, you can try to set ```master_host``` to be ```127.0.0.1```. In the spark/hadoop yarn, you can use ``` yarn logs -applicationId your_application_id``` command to get slave's logs.

| log file                      | details                                  | relevant scripts                         |
| :---------------------------- | :--------------------------------------- | :--------------------------------------- |
| log/master.log                | most logs are saved in this file including master startup logs, slave connecting logs, slave reported logs(info, error, exception). | all training shell scripts               |
| log/master_error.log          | master error logs, including most slave error logs | all training shell scripts               |
| log/master_debug.log          | master debug logs, including most slave debug logs | all training shell scripts               |
| log/slave.log                 | slave local logs(most slave logs are sent to master) | local_optimizer.sh/cluster_optimizer.sh  |
| log/slave_error.log           | slave local error logs(most slave error logs are sent to master) | local_optimizer.sh/cluster_optimizer.sh  |
| log/slave_debug.log           | slave local debug logs(most slave debug logs are sent to master) | local_optimizer.sh/cluster_optimizer.sh  |
| log/info.log                  | other info logs                          | predict.sh/libsvm_convert\_2\_ytklearn.sh |
| log/error.log                 | other error logs                         | predict.sh/libsvm_convert\_2\_ytklearn.sh |
| log/debug.log                 | other debug logs                         | predict.sh/libsvm_convert\_2\_ytklearn.sh |
| log/yarn\_${master_port}\.log | yarn job logs(spark, hadoop)             | spark_optimizer.sh/hadoop_optimizer.sh   |

##### Tips:

- ```tail -f log/master.log | grep "train loss"```: check train loss
- ```tail -f log/master.log | grep "test loss"```: check test loss
- ```tail -f log/master.log | grep "auc\|rmse\|confusion_matrix\|mae"```: check metrics.

### Demo

[linear](../demo/linear)

[multiclass_linear](../demo/multiclass_linear)

[gbdt](../demo/gbdt)

[fm](../demo/fm)

[ffm](../demo/ffm)

[gbmlr](../demo/gbmlr)

[gbsdt](../demo/gbsdt)

[gbhmlr](../demo/gbhmlr)

[gbhsdt](../demo/gbhsdt)
