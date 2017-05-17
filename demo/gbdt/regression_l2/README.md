#Regression_l2

Regression_l2 is a regression task with l2 loss function(mean squared error). The following steps train a GBDT model by **feature-parallel(exact greedy)** algorithm and use this model to get predictions. The script ``run.sh`` contains the whole steps.

**Notes**: 

If you want to train with data-parallel(histogram approximate) algorithm, just change ``optimization.tree_maker`` from "feature" to "data".  If you want to train with l1 loss function(mean absolute error), set ``optimization.loss_function`` to "l1".

#### 1. Train model####

```
# make sure you are in path "ytk-learn"
# cd ../../.. (go to path "ytk-learn")
sh demo/gbdt/regression_l2/local_optimizer.sh
```

Monitor train info in directory "ytk-learn/log"

```tail -f log/master.log```

Training is finished after you see "exit code:0" in log/master.log

Mode and feature importance files are saved in "demo/gbdt/regression_l2" which is configured in  "local_gbdt.conf"

**Notes**：

This demo converts data format by using `transform.py` which converts data in memory during data reading process. For each data don't for get to set ``data.max_feature_dim`` in local_gbdt.conf.

#### 2. Get predictions  for samples in  "machine.test.libsvm"####

```
# stay in current path ("ytk-learn")
sh demo/gbdt/regression_l2/predict.sh
```

Get predict info(evaluation) in directory "ytk-learn/log"

```tail -f log/info.log```

Predicting is finished after you see "predict complete!" in log/info.log, results are saved in demo/data/libsvm/machine.test.libsvm_gbdt_LABEL_AND_PREDICT

**Notes**: If you want to get the leaf index of each sample, just set ``predict_type`` to "leafid".






