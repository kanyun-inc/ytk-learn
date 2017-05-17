## Multi-class Classification

This is a six classification task. The following steps train a GBDT model by **data-parallel( histogram approximate) **algorithm and use this model to get prediction results. The script ``run.sh`` contains the whole steps.
**Notes**: This is a small dataset, so we set ``feature.approximate`` to "{cols: "default", type: "no_sample"}" which means no approximation, for large data set you can set specified bin count by method such as "sample_by_quantile", or the training process may be very slow.

#### 1. Prepare data####

Convert  training and testing data format from libsvm to ytklearn.

```
# make sure you are in path "ytk-learn"
# cd ../../.. (go to path "ytk-learn")
sh demo/gbdt/multiclass_classification/libsvm_convert_2_ytklearn.sh
```

You will find "dermatology.train.ytklearn" and "dermatology.test.ytklearn" in demo/gbdt/multiclass_classification.

#### 2. Train model####

```
# stay in current path ("ytk-learn")
sh demo/gbdt/multiclass_classification/local_optimizer.sh
```

local_gbdt.conf is the configuration. ``optimization.loss_function ``is set to "softmax" and ``optimization.class_num`` is set to "6" which is the total number of target classes.

Monitor train info in directory "ytk-learn/log"

```tail -f log/master.log```

Training is finished after you see "exit code:0" in log/master.log.

Mode and feature importance files are saved in "demo/gbdt/multiclass_classification" which is configured in  "local_gbdt.conf"

**Notes**: For each data don't for get to set ``data.max_feature_dim`` in local_gbdt.conf.

####3. Get predictions  for samples in  "dermatology.test.ytklearn"####

```
# stay in current path ("ytk-learn")
sh demo/gbdt/multiclass_classification/predict.sh
```

Get predict info(evaluation) in directory "ytk-learn/log"

```tail -f log/info.log```

Predicting is finished after you see "predict complete!" in log/info.log, results are saved in demo/gbdt/multiclass_classification/dermatology.test.ytklearn_gbdt_LABEL_AND_PREDICT.

Following is a line in the prediction result file:

```
0,0,0,1,0,0###0.12211822085981826,0.12527414781210444,0.12204564173421276,0.38662393640671316,0.12199698511704811,0.12194106807010317
```

It means the label is the fourth class of the six classes, and the model prediction for each class is 0.12211822085981826, 0.12527414781210444, 0.12204564173421276, 0.38662393640671316, 0.12199698511704811 and 0.12194106807010317.

**Notes**: If you want to get the leaf index of each sample, just set ``predict_type`` to "leafid".






