## Binary Classification

The following steps train a GBDT model by **data-parallel( histogram approximate) **algorithm and use this model to get prediction results. The script ``run.sh`` contains the whole steps.

**Notes**: 

If you want to train with feature-parallel algorithm, just change ``optimization.tree_maker`` from "data" to "feature". 

This is a small dataset, so we set ``feature.approximate`` to "{cols: "default", type: "no_sample"}" which means no approximation, for large data set you can set specified bin count by method such as "sample_by_quantile", or the training may be very slow.

#### 1. Prepare data####

Convert  training and testing data format from libsvm to ytklearn.
```
# make sure you are in path "ytk-learn"
# cd ../../.. (go to path "ytk-learn")
sh demo/gbdt/binary_classification/libsvm_convert_2_ytklearn.sh
```

You will find "agaricus.train.ytklearn" and "agaricus.test.ytklearn" in demo/gbdt/binary_classification. The ytklearn data format is as below:

```
1###1###3:1,10:1,11:1,21:1,30:1,34:1,36:1,40:1,41:1,53:1,58:1,65:1,69:1,77:1,86:1,88:1,92:1,95:1,102:1,105:1,117:1,124:1,24:0,25:0,26:0,27:0,20:0,22:0,23:0,28:0,29:0,4:0,8:0,120:0,121:0,122:0,123:0,125:0,126:0,55:0,54:0,56:0,51:0,50:0,52:0,115:0,114:0,116:0,111:0,110:0,113:0,112:0,82:0,83:0,80:0,81:0,119:0,87:0,84:0,85:0,7:0,108:0,109:0,100:0,101:0,106:0,107:0,39:0,32:0,31:0,37:0,60:0,61:0,62:0,63:0,64:0,66:0,67:0,68:0,2:0,6:0,99:0,98:0,91:0,90:0,93:0,94:0,96:0,13:0,12:0,15:0,14:0,17:0,16:0,19:0,18:0,48:0,49:0,46:0,47:0,44:0,45:0,42:0,43:0,118:0,1:0,5:0,9:0,76:0,75:0,74:0,73:0,72:0,71:0,70:0,79:0,78:0
1###0###3:1,10:1,20:1,21:1,23:1,34:1,36:1,39:1,41:1,53:1,56:1,65:1,69:1,77:1,86:1,88:1,92:1,95:1,102:1,106:1,116:1,120:1,24:0,25:0,26:0,27:0,22:0,28:0,29:0,4:0,8:0,121:0,122:0,123:0,124:0,125:0,126:0,58:0,55:0,54:0,51:0,50:0,52:0,115:0,114:0,117:0,111:0,110:0,113:0,112:0,82:0,83:0,80:0,81:0,119:0,87:0,84:0,85:0,7:0,108:0,109:0,100:0,101:0,107:0,105:0,32:0,31:0,30:0,37:0,60:0,61:0,62:0,63:0,64:0,66:0,67:0,68:0,2:0,6:0,99:0,98:0,91:0,90:0,93:0,94:0,96:0,11:0,13:0,12:0,15:0,14:0,17:0,16:0,19:0,18:0,48:0,49:0,46:0,47:0,44:0,45:0,42:0,43:0,40:0,118:0,1:0,5:0,9:0,76:0,75:0,74:0,73:0,72:0,71:0,70:0,79:0,78:0
```

The first '1' in ``1###1###3:1,..`` is the sample weight, the second '1' is the label, strings before colon such as  '3', '10', '11'  are feature names,  numbers behind colon are feature values.

#### 2. Train model####

```
# stay in current path ("ytk-learn")
sh demo/gbdt/binary_classification/local_optimizer.sh
```

Monitor train info in directory "ytk-learn/log"

```tail -f log/master.log```

You can see the loss of training and testing data in "log/master.log" during training as below:

```
INFO - ytk-learn model=gbdt [iter=2]  0.11500 sec elapse
train loss = 0.17249267375913763
test loss = 0.17219528385526198

INFO - ytk-learn model=gbdt [iter=3]  0.13600 sec elapse
train loss = 0.09960066518065772
test loss = 0.09943817574378232
```

You'll get evaluations at the end of training in "log/master.log":

```
INFO - [ytk-learn] training end, 0.25900 sec in all
train auc = 1.0
test auc = 1.0
```

If you set ``optimization.watch_train``  and `optimization.watch_test` to "true", you can mointor the evaluation of dataset in each iteration instead of only at the end of training process.

Training is finished after you see "exit code:0" in log/master.log

Mode and feature importance files are saved in "demo/gbdt/binary_classification" which is configured in  "local_gbdt.conf"

**Notes**: For each data don't for get to set ``data.max_feature_dim`` in local_gbdt.conf.

#### 3. Continue to train####

If you want to train model from existing model, set ``model.continue_train`` to true  and `optimization.round_num` to the final round number. e.g: If you set ``optimization.round_num`` from "3" to "5" then ytklearn will continue to train 2 trees.

#### 4. Get predictions  for samples in  "agaricus.txt.test" ####

```
# stay in current path ("ytk-learn")
sh demo/gbdt/binary_classification/predict.sh
```

Get prediction info(evaluation) in directory "ytk-learn/log"

```tail -f log/info.log```

Predicting is finished after you see "predict complete!" in log/info.log, results are saved in demo/gbdt/binary_classification/agaricus.test.ytklearn_gbdt_PREDICT_RESULT_ONLY

**Notes**: If you want to get the leaf index of each sample, just set ``predict_type`` to "leafid".






