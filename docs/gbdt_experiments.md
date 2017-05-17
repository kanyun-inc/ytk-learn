## Comparision Experiment

We compare our GBDT with [XGBoost](https://github.com/dmlc/xgboost) (commit [e38bea3](https://github.com/dmlc/xgboost/commit/e38bea3cdfdde3f1b33343f72db59da06043c73a)) and [LightGBM](https://github.com/Microsoft/LightGBM) (commit [57ad014](https://github.com/Microsoft/LightGBM/commit/57ad0149f9a4342298d95bf0b203115ddf84d7e1)) on 04/20/2017. 

### Experiment on single machine

#### Data

Experiments are conducted on [Higgs](https://archive.ics.uci.edu/ml/datasets/HIGGS). The task is of binary classfication. The size of train set is 10,500,000 and the number of features is 28, and we use the last 500,000 samples as test set.

#### Environment

We perform the experiment on a Linux Server. Following is the detailed info:

**OS**: CentOS Linux release 7.1.1503

**CPU**: 2 * Intel(R) Xeon(R) CPU E5-2640 v3 @ 2.60GHz

**Memory**: 128G

#### Configuration

Following are main configurations of XGBoost, LightGBM and our GBDT. To make the three experiments fair, all models are trained by histogram approximate algorithm while average of logloss of train and test data set are monitored during training process(loss is calculated in default during each training iteration in ytk-learn).

1. **XGBoost**

```
tree_method=hist
max_bin=255
grow_policy=lossguide
objective="binary:logistic"
nthread=16
max_leaves=255
num_round=500
max_depth=0
eta=0.1
min_child_weight=100
gamma=0
lambda=0
alpha=0
eval_metric="logloss"
eval[train]="higgs.train"
eval[test]="higgs.test"
```

2. **LightGBM**

```
objective=binary
num_threads=16
max_bin=255
num_leaves=255
num_trees=500
learning_rate=0.1
min_data_in_leaf=0
min_sum_hessian_in_leaf=100
metric=binary_logloss
is_training_metric=true
valid="higgs.test"
```

3. **ytk-learn**

```
tree_maker: "data",
tree_grow_policy: "loss",
max_leaf_cnt: 255,
round_num: 500,
max_depth:-1,
loss_function : "sigmoid",
regularization.learning_rate: 0.1,
regularization.l1 : 0.0,
regularization.l2: 0.0,
min_split_samples:-1,
min_child_hessian_sum:100,
feature.approximate: [
 {cols: "default", type: "sample_by_quantile", max_cnt: 255, use_sample_weight: false, alpha: 0.5}]
```

#### Code

We run the experiment of XGBoost and LightGBM based on [boosting_tree_benchmarks](https://github.com/guolinke/boosting_tree_benchmarks) and we modified the configurations to suit our needs. And the experiment of our GBDT is based on [higgs](../experiment/higgs).

#### Result

1. **Performance**

In our GBDT, the results of each training may be different because of the unstable algorithm of generating percentiles, so we repeate the experiment three times. The following table shows average logloss and AUC. 

|         | XGBoost                      | LightGBM                     | GBDT in ytk-learn                        |
| ------- | ---------------------------- | ---------------------------- | ---------------------------------------- |
| logloss | train:0.473704,test:0.482996 | train:0.473409,test:0.482948 | (1)train: 0.472630, test: 0.482095<br>(2)train:0.473604, test:0.483073<br>(3)train:0.473141, test:0.482539 |
| auc     | 0.845605                     | 0.845612                     | (1) 0.846235<br>(2) 0.845539<br>(3) 0.845923 |

2. **Speed**

The following table shows running time of experiments.

|                          | XGBoost | LightGBM | ytk-learn |
| ------------------------ | ------- | -------- | --------- |
| load data and preprocess | 11.96s  | 14.24s   | 35.46s    |
| train                    | 610.38s | 269.19s  | 567.83s   |
| total                    | 622.34s | 283.43s  | 603.30s   |

Loss is calculated in default during each training iteration in ytk-learn, so we also monitor train and test loss in xgboost and LightGBM. Without monitoring loss, the results are listed in the following table:

|                          | XGBoost | LightGBM |
| ------------------------ | ------- | -------- |
| load data and preprocess | 5.45s   | 13.46s   |
| train                    | 372.01s | 239.96s  |
| total                    | 377.46s | 253.42s  |


