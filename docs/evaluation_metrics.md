### **Evaluation Metrics**###

Ytk-learn can perform evaluation both in training(training set & testing test) and offline batch prediction. It provides the following evaluation metrics:

* mae: [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error)
* rmse:  [Root Mean Square Error](https://en.wikipedia.org/wiki/Root_mean_square_error)
* confusion_matrix: [Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix). It's used in binary classification and multi-class classification. ``precision``, ``recall`` for each class and ``accuracy`` are calculated along with confusion matrix. For binary classification, the default classification thredshold is 0.5, and you can provide specified threshold “t” by "confusion_matrix@t". For multi-class classification, providing specified threshold is useless.
* auc: [Area Under Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_curve) for binary classification. It's calculated by multi-thread(or distributed on multiple machines). The precision is 1E-5 by default(auc@100000), which means prediction is rounded to 5 decimal places for AUC calculation. You can specify the precision to “1E-m” by “auc@m".

**Notes**:  If samples are not equal-weighted, ytk-learn calculates both weighted and unweightd evaluation metrics. Otherwise, it only calculates the unweighted. You can see the results in log/master.log for training phase and log/info.log for offline bach prediction. Ytk-learn calculates **loss** during training and prediction as long as labels exist. 

### Evaluation in Training Phase ###

You can perform evaluation in train phase by setting ``eval_metric`` in the configuration file, the config value type is a list, where you can put multiple metrics. The default value for ``just_evaluation`` is false which means it's in training phase. If you just want to perform the evaluation without saving predicion results for each sample, set ``just_evaluation`` to true. It will calculate evalutions by parallel. The script is bin/**_optimaizer.sh.

### **Offline Batch Prediction**

The offline batch prediction will save prediction results for each sample in file meanwhile performing evaluation. Evaluation is optional.

#### **Testing data format** ####

Testing data format is almost the same as [training data](data_format.md) except that you can also provide data without labels for offline batch prediction. e.g:

```
test data with label: 1###0###1:1.2,9:3.2,19:33,21:1, '0' is the label

test data without label: 1######1:1.2,9:3.2,19:33,21:1
```

with data format configured as:

```
delim {
    x_delim : "###",
    y_delim : ",",
    features_delim : ",",
    feature_name_val_delim : ":"
}
```

You can also provide each sample with an initial prediction(regression and binary classification) or initial scores(multi-class classification, scores are the origin predict score before softmax). An example is as below:

```
test data with label: 1###0###1:1.2,9:3.2,19:33,21:1###0.01

test data without label: 1######1:1.2,9:3.2,19:33,21:1###0.01
```

#### Prediction mode ####

If you want to perform evaluation and save predictions offline, you can use the script ```bin/predict.sh```. There are three modes: ``PREDICT_RESULT_ONLY``, ``LABEL_AND_PREDICT`` and ``PREDICT_AS_FEATURE``. Details are listed in the following table. 

| mode                | output                                   | example                                  |
| ------------------- | ---------------------------------------- | ---------------------------------------- |
| PREDICT_RESULT_ONLY | label only                               | 4.397<br>4.397 is prediction             |
| LABEL_AND_PREDICT   | label and prediction                     | 3.585###4.397<br>3.585 is label and 4.397 is prediction |
| PREDICT_AS_FEATURE  | origin sample info with prediction as a feature(features) | predict_type=value:<br>1###3.585###1:2.60,2:5.2,3:2.22,4:4.37,gbdt_label_0:4.397<br>1###3.585###1:2.60,2:5.2,3:2.22,4:4.37 is origin sample info<br>predict_type=leafid: 1###3.585###1:2.60,2:5.2,3:2.22,4:4.37,tree_leaf_0:0.0,tree_leaf_1:6.0,tree_leaf_2:34.0,tree_leaf_3:48.0,tree_leaf_4:88.0<br> it means there are 5 trees, the leaf indexes of this sample are 0,6,34,48 and 88 |

#### Prediction type ####

``predict_type`` can be set to ``value`` for all models.  For tree-based models(GBDT, GBMLR, GBSDT, GBHMLR, GBHSDT), you can set ``predict_type`` to ``leafid`` to get leaf indexes for each sample. Leaf index is the inner tree node number. The prediction results are in the same order as input dataset.

#### Evaluation ####

If you provide labels,  you can set ``eval_metric`` to perform evaluations. Ytk-learn supports multiple evaluation metrics, splitted by comma(empty `eval_metric` means no evaluation). Loss is calculated by default as long as labels exist.

 You can see the details in ``log/info.log``, including evaluations and a file path of prediction results.

The following script is [predict.sh](../bin/predict.sh) ([win_predict.sh](../bin/win_predict.sh) for windows).

```shell
#!/usr/bin/env bash

echo "########################################################################"
echo "######          offline predicting "
echo "######  usage: cd ytk-learn & sh bin/predict.sh"
echo "######  attention : '???' means the value must be filled by user himself"
echo "#########################################################################"

# model name(linear, fm, ffm, gbdt, gbmlr, gbsdt, gbhmlr, gbhsdt)
model_name=???

config_path="config/model/${model_name}.conf"

# data file for predicting
file_dir=???

# train/test line python transform switch & script
transform="false"
transform_script_path="bin/transform.py"

# result save mode: PREDICT_RESULT_ONLY, LABEL_AND_PREDICT, PREDICT_AS_FEATURE
resultSaveMode="PREDICT_RESULT_ONLY"
resultFileSuffix="_"${model_name}"_"${resultSaveMode}

# max error data format tolerate number
max_error_tol=100
eval_metric="auc,mae"
# "value", "leafid"
predict_type="value"

nohup java -server -Xmx1000m -XX:-OmitStackTraceInFastThrow -classpath .:lib/*:config -Dlog4j.configuration=file:config/log4j.properties com.fenbi.ytklearn.predictor.Predicts  \
    "${config_path}" "${model_name}" "${file_dir}" "${transform}" "${transform_script_path}" "${resultSaveMode}" "${resultFileSuffix}" "${max_error_tol}" "${eval_metric}" "${predict_type}">> log/info.log 2>&1 &
```

**Notes**:  ytk-learn loads model from path configured in ``config_path``, data and model path should be in the **same file system** because they share the same ``fs_scheme``  which is set in config_path, e.g. both in local filesystem or in hdfs.

