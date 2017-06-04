### **GBDT Configuration**

### *Helpful Links*

- [Running Guide](running_guide.md)

- [Demo](../demo/gbdt)

- [Features](gbdt_features.md)

- [Experiments](gbdt_experiments.md)

- [Training Data Format](data_format.md)

- [Evaluation Metrics](evaluation_metrics.md)

- [Performance Guide](performance_guide.md)


### Configuration

The configuration for gbdt mainly consists of four parts: data, model, feature and optimization. 

- Data: training and testing data path, [data format](data_format.md), data processing and feature number of training data.
- Model: model path, user-provided feature dict path, model dump frequency during training, whether to continue to train and the output feature importance path when training is finished. 
- Feature: feature split type when constructing a tree, feature bin generating methods, missing value processing type and lower boundary of feature appearance count for feature filtering. 
- Optimization: training parameters such as round_num, max_depth and max_leaf_cnt.

Following are the detailed descriptions of gbdt configuration:

```
# filesystem scheme URI
# local filesystem : "local", "file:///", (you can use both of them in linux and os system, but in windows local filesystem, only"local" can be used)
# hdfs filesystem : "hdfs://host"
# other filesystem URI
fs_scheme : "file:///",

# whether to print more detailed logs
verbose : false,

# type: "gradient_boosting" or "random_forest", default="gradient_boosting"
type:"gradient_boosting",

data {
    # train data
    train {
        # train data path
        # local filesystem : supports file and recursive directories
        # hdfs filesystem : depends on spark or hadoop cluster whereas spark supports more complicated paths(more than one path, regex paths)
        data_path : "???"
        # max tolerable error format count in training data
        max_error_tol : 0
    },

    # testing/validation data
    test {
        # test data path(empty string means no test data)
        # local filesystem : supports local file and recursive directories
        # hdfs filesystem : depends on spark or hadoop cluster whereas spark supports more complicated paths(more than one path, regex paths)
        data_path : "",
        # max tolerable error format count in train data
        max_error_tol : 0,
    },

    # delimiters, see data_format.md for more details
    # train/testing data format:
    #   regression : weight###label###f1name:f1value,f2name:f2value,...(###init_prediction)
    #   binary classification : weight###label(0 or 1)###f1name:f1value,f2name:f2value,...(###init_prediction)
    #   binary cross_entropy : weight###label(0~1, positive)###f1name:f1value,f2name:f2value,...(###init_prediction)
    #   multi classficication : weight###2(this belongs to 3'rd class, label must be in range [0,K-1], K is class number)###f1name:f1value,f2name:f2value,...
    #   multi cross_entropy : weight###0.2,0.1,0.4,0.3(total 4 class, sum must be equal 1.0)###f1name:f1value,f2name:f2value,...(###init_prediction)
    # (###init_prediction) is optional. If you provide initial prediction(s) for each sample, set optimization.sample_dependent_base_prediction to true
    delim {
        # separates sample weight, labels, features, init_prediction
        x_delim : "###",
        # if you have more than one label, they will be separated by y_delim
        y_delim : ",",
        # separates features(a feature includes feature_name and feature_value)
        features_delim : ",",
        # separates feature_name and feature_value
        feature_name_val_delim : ":"
    },

    # used in feature storage, maximum feature number can be larger than the real value(larger value will consume more memory, better set to the real value)
    max_feature_dim: 154,

    # if your task is of classification(including multi-class classification), 
    # you can downsample/upsample some special classes in special probability/weight
    # format : y_sampling : ["class1@prob", "class2@prob", ...]
    # downsampling: binary classification. If you want to reserve negative samples in 0.1 prob, set y_sampling to ["0:0.1"]
    # upsampling: multi-class classification. If you want to increase the 6'th class sample weight by 10X and 8'th class sample weight by 5X, set y_sampling to ["6@10","8@5"] 
    # empty means no sampling
    y_sampling : [],

    # whether your train/test data is assigned. See "Train/Test Data Splitting Manner" in running_guide.md for more details on train/test data assignment method
    assigned : false,
    # if your train/test data is not assigned, we provide the following two ways for slaves to read files: 
    # lines_avg : different slaves read different lines of same file alternative. If you have a few train/test files and more than one slave, we recommend this manner
    # files_avg : different slaves read different files. If your files outnumber slaves, and the number of samples in each file is similar, we recommend this manner
    unassigned_mode: "lines_avg"
},

model {
    # model output path can be used in future online prediction, with which you can view and understand the model
    data_path : "???",
  
    # whether to use user-provided feature dict, can be used in filtering features 
    need_dict : true,
  
    # user-provided dict path, if "need_dict:false", dict_path doesn't have to be provided.
    # feature dict data format:
    # f1
    # f2
    # ....
    # 
    # attention: dict_path is input, and model dict data is output
    dict_path : "",
  
    # model save frequency, < 0 means not saving model during training, only saving model when training finished
    dump_freq : -1,
  
    # whether to continue train loaded from checkpoint
    continue_train: false,
  
    # feature importance output path, each line contains three fields, feature_name, sum_split count and sum_gain
    # "sum_split_count" is the number of times a feature is split in all trees
    # "sum_gain" is the total gain of splits which use the feature
    feature_importance_path: "???",
},

optimization {
    # tree learning algorithm, "feature" or "data". See "Comparision Of Parallel Training Algorithms" (in the same page) for more details.
    # "feature": feature-parallel, using exact greedy algorithm, grows a tree by level-wise. It only supports local(single machine) version. Tree_grow_policy, histogram_pool_capacity and feature.approximate have no effect
    # "data": data-parallel, using histogram approximate algorithm, grows a tree by level-wise or loss-wise and supports both local and distributed version. You can config tree_grow_policy, histogram_pool_capacity and feature.approximate according to your requirements and limits at least one of these two parameters: max_depth and max_leaf_cnt. It means the situation, when max_depth and max_leaf_cnt are both set to -1, is not enabled.
    tree_maker: "data",

    # tree_grow_policy: level or loss, enabled when tree_maker is "data"
    # "level" grows tree by level(depth)-wise, and tree is generated by level
    # "loss" grows tree by leaf-wise, and leaf with maximum loss gain is firstly expanded
    tree_grow_policy: "level",
    # max memory(MB) allocated for histograms, <= 0 means no limit, enabled when tree_maker is "data"
    histogram_pool_capacity: -1,

    # number of boosting round, should >= 1
    round_num: 50,
  
    # maximum depth of a tree, < 0 means no limit, e.g. the depth of a tree with only a root is 0
    max_depth: 7,
  
    # minimum hessian sum in a leaf
    min_child_hessian_sum: 1,
    
    # maximum absolute value(before multiplies learning rate) in a leaf, <= 0 means no limit
    max_abs_leaf_val: -1,
  
    # maximum leaf count in a tree, < 0 means no limit
    max_leaf_cnt: 128,
  
    # minimum loss reduction to make a split, should >= 0
    min_split_loss: 0,
  
    # minimum number of samples to make a split, should >=2 or < 0, < 0 means no limit
    min_split_samples: 2000,

    # loss function: sigmoid, l1, l2, softmax, poisson
    # binary classification: sigmoid,
    # regression: "l1" stands for mean absolute error; "l2" stands for mean squared error
    # multiclass classification: softmax, you should set class_num
    # See "Supported Objective Functions" in gbdt_features.md for more details
    loss_function : "sigmoid",
    # used in binary classification(loss function is sigmoid), default=0, 0 means no constraints, but when training dataset is extremely imbalanced, setting it to a positive value can help. The recommended value range is [2,4]
    sigmoid_zmax: 0,

    regularization : {
    # shrinks the contribution of each tree by learning_rate and helps to prevent overfitting. The value is between [0,1]
    learning_rate: 0.09,
        # l1 regularization, larger value makes model more conservative
        l1 : 0.0,
        # l2 regularization, larger value makes model more conservative
        l2 : 1.0
    },

    # global uniform base prediction
    # in binary classification, recommended value: pos / (pos + neg)
    # in regression, recommended value: mean of labels
    uniform_base_prediction: 0.5,

    # whether to use sample dependent score
    # if it's true, init_predicion must be provided at the end of each sample line, see data_format.md for more details
    sample_dependent_base_prediction: false,
   
    # sample rate of training samples when generating a new tree
    instance_sample_rate: 1.0,

    # sample rate of features used in generating a new tree
    feature_sample_rate: 1.0,

    # class_num, used in multiclass classification, class_num is set to 1 for regression and binary classification by default
    class_num: 1,

    # only perform evaluation on train and test(optional) set (training finished)
    just_evaluate : false,

    # evaluation metrics, e.g.,  ["auc", "mae", "rmse", "confusion_matrix"], perform evaluation after each training iteration(if watch_train or watch_test is switched on) and at the end of training. Loss is calculated in default. See evaluation_metrics.md for more details
    eval_metric: [],

    # whether to perform evaluation of train set after each training iteration(we'll print loss after each iteration no matter what)
    watch_train: true,

    # whether perform evaluation of test set after each training iteration(we'll print loss after each iteration no matter what)
    watch_test: true,
},

# feature approximation and missing value config
feature {
    # feature split method: "mean"(default), "median"
    split_type: "mean",
    
    # feature bin generating methods for features, used in data-parallel algorithm(tree_maker: "data"), see "Feature Bin Generating Methods for Data parallel" (in the same page) for more details.
    # cols: feature names, split by ",". "default" means features except those existing in other cols. "default" should appear once at most
    # type: method of constructing feature bins, we recommend "sample_by_quantile"
    # "sample_by_quantile" samples out feature values according to percentiles of feature distribution
    #   "max_cnt": maximum value count in total(not in each thread)
    #   "quantile_approximate_bin_factor": default=8 and larger value get higher accuracy but with more memory cost. Generally, 8 is enough
    #   "alpha": default=1.0, range [0.0, 1.0]. Weight of specified feature value is math.pow(count of the specified feature value, alpha); alpha decreases the importance of the value number of each feature
    #   "use_sample_weight": default=false, whether to use sample weight while getting precentiles. If it's true, weight of specified feature value is math.pow(sum of sample weight of the specified feature value, alpha)
  
    # "sample_by_cnt", "sample_by_rate" and "sample_by_precision" can't control total feature bins exactly. They sample out feature values in each thread, accumulate values from all threads in all slaves, remove repeat values and construct feature bins
    # "sample_by_cnt" samples out specified count of feature values in each thread of each slave
    #    "max_cnt":  maximum sampled value count in each thread
  
    # "sample_by_rate" samples out feature values by "sample_rate" in each thread
    #   "sample_rate": sample ratio
    #   "min_cnt": min distinct value count thredshold for sampling. If distinct value count is equal or less than "min_cnt", then all values will be reserved without sampling
 
    # "sample_by_precision" firstly performs normalization and then reserves values of specified precision("dot_precision") in each thread of each slave
    #   "use_log": true or false, true means converting origin values, new_value = log(origin_value)
    #   "use_min_max": true or false, true means performing min-max normalization
    #   if "use_log" and "use_min_max" are both true, we firstly take logarithm and then perform min-max normalization
    #   "dot_precision": each value is rounded to this specified 'dot_precision' decimal places
    #   e.g., 0.453312223, dot_precision=4, then the reserved value is 0.4533
  
    # "no sample" uses all feature values to construct feature bins and there is no feature approximation. If the approximation configurations of all features are set to "no_sample",  then the training result is similar to exact greedy algorithm. It may be more accurate but the training process is slower. You'd better config important features or features with a few distinct values to "no sample".
    approximate: [
        # {cols: "1,2,5,6", type: "sample_by_cnt", max_cnt: 5},
        # {cols: "8,9,10", type: "sample_by_rate", sample_rate: 0.8, min_cnt: 0},
        # {cols: "30,49", type: "sample_by_precision", dot_precision: 5, use_log: true, use_min_max: true},
        # {cols: "4,9,55,cat,dog", type: "no_sample"},
        # {cols: "default", type: "sample_by_rate", sample_rate: 0.9, min_cnt: 10},
        # {cols: "1", type: "no_sample"},
        {cols: "default", type: "sample_by_quantile", max_cnt: 255, quantile_approximate_bin_factor: 8, use_sample_weight: false, alpha: 1.0},

       ]

    # we supply three methods to handle missing values: filling with "mean", "quantile" or specified "value"
    # "mean": filling missing value with mean
    # "quantile": filling missing value with median, "quantile@n" means using n'th quantile, n belongs to [0, 1],  e.g., quantile@0.75 means third quartile
    # "value": filling with 0, "value@m" means using specified value m. eg: "value@0.68"
    missing_value: "mean",
    # if a feature's frequency of occurrence is less than filter_threshold,  the feature will be filtered
    filter_threshold : 0
}
```



### Comparision Of Parallel Training Algorithms

Parallel training methods([details](gbdt_features.md)) is configured via "optimization.tree_maker" in gbdt.conf. Feature-parallel corresponds to setting "optimization.tree_maker" to "feature" and data-parallel corresponds to setting "optimization.tree_maker" to "data".

**Table 1. comparison of  feature-parallel and data-parallel training** 

|                         | feature-parallel                         | data-parallel                            |
| ----------------------- | :--------------------------------------- | ---------------------------------------- |
| tree learning algorithm | exact greedy algorithm                   | histogram approximate algorithm          |
| tree growing policy     | level(depth)-wise                        | level(depth)-wise and leaf(loss)-wise    |
| supporting filesystem   | local file system, hdfs file system, user defined file system. | local file system, hdfs file system and user defined file system. |
| running environment     | local machine                            | local machine and distributed cluster(common cluster, spark, hadoop) |



**Table 2. related params of feature-parallel and data-parallel in configuration**

|                                      | feature-parallel                       | data-parallel                       |
| ------------------------------------ | -------------------------------------- | ----------------------------------- |
| optimization.tree_maker              | "feature"                              | "data"                              |
| optimization.tree_grow_policy        | ignored, set to "level" by hard coding | enabled, support "level" and "loss" |
| optimization.histogram_pool_capacity | ignored, set to "-1"  by hard coding   | enabled                             |
| feature. approximate                 | ingnored, not used                     | enabled                             |



### Feature Bin Generating Methods for Data Parallel Training ###

For data-parallel (optimization.tree_maker="data")

We also provide several feature bin generating methods:

- **sample_by_quantile**: generates feature bins according to percentiles of feature distribution based on the algorithm in [1] and extends to weighted version(we recommend this method). The results of each training may vary because of the unstable algorithm of generating percentiles(feature bins).

- The three methods below can't control total feature bins exactly. Each of them samples out feature values in each thread, accumulates values from all threads in all slaves, removes repeat values and constructs feature bins(those feature values are boundaries of feature bins).

  - **sample_by_cnt**: samples out specified count of feature values in each thread of each slave.

    e.g., 5 machines, each with 10 threads, max_cnt=10, then the total sample out value count is $5\times10\times10=500$  (bin count may be smaller after removing repeating values)

  - **sample_by_rate**: samples out feature values by a specified rate in each thread of each slave.

    e.g., 5 machines, each with 10 threads. Each thread has 3000 samples, sample_rate=0.01, then the total sample out value count is approximate to  $5\times10\times3000\times0.1=1500$ (bin count may be smaller after removing repeat values)

  - **sample_by_precision**: samples out feature values by a specified precision in each thread of each slave. Sampling by precision means that, firstly performing normalization(use_log, use_min_max, you can switch on or off) on feature values, and then reserving values of specified precision.

    e.g., after performing normalizaion the feature value becomes 0.453312223. If  you set "dot_precision=4", then the reserved value is 0.4533.

- **no_sample**: uses all feature values to construct feature bins. It means each feature bin contains only one feature value in training set. So there is no feature approximation. If the approximation configurations of all features are set to "no_sample",  then the training result is similar to exact greedy algorithm. It may be more accurate but the training process is slower. You'd better config important features or features with a few distinct values to "no sample".


### Reference

1.Zhang, Qi, and Wei Wang. "A fast algorithm for approximate quantiles in high speed data streams." Scientific and Statistical Database Management, 2007. SSBDM'07. 19th International Conference on. IEEE, 2007.
