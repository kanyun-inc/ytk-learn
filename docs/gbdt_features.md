## GBDT Features

### Tree Learning Algorithm

We support two algorithms for tree learning: exact greedy and histogram approximate algorithm.

- **Exact greedy algorithm**: pre-sorts feature values. For each split in the tree, it traverses the related feature values to find a split that most decreases the objective(loss), see [1]. In most cases, this algorithm is very effective because it enumerates all related splitting feature values, but not efficient in distributed training when data is too much that can't be entirely stored in memory. We implement this algorithm in feature-parallel training on a single machine.

- **Histogram approximate algorithm**: buckets continuous feature values into discrete bins. So the candidate splits are related feature bins which are usually much smaller than feature values in the exact greedy algorithm. It helps to **speed up distributed training**. We also implement the **histogram subtraction** trick like [LightGBM](https://github.com/Microsoft/LightGBM) for further speed-up. We implement this algorithm in data-parallel training both on a single machine and a cluster.

  We also provide several feature bin generating methods(see more details in [configuration](gbdt.config.md).):

  - sample_by_quantile: generates feature bins according to percentiles of feature distribution based on the algorithm in [2] and extends to weighted version(recommended).

  - Each of the three methods below samples out feature values in each thread, accumulates values from all threads in all slaves by **allreduce**, removes repeat values and constructs feature bins.
    - sample_by_cnt: samples out specified count of feature values in each thread of each slave.
    - sample_by_rate: samples out feature values by a specified rate in each thread of each slave.
    - sample_by_precision: samples out feature values by a specified precision in each thread of each slave.

  - no_sample: uses all feature values to construct feature bins. It means each feature bin contains only one feature value in training set. So there is no feature approximation. If the approximation configurations of all features are set to "no_sample",  then the training result is similar to exact greedy algorithm.


### Tree Growing Policy

We provide two policies for tree growing: level(depth)-wise and leaf(loss)-wise.

- **level(depth)-wise**: grows a tree by level.
- **leaf(loss)-wise**: grows a tree by leaf and chooses the leaf with max gain to grow.

In most cases, with the same number of leaves, level-wise policy is good at generalization, and leaf-wise is good at fitting.



### Memory and Network Communication

The communication in multi-thread and multi-process(distributed environments) training is based on [ytk-mp4j]() which supports them both. Ytk-mp4j implements state-of-art algorithms in [3,4] and modifies them. The time complexity of **allreduce** and **reduce-scatter** in ytk-mp4j is just half compared to  [LightGBM](https://github.com/Microsoft/LightGBM) if the number of slaves isn't a power of two. We use operations such as "allreduce", "reduce-scatter" and "allgather" for parallel training.



### Parallel Training

- **Feature parallel training**

  We implement feature-parallel training only for a single machine because the time complexity of this algorithm is O(#data)  which is inefficient when data is very large. In feature parallel training, we only support exact greedy tree learning algorithm and level-wise tree growing policy.

  The procedure of our feature parallel training is:

  1. Partition data **vertically and horizontally** into each thread.
  2. Each thread finds local best split on local feature set.
  3. Each thread communicates local best spit with others by **"allreduce"** and then, all threads calculate the best split.
  4. Each thread splits local data according to the best split.
  5. Each thread communicates local sample position with others by **"allgather"** and all threads gets the global sample position.

- **Data parallel training**

  We implement data-parallel training both on a single machine and a cluster. We support "histogram approximate algorithm" and both two tree growing policies above. All of the feature bin generating methods are applicable for distributed environments that each machine can see only a part of the data.

  The procedure of our data parallel training is:

  1. Partition data horizontally into each thread of each machine.

  2. Generate feature bins.

     (1) Each worker generates candidates for feature bin boundaries.

     (2) Merge local candidates to generate global feature bin boundaries by **"allreduce"**.

  3. Workers use local data to construct local histograms based on global feature bin boundaries.

  4. Merge local histograms and get global feature histograms by **"reduce-scatter"**.

  5. Each worker finds the best split from global histograms and then performs split on local data.

  The following table compares feature-parallel and data-parallel training.

  |                         | feature-parallel                         | data-parallel                            |
  | ----------------------- | ---------------------------------------- | ---------------------------------------- |
  | tree learning algorithm | exact greedy algorithm                   | histogram approximate algorithm          |
  | tree growing policy     | level(depth)-wise                        | level(depth)-wise and leaf(loss)-wise    |
  | supporting filesystem   | local file system, hdfs file system, user defined file system. | local file system, hdfs file system and user defined file system. |
  | running environment     | local machine                            | local machine and distributed cluster(common cluster, spark, hadoop) |


### Supported Objective Functions

- Classification
  - Binary classification
    - sigmoid(negative log bernoulli likelihood)
    - sigmoid_cross_entropy
  - Multi-class classification
    - softmax(negtive multinoulli likelihood)
    - softmax_cross_entropy
- Regression
  - l2(mean squared error, least squares regression)
  - l1(mean absolute error, least absolute deviation regression)
  - poisson

The objective functions listed above can be set via ```optimization.loss_function``` in model configuration.



### Supported Evaluation Metrics

- MAE
- RMSE
- Confusion Matrix (contains precision, recall and accuracy)
- AUC

Loss is calculated in default, see [evaluation metrics](evaluation_metrics.md) for details.



### Other Features

- Multiple objectives and multiple metrics

- L1、L2 and L1 + L2 regularization

- Filling missing value with mean, quantile or specified value

- Sampling, label-based sampling, feature sampling

- Weighted sample training

- Training with initial prediction

- Continuing to train with previous checkpoint

- Count-based feature filtering

- Python-based powerful data transformation script, transforming data easily without changing original data during training

- Supporting random forest

- [Common features](features.md)


### Reference

1. Chen T, Guestrin C. Xgboost: A scalable tree boosting system[C]//Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016: 785-794.
2. Zhang, Qi, and Wei Wang. "A fast algorithm for approximate quantiles in high speed data streams." Scientific and Statistical Database Management, 2007. SSBDM'07. 19th International Conference on. IEEE, 2007.
3. Thakur, Rajeev, Rolf Rabenseifner, and William Gropp. "Optimization of collective communication operations in MPICH." *The International Journal of High Performance Computing Applications* 19.1 (2005): 49-66.
4. Faraj, Ahmad, Pitch Patarasuk, and Xin Yuan. "Bandwidth efficient all-to-all broadcast on switched clusters." *Cluster Computing, 2005. IEEE International*. IEEE, 2005.
