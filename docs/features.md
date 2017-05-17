1. Supports most of operating systems: Linux, Mac OS, Windows.
2. Supports various platforms: single machine, common cluster, hadoop, spark.
3. Without complex installation, only needs Java SE Runtime Environment 8 installation.
4. Supports local file system, hdfs file system and uniform file system interface which can be applied to other file systems easily.
5. Provides user friendly codes for online prediction.
6. Multiple objectives and metrics.
7. All models support L1, L2 and L1 + L2 regularization.
8. Label-based instance sampling.
9. All tree models(GBDT, GBST) support instance sampling, feature sampling.
10. All tree models support training with initial prediction.
11. Supports continous training with previous checkpoint.
12. Weighted Instance training.
13. Two kinds of hyperparameter optimizition methods: grid search, hoag(automatic).
14. Supports unbiased feature hash.
15. Supports feature preprocessing(standardization, scaling).
16. Supports count-based feature filtering.
17. Provides python-based powerful data transformation [script](data_format.md), can transform data lines easily without changing its original data during training.
18. Laplace approximation in linear model(used for Thompson sampling in E&E application).
19. GBDT features: exact greedy algorithm and histogram approximate algorithm, tree growing by level-wise and leaf-wise policies, see more details in [gbdt features](gbdt_features.md).

