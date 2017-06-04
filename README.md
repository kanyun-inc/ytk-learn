**Ytk-learn** is a distributed machine learning library which implements most of popular machine learning algorithms. It runs on single, multiple machines and major distributed environments(hadoop, spark)，and supports major operating systems(Linux, Windows, Mac OS)，the communication of distributed environments is implemented based on [ytk-mp4j](https://github.com/yuantiku/ytk-mp4j) which is pure java, mpi-like message passing interface.

### Features

- Supports most of operating systems: Linux, Mac OS, Windows
- Supports various platforms: single machine, common cluster, hadoop, spark 
- Supports local file system and hdfs file system
- Provides uniform file system interface and can be applied to other file systems easily.
- Provides user friendly codes for online prediction.
- Without complex installation, only needs Java SE Runtime Environment 8 installation.

For more details, refer to [features](docs/features.md)

### Documents

- [Running Guide](docs/running_guide.md)
- [Demo](demo)
- [Model Introduction](docs/models.md)
- [Data Format](docs/data_format.md)
- [Evaluation Metrics](docs/evaluation_metrics.md)
- [Performance Guide](docs/performance_guide.md)
- [Online Prediction Guide](docs/online.md)

### Experiments

We compare our GBDT with [XGBoost](https://github.com/dmlc/xgboost) and [LightGBM](https://github.com/Microsoft/LightGBM), see [gbdt experiments](docs/gbdt_experiments.md) for more details.

### Environment Requirements

To run or develop ytk-learn，just install [JRE 8](http://www.oracle.com/technetwork/java/javase/downloads/jre8-downloads-2133155.html) or [JDK 8](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html) and set [JAVA_HOME](https://docs.oracle.com/cd/E19182-01/820-7851/inst_cli_jdk_javahome_t/).
