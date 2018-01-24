### Online Prediction

Ytk-learn provides online prediction java class for each model. All models provide score-calculating and predicting interfaces, which are thread-safe. 

| model             | class                                    | interface                                |
| :---------------- | :--------------------------------------- | :--------------------------------------- |
| linear            | com.fenbi.ytklearn.predictor.LinearOnlinePredictor | loss/score/predict/thompsonSamplingPredict |
| multiclass_linear | com.fenbi.ytklearn.predictor.MulticlassLinearOnlinePredictor | loss/scores/predicts                     |
| fm                | com.fenbi.ytklearn.predictor.FMOnlinePredictor | loss/score/predict                       |
| ffm               | com.fenbi.ytklearn.predictor.FFMOnlinePredictor | loss/score/predict                       |
| gbdt              | com.fenbi.ytklearn.predictor.GBDTOnlinePredictor | loss/score/predict/scores/predicts/predictLeaf |
| gbmlr             | com.fenbi.ytklearn.predictor.GBMLROnlinePredictor | loss/score/predict/predictLeaf           |
| gbsdt             | com.fenbi.ytklearn.predictor.GBSDTOnlinePredictor | loss/score/predict/predictLeaf           |
| gbhmlr            | com.fenbi.ytklearn.predictor.GBHMLROnlinePredictor | loss/score/predict/predictLeaf           |
| gbhsdt            | com.fenbi.ytklearn.predictor.GBHSDTOnlinePredictor | loss/score/predict/predictLeaf           |

#### Interfaces

```java
protected abstract OnlinePredictor loadModel() throws Exception;

    /**
     * Calculates score, used for single label model
     * @param features features map, key:featureName, value:featureValue
     * @param other if the model is a tree-based model, it uses sample dependent score(Float type)
     *              if not, this parameter will be omitted(set null).
     * @return score score value before entering active function
     */
    public abstract double score(Map<String, Float> features, Object other);

    /**
     * Calculates scores, used for multi-label model, e.g. multiclass_linear model
     * @param features features map, key:featureName, value:featureValue
     * @param other if the model is a tree=based model, it uses sample dependent score(Float[] 		 *				type)
     *              if not, this parameter will be omitted(set null)
     * @return
     */
    public double[] scores(Map<String, Float> features, Object other);

    /**
     * Predicts using active function, used for single-label model
     * @param features features map, key:featureName, value:featureValue
     * @param other see {@link #score(Map, Object)}
     * @return  prediction. e.g.:
     *          if your model is a linear model, and loss_function is "sigmoid"(Logistic Regression), then prediction is probability.
     *          if your model is a linear model, and loss_function is "L2"(Linear Regression, Identity active function), then prediction is equal to score
     */
    public abstract double predict(Map<String, Float> features, Object other);

    /**
     * Predicts using active function, used for multi-label model
     * @param features features map, key:featureName, value:featureValue
     * @param other see {@link #scores(Map, Object)}
     * @return predictions
     */
    public double[] predicts(Map<String, Float> features, Object other);
    /**
     * Calculates loss, used for single-label model
     * @param features features map, key:featureName, value:featureValue
     * @param label label
     * @param other see {@link #score(Map, Object)}
     * @return loss
     */
    public abstract double loss(Map<String, Float> features, double label, Object other);

    /**
     * Calculates loss, used for multi-label model
     * @param features features map, key:featureName, value:featureValue
     * @param labels labels
     * @param other see {@link #scores(Map, Object)}
     * @return
     */
    public double loss(Map<String, Float> features, double[] labels, Object other);

    /**
     * Thompson sampling prediction for E&E(Exploitation and Exploration).
     * Using Laplace approximation, distribution of parameters posterior will be approximate to diagonal gaussian distribution,
     * Details see "An Empirical Evaluation of Thompson Sampling"
     * @param features features map, key:featureName, value:featureValue
     * @param alpha multiplied by standard deviation, it controls the exploitation and the exploration; the larger alpha value, the more exploration, the less exploitation.
     * @return prediction
     */
    public double thompsonSamplingPredict(Map<String, Float> features, double alpha);

    /**
     * If the model is GBDT, leaf indexes are returned, length = tree number
     * else if the model is GBST, gating values(soft leaf indexes) are returned, length = tree number * mixture number
     * @param features
     * @return leaf indexes or gating values, these values are usually used as features for other models.     
     */
    double[] predictLeaf(Map<String, Float> features);
```

How to create online predictor?

```OnlinePredictorFactory``` static class provides two static functions to create online predictor, the content and format of configuration file is the same as training.

```java
public static OnlinePredictor createOnlinePredictor(String modelName, String configPath) throws Exception;
public static OnlinePredictor createOnlinePredictor(String modelName, Reader configReader) throws Exception;
```

### Integration with Maven

```xml
<dependency>
  <groupId>com.fenbi</groupId>
  <artifactId>ytk-learn</artifactId>
  <version>0.0.3</version>
</dependency>
```

