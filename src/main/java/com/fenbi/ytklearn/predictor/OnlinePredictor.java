/**
*
* Copyright (c) 2017 ytk-learn https://github.com/yuantiku
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

package com.fenbi.ytklearn.predictor;


import com.fenbi.ytklearn.exception.YtkLearnException;
import com.fenbi.ytklearn.fs.FileSystemFactory;
import com.fenbi.ytklearn.fs.IFileSystem;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import lombok.Data;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.Reader;
import java.net.URI;
import java.util.Map;

/**
 * @author xialong
 */

@Data
public abstract class OnlinePredictor {
    public static final Logger LOG = LoggerFactory.getLogger(OnlinePredictor.class);
    protected Config config;
    protected IFileSystem fs;

    public static enum ResultSaveMode {
        PREDICT_RESULT_ONLY("predict_result_only"),
        LABEL_AND_PREDICT("label_and_predict"),
        PREDICT_AS_FEATURE("label_as_feature");

        private String name;
        private ResultSaveMode(String name) {
            this.name = name;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    public static enum PredictType {
        PREDICT_LEAFID("leafid"),
        PREDICT_VALUE("value");

        String type;
        PredictType(String type) {
            this.type = type;
        }

        String getName() {
            return type;
        }

        static PredictType valueOfType(String type) {
            if (type == null || type.length() == 0) {
                return PREDICT_VALUE;
            }
            for (PredictType mode : values()) {
                if (mode.getName().equalsIgnoreCase(type)) {
                    return mode;
                }
            }
            throw new YtkLearnException("predict type invalid! value or leafid");
        }
    };

    public OnlinePredictor(String configPath) throws Exception {
        config = ConfigFactory.parseFile(new File(configPath));

        LOG.info("load config from file, config_path:" + configPath + ", existed:" + new File(configPath).exists());
        String uri = config.getString("fs_scheme");
        fs = FileSystemFactory.createFileSystem(new URI(uri));

    }

    public OnlinePredictor(Reader configReader) throws Exception {
        config = ConfigFactory.parseReader(configReader);

        LOG.info("load config from reader!");
        String uri = config.getString("fs_scheme");
        fs = FileSystemFactory.createFileSystem(new URI(uri));
    }

    protected abstract OnlinePredictor loadModel() throws Exception;

    /**
     * Calculate score, used for single label model
     * @param features features map, key:featureName, value:featureValue
     * @param other if model is tree model, and uses sample dependent score,
     *              other is sample dependent score(Float type),
     *              if not, other parameter will be omit(set null).
     * @return score score value before entering active function
     */
    public abstract double score(Map<String, Float> features, Object other);

    /**
     * Calculate scores, used for multi labels model, e.g. multiclass_linear model
     * @param features features map, key:featureName, value:featureValue
     * @param other if model is tree model, and uses sample dependent score,
     *              other is sample dependent scores(Float[] type),
     *              if not, other parameter will be omit(set null)
     * @return
     */
    public double[] scores(Map<String, Float> features, Object other) {
        return new double[] {score(features, other)};
    }

    /**
     * Predict using active function, used for single label model
     * @param features features map, key:featureName, value:featureValue
     * @param other see {@link #score(Map, Object)}
     * @return  prediction. e.g.:
     *          if your model is linear model, and loss_function is "sigmoid"(Logistic Regression), prediction is probability.
     *          if your model is linear model, and loss_function is "L2"(Linear Regression, Identity active function), prediction is equal to score
     */
    public abstract double predict(Map<String, Float> features, Object other);

    /**
     * Predict using active function, used for multi labels model
     * @param features features map, key:featureName, value:featureValue
     * @param other see {@link #scores(Map, Object)}
     * @return predictions
     */
    public double[] predicts(Map<String, Float> features, Object other) {
        return new double[]{predict(features, other)};
    }

    /**
     * Calculate loss, Used for single label model
     * @param features features map, key:featureName, value:featureValue
     * @param label label
     * @param other see {@link #score(Map, Object)}
     * @return loss
     */
    public abstract double loss(Map<String, Float> features, double label, Object other);

    /**
     * Calculate loss, Used for multi labels model
     * @param features features map, key:featureName, value:featureValue
     * @param labels labels
     * @param other see {@link #scores(Map, Object)}
     * @return
     */
    public double loss(Map<String, Float> features, double[] labels, Object other) {
        return Double.MAX_VALUE;
    }

    public abstract double batchPredictFromFiles(String modelName,
                                        String fileDir,
                                        boolean needPyTransform,
                                        String pyTransformScript,
                                        String resultSaveMode,
                                        String resultFileSuffix,
                                        int maxErrorTol,
                                        String evalMetricStr,
                                        String predictTypeStr) throws Exception;

//    public static void main(String []args) throws Exception {
//        String path = "hdfs://f04/user/xialong/test/linear.conf";
//        FileSystem fs = FileSystem.get(URI.create("hdfs://f04"), new Configuration());
//        LinearOnlinePredictor predictor = (LinearOnlinePredictor) OnlinePredictorFactory.createOnlinePredictor("linear", new InputStreamReader(fs.open(new Path(path))));
//    }

}
