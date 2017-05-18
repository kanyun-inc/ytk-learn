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

import com.fenbi.ytklearn.data.Tuple;
import lombok.Data;

import java.io.Reader;
import java.util.*;

/**
 * @author xialong
 */

@Data
public class LinearOnlinePredictor extends ContinuousOnlinePredictor<Tuple<Float, Float>> {
    public static final float PRECISION_MIN = 1e-9f;
    private final Random rand = new Random();

    public LinearOnlinePredictor(String configPath) throws Exception {
        super(configPath);
        loadModel();
    }

    public LinearOnlinePredictor(Reader configReader) throws Exception {
        super(configReader);
        loadModel();
    }


    @Override
    protected OnlinePredictor loadModel() throws Exception {
        if (!fs.exists(modelParams.data_path)) {
            throw new Exception ("linear model doesn't exist! path:" + modelParams.data_path);
        }

        int cnt = 0;
        List<Iterator<String>> iterators = fs.read(Arrays.asList(modelParams.data_path));
        for (Iterator<String> it : iterators) {
            while (it.hasNext()) {
                String line = it.next();
                if (line.trim().length() == 0) {
                    LOG.error("invalid model line:" + line);
                    continue;
                }
                String []info = line.trim().split(modelParams.delim);

                if (info.length < 2) {
                    LOG.error("[invalid model line:" + line);
                    continue;
                }


                if (!line.startsWith(modelParams.bias_feature_name)) {
                    String fname = info[0].trim();
                    float wei = Float.parseFloat(info[1].trim());
                    float precision = Math.max(Float.parseFloat(info[2].trim()), PRECISION_MIN);
                    float std = (float)Math.sqrt(1.0 / precision);
                    modelMap.put(fname, new Tuple<>(wei, std));

                    if (cnt++ < 10) {
                        LOG.info("fname:" + fname + ", wei:" + wei + ", precision:" + precision +
                                ", std:" + std);
                    }
                } else {
                    String fname = info[0].trim();
                    float wei = Float.parseFloat(info[1].trim());
                    float precision = 1e30f;
                    float std = (float)Math.sqrt(1.0 / precision);
                    modelMap.put(fname, new Tuple<>(wei, std));

                    if (cnt++ < 10) {
                        LOG.info("fname:" + fname + ", wei:" + wei + ", precision:" + precision +
                                ", std:" + std);
                    }
                }
            }
        }

        LOG.info("load model finished! feature cnt:" + cnt);
        return this;
    }

    @Override
    public double score(Map<String, Float> features, Object other) {
        features.remove(modelParams.bias_feature_name);

        if (commonParams.featureParams.feature_hash.need_feature_hash) {
            features = featureHash.hashMap2Map(features);
        }

        double score = 0.0;
        for (Map.Entry<String, Float> fnv : features.entrySet()) {
            Tuple<Float, Float> param = modelMap.get(fnv.getKey());
            if (param == null) {
                continue;
            }
            score += param.v1 * transform(fnv.getKey(), fnv.getValue());
        }

        if (modelParams.need_bias) {
            Tuple<Float, Float> param = modelMap.get(modelParams.bias_feature_name);
            if (param != null) {
                score += param.v1;
            }
        }

        return score;
    }

    /**
     * Thompson sampling prediction for Exploitation and Exploration.
     * using Laplace approximate, Distribution of parameters posterior are approximate to diagonal gaussian distribution,
     * details see "An Empirical Evaluation of Thompson Sampling"
     * @param features features map, key:featureName, value:featureValue
     * @param alpha multiply standard deviation, control the exploitation and the exploration,
     *              the larger alpha value, the more exploration, the less exploitation.
     * @return prediction
     */
    public double thompsonSamplingPredict(Map<String, Float> features, double alpha) {
        features.remove(modelParams.bias_feature_name);

        if (commonParams.featureParams.feature_hash.need_feature_hash) {
            features = featureHash.hashMap2Map(features);
        }

        double score = 0.0;
        for (Map.Entry<String, Float> fnv : features.entrySet()) {
            Tuple<Float, Float> param = modelMap.get(fnv.getKey());
            if (param == null) {
                continue;
            }
            score += (param.v1 + rand.nextGaussian() * alpha * param.v2) * transform(fnv.getKey(), fnv.getValue());
        }

        if (modelParams.need_bias) {
            Tuple<Float, Float> param = modelMap.get(modelParams.bias_feature_name);
            if (param != null) {
                score += param.v1;
            }
        }

        return lossFunction.predict(score);
    }

}
