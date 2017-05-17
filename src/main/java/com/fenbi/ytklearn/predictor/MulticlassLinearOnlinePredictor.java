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

import com.fenbi.ytklearn.eval.ConfusionMatrixEvaluator;
import lombok.Data;

import java.io.Reader;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * @author xialong
 */

@Data
public class MulticlassLinearOnlinePredictor extends ContinuousOnlinePredictor<float[]> {
    private final int K;
    private final ThreadLocal<double[]> predbuffer = new ThreadLocal<>();
    private final ThreadLocal<double[]> wxbuffer = new ThreadLocal<>();

    public MulticlassLinearOnlinePredictor(String configPath) throws Exception {
        super(configPath);
        this.K = config.getInt("k");
        loadModel();
    }

    public MulticlassLinearOnlinePredictor(Reader configReader) throws Exception {
        super(configReader);
        this.K = config.getInt("k");
        loadModel();
    }

    public Object getEvalObjectInfo() {
        return new ConfusionMatrixEvaluator.ConfusionMatrixInfo(K, true);
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
                float []w = new float[K - 1];
                for (int k = 0; k < K - 1; k++) {
                    float wei = Float.parseFloat(info[1 + k]);
                    w[k] = wei;
                }

                modelMap.put(info[0], w);

                if (cnt++ < 10) {
                    LOG.info("fname:" + info[0] + ", w:" + Arrays.toString(w));
                }
            }
        }

        LOG.info("load model finished! feature cnt:" + cnt);
        return this;
    }

    @Override
    public double score(Map<String, Float> features, Object other) {
        return 0;
    }

    @Override
    public double[] scores(Map<String, Float> features, Object other) {
        features.remove(modelParams.bias_feature_name);

        if (commonParams.featureParams.feature_hash.need_feature_hash) {
            features = featureHash.hashMap2Map(features);
        }

        double []wx = wxbuffer.get();
        if (wx == null) {
            wx = new double[K];
            wxbuffer.set(wx);
        }
        if (modelParams.need_bias) {
            features.put(modelParams.bias_feature_name, 1.0f);
        }

        // wx
        for (int p = 0; p < K; p++) {
            wx[p] = 0.0;
        }
        for (Map.Entry<String, Float> feature : features.entrySet()) {
            float []w = modelMap.get(feature.getKey());
            if (w == null) {
                continue;
            }
            for (int p = 0; p < K - 1; p++) {
                wx[p] += w[p] * transform(feature.getKey(), feature.getValue());
            }
        }
//        // softmax
//        double maxXv = wx[K - 1];
//        for (int j = 0; j < K - 1; j++) {
//            if (wx[j] > maxXv) {
//                maxXv = wx[j];
//            }
//        }
//        double esum = 0.0;
//        for (int j = 0; j < K; j++) {
//            double newwx = wx[j] - maxXv;
//            wx[j] = Math.exp(newwx);
//            esum += wx[j];
//        }
//
//        // prob
//        double inv = 1.0 / esum;
//        for (int j = 0; j < K; j++) {
//            wx[j] = wx[j] * inv;
//        }

        return wx;
    }

    @Override
    public double loss(Map<String, Float> features, double[] labels, Object other) {
        return lossFunction.loss(scores(features, other), labels);
    }

    @Override
    public double[] predicts(Map<String, Float> features, Object other) {
        double []pred = predbuffer.get();
        if (pred == null) {
            pred = new double[K];
            predbuffer.set(pred);
        }
        lossFunction.predict(scores(features, null), pred);
        return pred;
    }
}
