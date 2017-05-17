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
public class FMOnlinePredictor extends ContinuousOnlinePredictor<float[]> {
    private final int K;
    private final ThreadLocal<double[]> sumbuffer = new ThreadLocal<>();

    public FMOnlinePredictor(String configPath) throws Exception {
        super(configPath);

        List<Integer> klist = config.getIntList("k");
        K = klist.get(1);

        loadModel();

    }

    public FMOnlinePredictor(Reader configReader) throws Exception {
        super(configReader);

        List<Integer> klist = config.getIntList("k");
        K = klist.get(1);

        loadModel();

    }

    @Override
    protected OnlinePredictor loadModel() throws Exception {
        if (!fs.exists(modelParams.data_path)) {
            throw new Exception("fm model doesn't exist! path:" + modelParams.data_path);
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


                float []w = modelMap.get(info[0]);
                if (w == null) {
                    w = new float[1 + K];
                    modelMap.put(info[0], w);
                }

                for (int k = 0; k < K + 1; k++) {
                    float wei = Float.parseFloat(info[1 + k]);
                    w[k] = wei;
                }

                if (cnt < 10) {
                    LOG.info("feature:" + info[0] + ", wei:" + Arrays.toString(w));
                }
                cnt++;
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

        double []sum = sumbuffer.get();
        int size = K << 1;
        if (sum == null) {
            sum = new double[size];
            sumbuffer.set(sum);
        }

        double wx = 0.0;
        for (int f = 0; f < size; f++) {
            sum[f] = 0.0;
        }
        int idx = 0;

        float []w = modelMap.get(modelParams.bias_feature_name);
        if (w != null && modelParams.need_bias) {
            wx += w[0];
            for (int f = 0; f < K; f ++) {
                sum[idx] += w[1 + f];
                sum[idx + 1] += w[1 + f] * w[1 + f];
                idx += 2;
            }
        }
        for (Map.Entry<String, Float> ft : features.entrySet()) {
            w = modelMap.get(ft.getKey());
            if (w == null) {
                continue;
            }
            float val = transform(ft.getKey(), ft.getValue());
            wx += w[0] * val;
            idx = 0;
            for (int f = 0; f < K; f ++) {
                double v = w[1 + f] * val;
                sum[idx] += v;
                sum[idx + 1] += v * v;
                idx += 2;
            }
        }

        double fx = 0.0;
        idx = 0;
        for (int f = 0; f < K; f++) {
            fx += (sum[idx] * sum[idx] - sum[idx + 1]);
            idx += 2;
        }
        fx *= 0.5;
        fx += wx;

        return (float)fx;
    }

}
