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

import com.fenbi.ytklearn.dataflow.GBMLRDataFlow;
import com.fenbi.ytklearn.utils.MathUtils;

import java.io.Reader;
import java.util.Map;

/**
 * @author xialong
 */

public class GBHSDTOnlinePredictor extends GBSDTOnlinePredictor {
    protected final ThreadLocal<double[]> mubuffer = new ThreadLocal<>();
    protected final int L;

    public GBHSDTOnlinePredictor(String configPath) throws Exception {
        super(configPath);
        int l = -1;
        for (int i = 1; i <= 31; i++) {
            if ((1 << i) >= K) {
                l = i;
                break;
            }
        }
        this.L = l;
    }

    public GBHSDTOnlinePredictor(Reader configReader) throws Exception {
        super(configReader);
        int l = -1;
        for (int i = 1; i <= 31; i++) {
            if ((1 << i) >= K) {
                l = i;
                break;
            }
        }
        this.L = l;
    }

    @Override
    public double score(Map<String, Float> features, Object other) {
        features.remove(modelParams.bias_feature_name);

        if (commonParams.featureParams.feature_hash.need_feature_hash) {
            features = featureHash.hashMap2Map(features);
        }

        int stride = K - 1;
        int size = stride * treeNum;

        double []wx = wxbuffer.get();
        if (wx == null) {
            wx = new double[size];
            wxbuffer.set(wx);
        }

        double []gating = gatingbuffer.get();
        if (gating == null) {
            gating = new double[K * treeNum];
            gatingbuffer.set(gating);
        }

        double []mu = mubuffer.get();
        if (mu == null) {
            mu = new double[(2 * K - 1) * treeNum];
            mubuffer.set(mu);
        }

        // xv, xw
        // bias
        float []w = modelMap.get(modelParams.bias_feature_name);
        if (w != null && modelParams.need_bias) {
            for (int p = 0; p < size; p++) {
                wx[p] = w[p];
            }
        } else {
            for (int p = 0; p < size; p++) {
                wx[p] = 0;
            }
        }


        for (Map.Entry<String, Float> f : features.entrySet()) {
            w = modelMap.get(f.getKey());
            if (w == null) {
                continue;
            }
            for (int i = 0; i < size; i++) {
                wx[i] += w[i] * transform(f.getKey(), f.getValue());
            }
        }

        int idx = 0;
        int idxg = 0;
        int idxm = 0;
        double fx = 0.0;
        for (int tree = 0; tree < treeNum; tree++) {
            for (int j = 0; j < K - 1; j++) {
                wx[idx + j] = MathUtils.logistic(wx[idx + j]);
            }

            // calc mu
            for (int j = stride; j < 2 * K - 1; j++) {
                int gidx = j - stride;
                gating[idxg + gidx] = 1.0;
                int prevIdx = j + 1;
                int curIdx;
                for (int p = 0; p < L; p++) {
                    curIdx = prevIdx >>> 1;
                    gating[idxg + gidx] *= ((prevIdx & 1) == 0 ? wx[idx + curIdx - 1] : 1.0 - wx[idx + curIdx - 1]);
                    prevIdx = curIdx;
                    if (curIdx == 1) {
                        break;
                    }
                }
                mu[idxm + j] = gating[idxg + gidx] * leaf[tree][j - stride];
            }
            double gs = 0.0;
            for (int i = 0; i < K; i++) {
                gs += gating[idxg + i];
            }

            for (int j = stride - 1; j >= 0; j--) {
                int vidx = ((j + 1) << 1) - 1;
                mu[idxm + j] = mu[idxm + vidx] + mu[idxm + vidx + 1];
            }
            fx += learningRate * mu[idxm];

            idx += K - 1;
            idxg += K;
            idxm += 2 * K - 1;
        }

        double lbias = uniformBaseScore;
        if (sampleDepdtBaseScore) {
            lbias += lossFunction.pred2Score(((Float)other).floatValue());
        }

        if (type == GBMLRDataFlow.Type.RF) {
            fx /= treeNum;
        }

        return lbias + fx;
    }
}
