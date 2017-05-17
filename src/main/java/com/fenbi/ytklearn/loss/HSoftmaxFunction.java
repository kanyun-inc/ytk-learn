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

package com.fenbi.ytklearn.loss;

import com.fenbi.ytklearn.utils.MathUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author xialong
 */

public class HSoftmaxFunction implements ILossFunction {
    public static final Logger LOG = LoggerFactory.getLogger(HSoftmaxFunction.class);

    private double []mu;
    private double []gx;

    @Override
    public double loss(double score, double label) {
        return 0;
    }

    @Override
    public double predict(double score) {
        return score;
    }

    @Override
    public double loss(double[] score, double[] label) {
        int K = score.length;
        int stride = 2 * K - 1;
        int vstride = K - 1;

        if (mu == null) {
            mu = new double[stride];
        }

        for (int j = vstride; j < stride; j++) {
            mu[j] = label[j - vstride];
        }

        for (int j = vstride - 1; j >= 0; j--) {
            int idx = ((j + 1) << 1) - 1;
            mu[j] = mu[idx] + mu[idx + 1];
        }

        double loss = 0.0;
        for (int k = 1; k <= vstride; k++) {
            if (score[k - 1] >= 0.0) {
                loss += (mu[2 * k] * score[k - 1] + mu[k - 1] * Math.log(1.0 + Math.exp(-score[k - 1])));
            } else {
                loss += (mu[k - 1] * Math.log(1.0 + Math.exp(score[k - 1])) - mu[2 * k - 1] * score[k - 1]);
            }

        }
        return loss;
    }

    @Override
    public void predict(double[] score, double[]pred) {

        int K = score.length;
        int stride = 2 * K - 1;
        int vstride = K - 1;

        if (gx == null) {
            gx = new double[stride];
        }

        for (int i = 0; i < vstride; i++) {
            gx[i] = MathUtils.logistic(score[i]);
        }

        for (int j = vstride; j < stride; j++) {
            int gidx = j - vstride;
            pred[gidx] = 1.0;
            int prevIdx = j + 1;
            int curIdx;
            for (int p = 0; p < 100; p++) {
                curIdx = prevIdx >>> 1;
                pred[gidx] *= ((prevIdx & 1) == 0 ? gx[curIdx - 1] : 1.0 - gx[curIdx - 1]);
                prevIdx = curIdx;
                if (curIdx == 1) {
                    break;
                }
            }
        }
    }

    @Override
    public double firstDerivative(double score, double label) {
        return 0;
    }

    @Override
    public double secondDerivative(double score, double label) {
        return 0;
    }

    @Override
    public void firstDerivative(double []score, double []label, double []firstDeri) {
        predict(score, firstDeri);
        for (int i = 0; i < score.length; i++) {
            firstDeri[i] = firstDeri[i] - label[i];
        }
    }


    @Override
    public void getDerivativeFast(double []pred, double []label, double[] firstDeri, double[] secondDeri) {

    }

    @Override
    public double all(double []score,
                      double []label,
                      double []predict,
                      double []firstDeri,
                      double []secondDeri
    ) {
        // TODO: softmax, softmax_cross_entropy diff
        int K = score.length;
        int stride = 2 * K - 1;
        int vstride = K - 1;

        if (mu == null) {
            mu = new double[stride];
        }
        if (gx == null) {
            gx = new double[stride];
        }

        for (int i = 0; i < vstride; i++) {
            gx[i] = MathUtils.logistic(score[i]);
        }

        double pred = 0;
        for (int j = vstride; j < stride; j++) {
            int gidx = j - vstride;
            predict[gidx] = 1.0;
            int prevIdx = j + 1;
            int curIdx;
            for (int p = 0; p < 100; p++) {
                curIdx = prevIdx >>> 1;
                predict[gidx] *= ((prevIdx & 1) == 0 ? gx[curIdx - 1] : 1.0 - gx[curIdx - 1]);
                prevIdx = curIdx;
                if (curIdx == 1) {
                    break;
                }
            }
            pred += predict[gidx];
            mu[j] = label[j - vstride];
        }

        for (int j = vstride - 1; j >= 0; j--) {
            int idx = ((j + 1) << 1) - 1;
            mu[j] = mu[idx] + mu[idx + 1];
        }

        double loss = 0.0;
        for (int k = 1; k <= vstride; k++) {
            if (score[k - 1] >= 0.0) {
                loss += (mu[2 * k] * score[k - 1] + mu[k - 1] * Math.log(1.0 + Math.exp(-score[k - 1])));
            } else {
                loss += (mu[k - 1] * Math.log(1.0 + Math.exp(score[k - 1])) - mu[2 * k - 1] * score[k - 1]);
            }

            firstDeri[k - 1] = gx[k - 1] * mu[k - 1]  - mu[2 * k - 1];
        }
        return loss;
    }


    @Override
    public boolean checkLabel(float[] label) {
        double sum = 0.0;
        for (int i = 0; i < label.length; i++) {
            sum += label[i];
        }

        if (Math.abs(sum - 1.f) < 1E-3) {
            return true;
        } else {
            return false;
        }
    }

    @Override
    public String getName() {
        return "hsoftmax";
    }
}
