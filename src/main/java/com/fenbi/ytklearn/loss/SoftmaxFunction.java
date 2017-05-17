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

/**
 * @author xialong
 */

public class SoftmaxFunction implements ILossFunction {

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
        // softmax
        double maxXv = score[K - 1];
        for (int j = 0; j < K - 1; j++) {
            if (score[j] > maxXv) {
                maxXv = score[j];
            }
        }
        double esum = 0.0;
        double sigmawx = 0.0;
        for (int j = 0; j < K; j++) {
            double newwx = score[j] - maxXv;
            sigmawx += newwx * label[j];
            esum += Math.exp(newwx);
        }

        // loss
        return Math.log(esum) - sigmawx;
    }

    @Override
    public void predict(double[] score, double[]pred) {

        int K = score.length;
        // softmax
        double maxXv = score[K - 1];
        for (int j = 0; j < K - 1; j++) {
            if (score[j] > maxXv) {
                maxXv = score[j];
            }
        }
        double esum = 0.0;
        for (int j = 0; j < K; j++) {
            pred[j] = Math.exp(score[j] - maxXv);
            esum += pred[j];
        }

        double inv = 1.0 / esum;
        for (int j = 0; j < K; j++) {
            pred[j] = pred[j] * inv;
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
        for (int i = 0; i < pred.length; i++) {
            firstDeri[i] = pred[i] - label[i];
            secondDeri[i] = 2 * (pred[i] * (1 - pred[i]));
        }
    }

    @Override
    public double all(double []score,
                      double []label,
                      double []predict,
                      double []firstDeri,
                      double []secondDeri
    ) {
        int K = score.length;
        double loss;
        // softmax
        double maxXv = score[K - 1];
        for (int j = 0; j < K - 1; j++) {
            if (score[j] > maxXv) {
                maxXv = score[j];
            }
        }
        double esum = 0.0;
        double sigmascore = 0.0;
        for (int j = 0; j < K; j++) {
            double newscore = score[j] - maxXv;
            sigmascore += newscore * label[j];
            score[j] = Math.exp(newscore);
            esum += score[j];
        }

        loss = (Math.log(esum) - sigmascore);

        // prob
        double inv = 1.0 / esum;
        for (int j = 0; j < K; j++) {
            score[j] = score[j] * inv;
            predict[j] = (float) score[j];
            firstDeri[j] = predict[j] - label[j];
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
        return "softmax";
    }
}
