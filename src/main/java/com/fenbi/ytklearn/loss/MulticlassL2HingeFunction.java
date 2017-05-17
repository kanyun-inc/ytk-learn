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

public class MulticlassL2HingeFunction implements ILossFunction {
    private int target(double []label) {
        int target = -1;
        for (int i = 0; i < label.length; i++) {
            if (label[i] == 1.0) {
                target = i;
            }
        }

        return target;
    }

    @Override
    public double loss(double score, double label) {
        return 0;
    }

    @Override
    public double predict(double score) {
        return 0;
    }

    @Override
    public double loss(double[] score, double[] label) {
        int K = score.length;
        double loss = 0.0;
        int target = target(label);

        for (int j = 0; j < K; j++) {
            double margin = Math.max(0, score[j] - score[target] + 1);
            loss += margin * margin;
        }
        loss -= 1.0;
        loss *= 0.5;

        return loss;
    }

    @Override
    public void predict(double[] score, double[]pred) {
        int K = score.length;
        for (int i = 0; i < K; i++) {
            pred[i] = score[i];
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
        int K = score.length;
        double gradAccumu = 0.0;
        int target = target(label);
        for (int j = 0; j < K; j++) {
            double tmp = score[j] - score[target] + 1;
            firstDeri[j] = tmp > 0 ? tmp : 0.0;
            gradAccumu += firstDeri[j];
        }

        if (target != K - 1) {
            firstDeri[target] = -gradAccumu + 1.0;
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

        double loss = 0.0;
        int target = target(label);

        for (int j = 0; j < K; j++) {
            double margin = Math.max(0, score[j] - score[target] + 1);
            loss += margin * margin;
            predict[j] = score[j];
        }
        loss -= 1.0;
        loss *= 0.5;

        double gradAccumu = 0.0;
        for (int j = 0; j < K; j++) {
            double tmp = score[j] - score[target] + 1;
            firstDeri[j] = tmp > 0 ? tmp : 0.0;
            gradAccumu += firstDeri[j];
        }

        if (target != K - 1) {
            firstDeri[target] = -gradAccumu + 1.0;
        }

        return loss;
    }

    @Override
    public String getName() {
        return "multiclass_l2_hinge";
    }
}
