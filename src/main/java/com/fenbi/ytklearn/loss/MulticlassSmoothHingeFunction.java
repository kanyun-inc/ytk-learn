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

public class MulticlassSmoothHingeFunction implements ILossFunction {

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
            double diff = score[j] - score[target];
            if (diff >= 0) {
                loss += diff + 0.5;
            } else if (diff < -1) {
                loss += 0.0;
            } else if (diff >= -1 && diff < 0.0) {
                loss += 0.5 * (1 + diff) * (1 + diff);
            }
        }
        loss -= 0.5;

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
            double diff = score[j] - score[target];
            if (diff >= 0) {
                firstDeri[j] = 1;
            } else if (diff < -1) {
                firstDeri[j] = 0.0;
            } else if (diff >= -1 && diff < 0.0) {
                firstDeri[j] = (1 + diff);
            }
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
        double gradAccumu = 0.0;

        double loss = 0.0;
        int target = target(label);

        for (int j = 0; j < K; j++) {
            double diff = score[j] - score[target];
            if (diff >= 0) {
                loss += diff + 0.5;
            } else if (diff < -1) {
                loss += 0.0;
            } else if (diff >= -1 && diff < 0.0) {
                loss += 0.5 * (1 + diff) * (1 + diff);
            }
            predict[j] = score[j];
        }
        loss -= 0.5;

        for (int j = 0; j < K; j++) {
            double diff = score[j] - score[target];
            if (diff >= 0) {
                firstDeri[j] = 1;
            } else if (diff < -1) {
                firstDeri[j] = 0.0;
            } else if (diff >= -1 && diff < 0.0) {
                firstDeri[j] = (1 + diff);
            }
            gradAccumu += firstDeri[j];

        }

        if (target != K - 1) {
            firstDeri[target] = -gradAccumu + 1.0;
        }
        return loss;
    }

    @Override
    public String getName() {
        return "multiclass_smooth_hinge";
    }
}
