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

package com.fenbi.ytklearn.optimizer;

import com.fenbi.ytklearn.dataflow.ContinuousDataFlow;
import com.fenbi.ytklearn.dataflow.MulticlassLinearModelDataFlow;
import com.fenbi.ytklearn.eval.ConfusionMatrixEvaluator;

/**
 * @author xialong
 */

public class MulticlassLinearHoagOptimizer extends HoagOptimizer {
    private final int K;
    private final double []wx;
    private final double []pred;
    private final double []firstDeri;
    private final double []label;

    public MulticlassLinearHoagOptimizer(String modelName,
                                         ContinuousDataFlow dataFlow,
                                         int threadIdx) throws Exception {
        super(modelName, dataFlow, threadIdx);

        MulticlassLinearModelDataFlow multiclassLinearModelDataFlow = (MulticlassLinearModelDataFlow)dataFlow;
        K = multiclassLinearModelDataFlow.getK();
        wx = new double[K];
        pred = new double[K];
        firstDeri = new double[K];
        label = new double[K];

//        l2[0] /= (K - 1);
//        l1[0] /= (K - 1);
    }


    @Override
    public int[] getRegularStart() {
        int start[] = new int[1];
        if (modelParams.need_bias) {
            start[0] = K - 1;
        } else {
            start[1] = 0;
        }
        return start;
    }

    @Override
    public int[] getRegularEnd() {
        int end[] = new int[1];
        end[0] = dim;
        return end;
    }

    @Override
    public Object getEvalObjectInfo() {
        return new ConfusionMatrixEvaluator.ConfusionMatrixInfo(K, true);
    }

    @Override
    public double calcPureLossAndGrad(float[] w, float[] g, int iter) {
        double loss = 0.0;
        int stride = K - 1;
        for (int i = 0; i < dim; i++) {
            g[i] = 0.0f;
        }
        for (int k = 0; k < threadTrainCoreData.cursor2d; k++) {
            int lsNumInt = (int)realNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                double wei = weight[k][i];
                // wx
                int yidx = i * K;
                for (int p = 0; p < K; p++) {
                    wx[p] = 0.0;
                    label[p] = y[k][yidx + p];
                }
                for (int j = xidx[k][i]; j < xidx[k][i + 1]; j+=2) {
                    double fval = Float.intBitsToFloat(x[k][j+1]);
                    int idx = x[k][j] * stride;
                    for (int p = 0; p < stride; p++) {
                        wx[p] += w[idx + p] * fval;
                    }
                }
//                // softmax
//                double maxXv = wx[K - 1];
//                for (int j = 0; j < stride; j++) {
//                    if (wx[j] > maxXv) {
//                        maxXv = wx[j];
//                    }
//                }
//                double esum = 0.0;
//                double sigmawx = 0.0;
//                int yidx = i * K;
//                for (int j = 0; j < K; j++) {
//                    double newwx = wx[j] - maxXv;
//                    sigmawx += newwx * y[k][yidx + j];
//                    wx[j] = Math.exp(newwx);
//                    esum += wx[j];
//                }
//
//                // loss
//                loss += wei * (Math.log(esum) - sigmawx);
//                //loss += wei * lossFunction.loss(wx, label);
//
//                // prob
//                double inv = 1.0 / esum;
//                for (int j = 0; j < K; j++) {
//                    wx[j] = wx[j] * inv;
//                    predict[k][yidx + j] = (float)wx[j];
//                }
                loss += wei * lossFunction.all(wx, label, pred, firstDeri, null);
                for (int p = 0; p < K; p++) {
                    predict[k][yidx + p] = (float)pred[p];
                }

                // grad
                for (int j = xidx[k][i]; j < xidx[k][i + 1]; j+=2) {
                    int idx = x[k][j] * stride;
                    double xv = Float.intBitsToFloat(x[k][j+1]);
                    for (int p = 0; p < stride; p++) {
                        //g[idx + p] += wei * (wx[p] - y[k][yidx + p]) * xv;
                        g[idx + p] += wei * firstDeri[p] * xv;
                    }
                }
            }
        }
        return loss;
    }

    @Override
    public double calTestPureLossAndGrad(float []wtest, float []gtest, int iter, boolean needCalcGrad) {
        if (!hasTestData) {
            return -1.0;
        }

        double loss = 0.0;
        int stride = K - 1;
        if (needCalcGrad) {
            for (int i = 0; i < dim; i++) {
                gtest[i] = 0.0f;
            }
        }

        for (int k = 0; k < threadTestCoreData.cursor2d; k++) {
            int lsNumInt = (int)realNumtest[k];
            for (int i = 0; i < lsNumInt; i++) {
                double wei = weighttest[k][i];
                // wx
                int yidx = i * K;
                for (int p = 0; p < K; p++) {
                    wx[p] = 0.0;
                    label[p] = ytest[k][yidx + p];
                }
                for (int j = xidxtest[k][i]; j < xidxtest[k][i + 1]; j+=2) {
                    double fval = Float.intBitsToFloat(xtest[k][j+1]);
                    int idx = xtest[k][j] * stride;
                    for (int p = 0; p < stride; p++) {
                        wx[p] += wtest[idx + p] * fval;
                    }
                }
                // softmax
//                double maxXv = wx[K - 1];
//                for (int j = 0; j < stride; j++) {
//                    if (wx[j] > maxXv) {
//                        maxXv = wx[j];
//                    }
//                }
//                double esum = 0.0;
//                double sigmawx = 0.0;
//                int yidx = i * K;
//                for (int j = 0; j < K; j++) {
//                    double newwx = wx[j] - maxXv;
//                    sigmawx += newwx * ytest[k][yidx + j];
//                    wx[j] = Math.exp(newwx);
//                    esum += wx[j];
//                }
//
//                // loss
//                loss += wei * (Math.log(esum) - sigmawx);
//
//                // prob
//                double inv = 1.0 / esum;
//                for (int j = 0; j < K; j++) {
//                    wx[j] = wx[j] * inv;
//                    predicttest[k][yidx + j] = (float)wx[j];
//                }

                loss += wei * lossFunction.all(wx, label, pred, firstDeri, null);
                for (int p = 0; p < K; p++) {
                    predicttest[k][yidx + p] = (float)pred[p];
                }

                if (needCalcGrad) {
                    // grad
                    for (int j = xidxtest[k][i]; j < xidxtest[k][i + 1]; j+=2) {
                        int idx = xtest[k][j] * stride;
                        double xv = Float.intBitsToFloat(xtest[k][j+1]);
                        for (int p = 0; p < stride; p++) {
                            //gtest[idx + p] += wei * (wx[p] - ytest[k][yidx + p]) * xv;
                            gtest[idx + p] += wei * firstDeri[p] * xv;
                        }
                    }
                }

            }
        }
        return loss;
    }
}
