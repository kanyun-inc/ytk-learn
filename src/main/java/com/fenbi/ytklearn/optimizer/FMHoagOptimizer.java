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

import com.fenbi.ytklearn.dataflow.FMModelDataFlow;
import com.fenbi.ytklearn.dataflow.ContinuousDataFlow;

/**
 * @author xialong
 */
public class FMHoagOptimizer extends HoagOptimizer {

    private final int firstOrderIndexStart;
    private final int secondOrderIndexStart;

    private final double []soSum;
    private final double []soSum2;

    private final int[]K;
    private final int sok;

    private final boolean needFirstOrder;
    private final boolean needSecondOrder;
    private final boolean biasNeedLatentFactor;

    public FMHoagOptimizer(String modelName,
                           ContinuousDataFlow dataFlow,
                           int threadIdx) throws Exception {
        super(modelName, dataFlow, threadIdx);

        FMModelDataFlow fmModelDataFlow = (FMModelDataFlow)dataFlow;
        firstOrderIndexStart = fmModelDataFlow.firstOrderIndexStart();
        secondOrderIndexStart = fmModelDataFlow.secondOrderIndexStart();
        K = fmModelDataFlow.getK();
        sok = K[1];

        soSum = new double[sok];
        soSum2 = new double[sok];

        needFirstOrder = fmModelDataFlow.isNeedFirstOrder();
        needSecondOrder = fmModelDataFlow.isNeedSecondOrder();
        biasNeedLatentFactor = fmModelDataFlow.isBiasNeedLatentFactor();

//        if (needSecondOrder) {
//            l2[1] /= K[1];
//            l1[1] /= K[1];
//        }
    }

    @Override
    public int[] getRegularStart() {
        int []start = new int[2];
        start[0] = firstOrderIndexStart;
        start[1] = secondOrderIndexStart;
        return start;
    }

    @Override
    public int[] getRegularEnd() {
        int []end = new int[2];
        end[0] = secondOrderIndexStart;
        end[1] = dim;
        return end;
    }

    @Override
    public double calcPureLossAndGrad(float[] w, float[] g, int iter) {
        double loss = 0.0;
        for (int i = 0; i < dim; i++) {
            g[i] = 0.0f;
        }

        for (int k = 0; k < threadTrainCoreData.cursor2d; k++) {
            int lsNumInt = (int)realNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                double wei = weight[k][i];

                for (int f = 0; f < sok; f++) {
                    soSum[f] = 0.0;
                    soSum2[f] = 0.0;
                }

                double wx = 0.0;
                for (int j = xidx[k][i]; j < xidx[k][i + 1]; j+=2) {
                    double fval = Float.intBitsToFloat(x[k][j+1]);
                    wx += w[x[k][j]] * fval;
                    int idx = secondOrderIndexStart + x[k][j] * sok;
                    for (int f = 0; f < sok; f++) {
                        double v = w[idx + f] * fval;
                        soSum[f] += v;
                        soSum2[f] += v * v;
                    }

                }

                double fx = 0.0;
                for (int f = 0; f < sok; f++) {
                    fx += (soSum[f] * soSum[f] - soSum2[f]);
                }
                fx *= 0.5;
                fx += wx;

                loss += wei * lossFunction.loss(fx, y[k][i]);
                predict[k][i] = (float) lossFunction.predict(fx);

                // grad
                double gradscalar = wei * lossFunction.firstDerivative(fx, y[k][i]);
                for (int j = xidx[k][i]; j < xidx[k][i + 1]; j+=2) {
                    double fval = Float.intBitsToFloat(x[k][j+1]);
                    g[x[k][j]] += gradscalar * fval;

                    int idx = secondOrderIndexStart + x[k][j] * sok;
                    for (int f = 0; f < sok; f++) {
                        g[idx + f] += gradscalar * (soSum[f] - w[idx + f] * fval) * fval;
                    }
                }
            }
        }

        if (!needFirstOrder) {
            for (int i = firstOrderIndexStart; i < secondOrderIndexStart; i++) {
                g[i] = 0.0f;
            }
        }

        if (!needSecondOrder) {
            for (int i = secondOrderIndexStart; i < g.length; i++) {
                g[i] = 0.0f;
            }
        }

        if (!biasNeedLatentFactor && needSecondOrder && modelParams.need_bias) {
            for (int i = secondOrderIndexStart; i < secondOrderIndexStart + sok; i++) {
                g[i] = 0.0f;
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
        if (needCalcGrad) {
            for (int i = 0; i < dim; i++) {
                gtest[i] = 0.0f;
            }
        }


        for (int k = 0; k < threadTestCoreData.cursor2d; k++) {
            int lsNumInt = (int)realNumtest[k];
            for (int i = 0; i < lsNumInt; i++) {
                double wei = weighttest[k][i];

                for (int f = 0; f < sok; f++) {
                    soSum[f] = 0.0;
                    soSum2[f] = 0.0;
                }

                double wx = 0.0;
                for (int j = xidxtest[k][i]; j < xidxtest[k][i + 1]; j+=2) {
                    double fval = Float.intBitsToFloat(xtest[k][j+1]);
                    wx += wtest[xtest[k][j]] * fval;
                    int idx = secondOrderIndexStart + xtest[k][j] * sok;
                    for (int f = 0; f < sok; f++) {
                        double v = wtest[idx + f] * fval;
                        soSum[f] += v;
                        soSum2[f] += v * v;
                    }

                }

                double fx = 0.0;
                for (int f = 0; f < sok; f++) {
                    fx += (soSum[f] * soSum[f] - soSum2[f]);
                }
                fx *= 0.5;
                fx += wx;

                loss += wei * lossFunction.loss(fx, ytest[k][i]);
                predicttest[k][i] = (float) lossFunction.predict(fx);

                if (needCalcGrad) {

                    // grad
                    double gradscalar = wei * lossFunction.firstDerivative(fx, ytest[k][i]);
                    for (int j = xidxtest[k][i]; j < xidxtest[k][i + 1]; j+=2) {
                        double fval = Float.intBitsToFloat(xtest[k][j+1]);
                        gtest[xtest[k][j]] += gradscalar * fval;

                        int idx = secondOrderIndexStart + xtest[k][j] * sok;
                        for (int f = 0; f < sok; f++) {
                            gtest[idx + f] += gradscalar * (soSum[f] - wtest[idx + f] * fval) * fval;
                        }
                    }
                }

            }
        }

        if (needCalcGrad) {
            if (!needFirstOrder) {
                for (int i = firstOrderIndexStart; i < secondOrderIndexStart; i++) {
                    gtest[i] = 0.0f;
                }
            }

            if (!needSecondOrder) {
                for (int i = secondOrderIndexStart; i < gtest.length; i++) {
                    gtest[i] = 0.0f;
                }
            }

            if (!biasNeedLatentFactor && needSecondOrder && modelParams.need_bias) {
                for (int i = secondOrderIndexStart; i < secondOrderIndexStart + sok; i++) {
                    gtest[i] = 0.0f;
                }
            }

        }

        return loss;
    }

}
