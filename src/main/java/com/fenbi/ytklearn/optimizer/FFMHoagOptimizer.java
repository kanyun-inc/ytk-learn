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

import com.fenbi.ytklearn.dataflow.FFMModelDataFlow;
import com.fenbi.ytklearn.dataflow.ContinuousDataFlow;

/**
 * @author xialong
 */

public class FFMHoagOptimizer extends HoagOptimizer {

    private final int firstOrderIndexStart;
    private final int secondOrderIndexStart;

    private final int[]K;
    private final int sok;

    private final boolean needFirstOrder;
    private final boolean needSecondOrder;
    private final boolean biasNeedLatentFactor;
    private final int fieldSize;

    private final float[] assist;

    public FFMHoagOptimizer(String modelName,
                            ContinuousDataFlow dataFlow,
                            int threadIdx) throws Exception {
        super(modelName, dataFlow, threadIdx);

        FFMModelDataFlow ffmModelDataFlow = (FFMModelDataFlow)dataFlow;
        firstOrderIndexStart = ffmModelDataFlow.firstOrderIndexStart();
        secondOrderIndexStart = ffmModelDataFlow.secondOrderIndexStart();
        K = ffmModelDataFlow.getK();
        sok = K[1];

        needFirstOrder = ffmModelDataFlow.isNeedFirstOrder();
        needSecondOrder = ffmModelDataFlow.isNeedSecondOrder();
        biasNeedLatentFactor = ffmModelDataFlow.isBiasNeedLatentFactor();
        fieldSize = ffmModelDataFlow.getFieldSize();

//        if (needSecondOrder) {
//            l2[1] /= (sok * fieldSize);
//            l1[1] /= (sok * fieldSize);
//        }

        int maxFeatureNum = ffmModelDataFlow.getMaxFeatureNum();
        assist = new float[sok * fieldSize * (maxFeatureNum + 1)];
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

        int stride = sok * fieldSize;
        for (int k = 0; k < threadTrainCoreData.cursor2d; k++) {
            int lsNumInt = realNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                double wei = weight[k][i];

                double wx = 0.0;
                int cidx = 0;
                for (int j = xidx[k][i]; j < xidx[k][i + 1]; j+=3) {
                    if (x[k][j] == -1) {
                        continue;
                    }
                    double fval = Float.intBitsToFloat(x[k][j+1]);
                    wx += w[x[k][j]] * fval;
                    int idx = secondOrderIndexStart + x[k][j] * stride;
                    System.arraycopy(w, idx, assist, cidx * stride, stride);
                    cidx ++;
                }

                double fx = 0.0;
                int pidx = 0;
                for (int p = xidx[k][i]; p < xidx[k][i + 1]; p += 3) {
                    double pval = Float.intBitsToFloat(x[k][p + 1]);
                    int pfieldstart = x[k][p + 2] * sok;
                    int qidx = pidx + 1;

                    int pstartIdx = pidx * stride;
                    for (int q = p + 3; q < xidx[k][i + 1]; q += 3) {
                        double qval = Float.intBitsToFloat(x[k][q + 1]);
                        int qfieldstart = x[k][q + 2] * sok;
                        int qstartIdx = qidx * stride;
                        double wTw = 0.0;
                        for (int f = 0; f < sok; f++) {
                            wTw += assist[pstartIdx + qfieldstart + f] * assist[qstartIdx + pfieldstart + f];
                        }
                        wTw *= pval * qval;
                        fx += wTw;
                        qidx ++;
                    }
                    pidx ++;
                }
                fx += wx;

                loss += wei * lossFunction.loss(fx, y[k][i]);
                predict[k][i] = (float) lossFunction.predict(fx);

                // grad
                double gradscalar = wei * lossFunction.firstDerivative(fx, y[k][i]);
                for (int j = xidx[k][i]; j < xidx[k][i + 1]; j+=3) {
                    double fval = Float.intBitsToFloat(x[k][j+1]);
                    g[x[k][j]] += gradscalar * fval;
                }

                pidx = 0;
                for (int p = xidx[k][i]; p < xidx[k][i + 1]; p += 3) {
                    double pval = Float.intBitsToFloat(x[k][p + 1]);
                    int pfieldstart = x[k][p + 2] * sok;
                    int qidx = pidx + 1;

                    //int pstartIdx = pidx * stride;
                    int gidx = secondOrderIndexStart + x[k][p] * stride;
                    for (int q = p + 3; q < xidx[k][i + 1]; q += 3) {
                        double qval = Float.intBitsToFloat(x[k][q + 1]);
                        int qfieldstart = x[k][q + 2] * sok;
                        int qstartIdx = qidx * stride;
                        for (int f = 0; f < sok; f++) {
                            g[gidx + qfieldstart + f] += gradscalar * assist[qstartIdx + pfieldstart + f] * pval * qval;
                        }
                        qidx ++;
                    }
                    pidx ++;
                }

                int qidx = 1;
                for (int q = xidx[k][i] + 3; q < xidx[k][i + 1]; q += 3) {
                    double qval = Float.intBitsToFloat(x[k][q + 1]);
                    int qfieldstart = x[k][q + 2] * sok;
                    pidx = qidx - 1;

                    //int pstartIdx = pidx * stride;
                    int gidx = secondOrderIndexStart + x[k][q] * stride;
                    for (int p = q - 3; p >= xidx[k][i]; p -= 3) {
                        double pval = Float.intBitsToFloat(x[k][p + 1]);
                        int pfieldstart = x[k][p + 2] * sok;
                        int pstartIdx = pidx * stride;
                        for (int f = 0; f < sok; f++) {
                            g[gidx + pfieldstart + f] += gradscalar * assist[pstartIdx + qfieldstart + f] * pval * qval;
                        }
                        pidx --;
                    }
                    qidx ++;
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
            for (int i = secondOrderIndexStart; i < secondOrderIndexStart + stride; i++) {
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


        int stride = sok * fieldSize;
        for (int k = 0; k < threadTestCoreData.cursor2d; k++) {
            int lsNumInt = (int)realNumtest[k];
            for (int i = 0; i < lsNumInt; i++) {
                double wei = weighttest[k][i];

                double wx = 0.0;
                int cidx = 0;
                for (int j = xidxtest[k][i]; j < xidxtest[k][i + 1]; j+=3) {
                    double fval = Float.intBitsToFloat(xtest[k][j+1]);
                    wx += wtest[xtest[k][j]] * fval;
                    int idx = secondOrderIndexStart + xtest[k][j] * stride;
                    System.arraycopy(wtest, idx, assist, cidx * stride, stride);
                    cidx ++;
                }

                double fx = 0.0;
                int pidx = 0;
                for (int p = xidxtest[k][i]; p < xidxtest[k][i + 1]; p += 3) {
                    double pval = Float.intBitsToFloat(xtest[k][p + 1]);
                    int pfieldstart = xtest[k][p + 2] * sok;
                    int qidx = pidx + 1;

                    int pstartIdx = pidx * stride;
                    for (int q = p + 3; q < xidxtest[k][i + 1]; q += 3) {
                        double qval = Float.intBitsToFloat(xtest[k][q + 1]);
                        int qfieldstart = xtest[k][q + 2] * sok;
                        int qstartIdx = qidx * stride;
                        double wTw = 0.0;
                        for (int f = 0; f < sok; f++) {
                            wTw += assist[pstartIdx + qfieldstart + f] * assist[qstartIdx + pfieldstart + f];
                        }
                        wTw *= pval * qval;
                        fx += wTw;
                        qidx ++;
                    }
                    pidx ++;
                }
                fx += wx;

                loss += wei * lossFunction.loss(fx, ytest[k][i]);
                predicttest[k][i] = (float) lossFunction.predict(fx);

                if (needCalcGrad) {

                    // grad
                    double gradscalar = wei * lossFunction.firstDerivative(fx, ytest[k][i]);
                    for (int j = xidxtest[k][i]; j < xidxtest[k][i + 1]; j+=3) {
                        double fval = Float.intBitsToFloat(xtest[k][j+1]);
                        gtest[xtest[k][j]] += gradscalar * fval;
                    }

                    pidx = 0;
                    for (int p = xidxtest[k][i]; p < xidxtest[k][i + 1]; p += 3) {
                        double pval = Float.intBitsToFloat(xtest[k][p + 1]);
                        int pfieldstart = xtest[k][p + 2] * sok;
                        int qidx = pidx + 1;

                        //int pstartIdx = pidx * stride;
                        int gidx = secondOrderIndexStart + xtest[k][p] * stride;
                        for (int q = p + 3; q < xidxtest[k][i + 1]; q += 3) {
                            double qval = Float.intBitsToFloat(xtest[k][q + 1]);
                            int qfieldstart = xtest[k][q + 2] * sok;
                            int qstartIdx = qidx * stride;
                            for (int f = 0; f < sok; f++) {
                                gtest[gidx + qfieldstart + f] += gradscalar * assist[qstartIdx + pfieldstart + f] * pval * qval;
                            }
                            qidx ++;
                        }
                        pidx ++;
                    }

                    int qidx = 1;
                    for (int q = xidxtest[k][i] + 3; q < xidxtest[k][i + 1]; q += 3) {
                        double qval = Float.intBitsToFloat(xtest[k][q + 1]);
                        int qfieldstart = xtest[k][q + 2] * sok;
                        pidx = qidx - 1;

                        //int pstartIdx = pidx * stride;
                        int gidx = secondOrderIndexStart + xtest[k][q] * stride;
                        for (int p = q - 3; p >= xidxtest[k][i]; p -= 3) {
                            double pval = Float.intBitsToFloat(xtest[k][p + 1]);
                            int pfieldstart = xtest[k][p + 2] * sok;
                            int pstartIdx = pidx * stride;
                            for (int f = 0; f < sok; f++) {
                                gtest[gidx + pfieldstart + f] += gradscalar * assist[pstartIdx + qfieldstart + f] * pval * qval;
                            }
                            pidx --;
                        }
                        qidx ++;
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
                for (int i = secondOrderIndexStart; i < secondOrderIndexStart + stride; i++) {
                    gtest[i] = 0.0f;
                }
            }
        }

        return loss;

    }
}
