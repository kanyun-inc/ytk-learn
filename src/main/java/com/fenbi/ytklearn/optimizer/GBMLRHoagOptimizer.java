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

import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.ytklearn.dataflow.GBMLRDataFlow;
import com.fenbi.ytklearn.dataflow.ContinuousDataFlow;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

/**
 * @author xialong
 */

public class GBMLRHoagOptimizer extends HoagOptimizer {

    private final int K;

    private double wx[];

    private float[][] z;
    private float[][] ztest;
    private final BitSet[]randMask;
    private final BitSet featureMask;

    private final double learningRate;


    private final double randomSampleRate;
    private final double randomFeatureRate;

    private final double []samples;
    private final GBMLRDataFlow.Type type;
    private final GBMLRDataFlow gbmlrDataFlow;
    private final List<String> trainIterOtherInfos = new ArrayList<>();
    private final List<String> testIterOtherInfos = new ArrayList<>();

    public GBMLRHoagOptimizer(String modelName,
                              ContinuousDataFlow dataFlow,
                              int threadIdx) throws Exception {
        super(modelName, dataFlow, threadIdx);

        GBMLRDataFlow gbmlrDataFlow = (GBMLRDataFlow)dataFlow;
        this.gbmlrDataFlow = gbmlrDataFlow;

        this.K = gbmlrDataFlow.getK();
        this.wx = new double[2 * K - 1];
        this.samples = new double[K];
        this.z = ((GBMLRDataFlow.GBMLRCoreData)threadTrainCoreData).getZ();

        this.randMask = ((GBMLRDataFlow.GBMLRCoreData)threadTrainCoreData).getRandMask();
        this.featureMask = ((GBMLRDataFlow.GBMLRCoreData)threadTrainCoreData).getFeatureMask();
        this.randomSampleRate = gbmlrDataFlow.getRandomSampleRate();
        this.randomFeatureRate = gbmlrDataFlow.getRandomFeatureRate();
        this.learningRate = gbmlrDataFlow.getLearningRate();
        this.type = gbmlrDataFlow.getType();

        if (hasTestData) {
            this.ztest = ((GBMLRDataFlow.GBMLRCoreData)threadTestCoreData).getZ();
        }

//        l2[0] /= (2 * K - 1);
//        l1[0] /= (2 * K - 1);
    }


    @Override
    public int[] getRegularStart() {
        int []start = new int[1];
        start[0] = modelParams.need_bias ? 2 * K - 1 : 0;
        return start;
    }

    @Override
    public int[] getRegularEnd() {
        int []end = new int[1];
        end[0] = dim;
        return end;
    }

    @Override
    protected String extraInfo() {
        return "[round=" + (gbmlrDataFlow.getFinishedTreeNum() + 1) + "] ";
    }

    @Override
    protected void otherTrainHandle(int iter) throws Mp4jException {
        for (String info : trainIterOtherInfos) {
            importantInfo(iter, info);
        }
        trainIterOtherInfos.clear();

    };

    @Override
    protected void otherTestHandle(int iter) throws Mp4jException {
        for (String info : testIterOtherInfos) {
            importantInfo(iter, info);
        }
        testIterOtherInfos.clear();
    };

    @Override
    public double calcPureLossAndGrad(float[] w, float[] g, int iter) throws Exception {

        double rfLoss = 0.0;
        double loss = 0.0;
        int stride = 2 * K - 1;
        int vstride = K - 1;
        double compensate = 1.0 / randomSampleRate;
        for (int i = 0; i < dim; i++) {
            g[i] = 0.0f;
        }

        int treeNum = gbmlrDataFlow.getFinishedTreeNum() + 1;

        for (int i = 0; i < K; i++) {
            samples[i] = 0.0;
        }
        for (int k = 0; k < threadTrainCoreData.cursor2d; k++) {
            int lsNumInt = (int)realNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                if (!randMask[k].get(i)) {
                    continue;
                }
                double wei = weight[k][i];
                wei *= compensate;

                // xv, xw
                for (int p = 0; p < stride; p++) {
                    wx[p] = 0.0;
                }
                for (int j = xidx[k][i]; j < xidx[k][i + 1]; j+=2) {
                    double fval = Float.intBitsToFloat(x[k][j+1]);
                    int idx = x[k][j] * stride;
                    int pstart = featureMask.get(x[k][j]) ? 0 : vstride;
                    for (int p = pstart; p < stride; p++) {
                        wx[p] += w[idx + p] * fval;
                    }
                }
                // softmax
                double maxXv = wx[0];
                for (int j = 1; j < vstride; j++) {
                    if (wx[j] > maxXv) {
                        maxXv = wx[j];
                    }
                }
                double esum = 0.0;
                for (int j = 0; j < vstride; j++) {
                    wx[j] = Math.exp(wx[j] - maxXv);
                    esum += wx[j];
                }

                // g(x), fx
                double gk_1 = 1.0;
                double fx = type == GBMLRDataFlow.Type.RF ? 0 : z[k][i];
                // need't shrinkage
                double inv = 1.0 / (Math.exp(-maxXv) + esum);
                for (int j = 0; j < vstride; j++) {
                    wx[j] = wx[j] * inv;
                    samples[j] += wx[j];
                    gk_1 -= wx[j];
                    fx += wx[j] * wx[vstride + j];
                }
                fx += gk_1 * wx[stride - 1];
                samples[K - 1] += gk_1;

                loss += wei * lossFunction.loss(fx, y[k][i]);
                if (type == GBMLRDataFlow.Type.RF) {
                    rfLoss += wei * lossFunction.loss((z[k][i] + fx) / treeNum, y[k][i]);
                    predict[k][i] = (float) lossFunction.predict((z[k][i] + fx) / treeNum);
                } else {
                    predict[k][i] = (float) lossFunction.predict(fx);
                }

                // grad
                double purefx = fx - z[k][i];
                double gradscalar = wei * lossFunction.firstDerivative(fx, y[k][i]);
                for (int j = xidx[k][i]; j < xidx[k][i + 1]; j+=2) {
                    int idx = x[k][j] * stride;
                    double fval = Float.intBitsToFloat(x[k][j+1]);
                    if (featureMask.get(x[k][j])) {
                        // w, v' grad
                        for (int p = 0; p < K - 1; p++) {
                            g[idx + p] += gradscalar * wx[p] * (wx[p + vstride] - purefx) * fval;
                            g[idx + p + vstride] += gradscalar * wx[p] * fval;
                        }
                        g[idx + stride - 1] += gradscalar * gk_1 * fval;
                    } else {
                        for (int p = 0; p < K - 1; p++) {
                            g[idx + p + vstride] += gradscalar * wx[p] * fval;
                        }
                        g[idx + stride - 1] += gradscalar * gk_1 * fval;
                    }

                }
            }
        }

        if (type == GBMLRDataFlow.Type.RF) {
            rfLoss = comm.allreduce(rfLoss, Operands.DOUBLE_OPERAND(), Operators.Double.SUM);
            trainIterOtherInfos.add("train loss(random forest):" + rfLoss / gWeightTrainNum);
        }

        comm.allreduceArray(samples, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, samples.length);
        double sampleSum = 0.0;
        for (int i = 0; i < K; i++) {
            sampleSum += samples[i];
        }

        for (int i = 0; i < K; i++) {
            samples[i] /= sampleSum;
        }

        trainIterOtherInfos.add("all samples:" + sampleSum + ", ideal avg samples:" + sampleSum / (comm.getSlaveNum() * comm.getThreadNum()) + ", samples distribution:" + Arrays.toString(samples));
        return loss;
    }

    @Override
    public double calTestPureLossAndGrad(float []wtest, float []gtest, int iter, boolean needCalcGrad) throws Mp4jException {
        if (!hasTestData) {
            return -1.0;
        }

        double rfLoss = 0.0;
        double loss = 0.0;
        int stride = 2 * K - 1;
        int vstride = K - 1;
        if (needCalcGrad) {
            for (int i = 0; i < dim; i++) {
                gtest[i] = 0.0f;
            }
        }

        int treeNum = gbmlrDataFlow.getFinishedTreeNum() + 1;

        for (int k = 0; k < threadTestCoreData.cursor2d; k++) {
            int lsNumInt = (int)realNumtest[k];
            for (int i = 0; i < lsNumInt; i++) {
                double wei = weighttest[k][i];

                // xv, xw
                for (int p = 0; p < stride; p++) {
                    wx[p] = 0.0;
                }
                for (int j = xidxtest[k][i]; j < xidxtest[k][i + 1]; j+=2) {
                    double fval = Float.intBitsToFloat(xtest[k][j+1]);
                    int idx = xtest[k][j] * stride;
                    int pstart = featureMask.get(xtest[k][j]) ? 0 : vstride;
                    for (int p = pstart; p < stride; p++) {
                        wx[p] += wtest[idx + p] * fval;
                    }
                }
                // softmax
                double maxXv = wx[0];
                for (int j = 1; j < vstride; j++) {
                    if (wx[j] > maxXv) {
                        maxXv = wx[j];
                    }
                }
                double esum = 0.0;
                for (int j = 0; j < vstride; j++) {
                    wx[j] = Math.exp(wx[j] - maxXv);
                    esum += wx[j];
                }

                // g(x), fx
                double gk_1 = 1.0;
                double fx = type == GBMLRDataFlow.Type.RF ? 0 : ztest[k][i];
                // need't shrinkage
                double inv = 1.0 / (Math.exp(-maxXv) + esum);
                for (int j = 0; j < vstride; j++) {
                    wx[j] = wx[j] * inv;
                    gk_1 -= wx[j];
                    fx += wx[j] * wx[vstride + j];
                }
                fx += gk_1 * wx[stride - 1];

                loss += wei * lossFunction.loss(fx, ytest[k][i]);
                if (type == GBMLRDataFlow.Type.RF) {
                    rfLoss += wei * lossFunction.loss((ztest[k][i] + fx) / treeNum, ytest[k][i]);
                    predicttest[k][i] = (float) lossFunction.predict((ztest[k][i] + fx) / treeNum);
                } else {
                    predicttest[k][i] = (float) lossFunction.predict(fx);
                }

                if (needCalcGrad) {
                    // grad
                    double purefx = fx - ztest[k][i];
                    //double gradscalar = wei * (prob - ytest[k][i]);
                    double gradscalar = wei * lossFunction.firstDerivative(fx, ytest[k][i]);
                    for (int j = xidxtest[k][i]; j < xidxtest[k][i + 1]; j+=2) {
                        int idx = xtest[k][j] * stride;
                        double fval = Float.intBitsToFloat(xtest[k][j+1]);
                        if (featureMask.get(xtest[k][j])) {
                            // w, v' grad
                            for (int p = 0; p < K - 1; p++) {
                                gtest[idx + p] += gradscalar * wx[p] * (wx[p + vstride] - purefx) * fval;
                                gtest[idx + p + vstride] += gradscalar * wx[p] * fval;
                            }
                            gtest[idx + stride - 1] += gradscalar * gk_1 * fval;
                        } else {
                            for (int p = 0; p < K - 1; p++) {
                                gtest[idx + p + vstride] += gradscalar * wx[p] * fval;
                            }
                            gtest[idx + stride - 1] += gradscalar * gk_1 * fval;
                        }

                    }
                }

            }
        }

        if (type == GBMLRDataFlow.Type.RF) {
            rfLoss = comm.allreduce(rfLoss, Operands.DOUBLE_OPERAND(), Operators.Double.SUM);
            testIterOtherInfos.add("test loss(random forest):" + rfLoss / gWeightTestNum);
        }

        return loss;
    }
}
