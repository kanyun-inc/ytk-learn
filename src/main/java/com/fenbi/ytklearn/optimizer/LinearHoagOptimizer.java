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

/**
 * @author xialong
 */

public class LinearHoagOptimizer extends HoagOptimizer {

    private final float[][] z;
    private final float[][] D;

    private final float[][] ztest;
    private final float[][] Dtest;

    public LinearHoagOptimizer(String modelName,
                               ContinuousDataFlow dataFlow,
                               int threadIdx) throws Exception {
        super(modelName, dataFlow, threadIdx);

        z = new float[threadTrainCoreData.cursor2d][];
        D = new float[threadTrainCoreData.cursor2d][];
        for (int i = 0; i < threadTrainCoreData.cursor2d; i++) {
            z[i] = new float[realNum[i]];
            D[i] = new float[realNum[i]];
        }

        if (hasTestData) {
            ztest = new float[threadTestCoreData.cursor2d][];
            Dtest = new float[threadTestCoreData.cursor2d][];
            for (int i = 0; i < threadTestCoreData.cursor2d; i++) {
                ztest[i] = new float[realNumtest[i]];
                Dtest[i] = new float[realNumtest[i]];
            }
        } else {
            ztest = null;
            Dtest = null;
        }

    }

//    @Override
//    public Object getEvalObjectInfo() {
//        if (lossFunction instanceof SigmoidFunction) {
//            return new ConfusionMatrixEvaluator.ConfusionMatrixInfo(2, true);
//        } else {
//            return null;
//        }
//    }

    private void Xv(float[] v, float[][] Xv, int []lsNum, int [][]xidx, int [][]x, int cursor2d) {
        for (int k = 0; k < cursor2d; k++) {
            int lsNumInt = lsNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                Xv[k][i] = 0;
                for (int j = xidx[k][i]; j < xidx[k][i + 1]; j+=2) {
                    Xv[k][i] += (v[x[k][j]] * Float.intBitsToFloat(x[k][j+1]));
                }
            }
        }

    }

    private void XTv(float[][] v, float[] XTv, int []lsNum, int [][]xidx, int [][]x, float [][]weight, int cursor2d) {

        for (int i = 0; i < dim; i++) {
            XTv[i] = 0;
        }

        for (int k = 0; k  < cursor2d; k++) {
            int lsNumInt = lsNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                double wei = weight[k][i];
                double gradscalar = wei * v[k][i];
                for (int j = xidx[k][i]; j < xidx[k][i + 1]; j+=2) {
                    XTv[x[k][j]] += (gradscalar * Float.intBitsToFloat(x[k][j+1]));
                }
            }
        }

    }

    @Override
    public int[] getRegularStart() {
        int []start = new int[1];
        if (modelParams.need_bias) {
            start[0] = 1;
        } else {
            start[0] = 0;
        }
        return start;
    }

    @Override
    public int[] getRegularEnd() {
        int []end = new int[1];
        end[0] = dim;
        return end;
    }

    @Override
    public double calcPureLossAndGrad(float[] w, float[] g, int iter) {
        double loss = 0.0;

        Xv(w, z, realNum, xidx, x, threadTrainCoreData.cursor2d);
        for (int k = 0; k < threadTrainCoreData.cursor2d; k++) {
            int lsNumInt = (int)realNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                double wei = weight[k][i];
                loss += wei * lossFunction.loss(z[k][i], y[k][i]);
                predict[k][i] = (float) lossFunction.predict(z[k][i]);
                D[k][i] = (float) lossFunction.secondDerivative(z[k][i], y[k][i]);
                z[k][i] = (float) lossFunction.firstDerivative(z[k][i], y[k][i]);

            }
        }

        // XT*z
        XTv(z, g, realNum, xidx, x, weight, threadTrainCoreData.cursor2d);

        return loss;
    }

    @Override
    public double calTestPureLossAndGrad(float []wtest, float []gtest, int iter, boolean needCalcGrad) {
        if (!hasTestData) {
            return -1.0;
        }

        double loss = 0.0;

        Xv(wtest, ztest, realNumtest, xidxtest, xtest, threadTestCoreData.cursor2d);
        for (int k = 0; k < threadTestCoreData.cursor2d; k++) {
            int lsNumInt = (int)realNumtest[k];
            for (int i = 0; i < lsNumInt; i++) {
                double wei = weighttest[k][i];
                loss += wei * lossFunction.loss(ztest[k][i], ytest[k][i]);
                predicttest[k][i] = (float) lossFunction.predict(ztest[k][i]);
                Dtest[k][i] = (float) lossFunction.secondDerivative(ztest[k][i], ytest[k][i]);
                ztest[k][i] = (float) lossFunction.firstDerivative(ztest[k][i], ytest[k][i]);

            }
        }

        if (needCalcGrad) {
            // XT*z
            XTv(ztest, gtest, realNumtest, xidxtest, xtest, weighttest, threadTestCoreData.cursor2d);
        }

        return loss;
    }

    @Override
    public void calPrecision() {
        for (int r = 0; r < l2.length; r++) {
            for (int i = regularStart[r]; i < regularEnd[r]; i++) {
                precision[i] = 0;
            }
        }


        for (int k = 0; k < threadTrainCoreData.cursor2d; k++) {
            for (int i = 0; i < realNum[k]; i++) {
                double wei = weight[k][i];
                // intercept不考虑
                for (int j = xidx[k][i]; j < xidx[k][i + 1] - 2; j+=2) {
                    double val = Float.intBitsToFloat(x[k][j+1]);
                    precision[x[k][j]] += wei * D[k][i] * (val * val);
                }
            }
        }

        for (int r = 0; r < l2.length; r++) {
            if (l2[r] > 0.0) {
                for (int i = regularStart[r]; i < regularEnd[r]; i++) {
                    precision[i] += tWeightTrainNum * l2[r];
                }
            }
        }

    }
}

