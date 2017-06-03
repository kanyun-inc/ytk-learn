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

package com.fenbi.ytklearn.optimizer.gbdt;

import com.fenbi.ytklearn.data.gbdt.GradStats;
import com.fenbi.ytklearn.param.gbdt.GBDTOptimizationParams;

/**
 * @author wufan
 * @author xialong
 */

public class UpdateStrategy {

    private final double minChildWeight;
    private final long minSplitSamples;
    private final double maxAbsLeafVal;
    private final double regL1;
    private final double regL2;

    public UpdateStrategy(GBDTOptimizationParams params) {
        minChildWeight = params.min_child_hessian_sum;
        minSplitSamples = params.min_split_samples;
        maxAbsLeafVal = params.max_abs_leaf_val;
        regL1 = params.regularization.l1;
        regL2 = params.regularization.l2;
    }

    public boolean canSplit(double hessSum, long sampleCnt) {
        return hessSum >= minChildWeight * 2.0
                && (sampleCnt >= minSplitSamples);
    }

    public double calcGain(GradStats stats) {
        return calcGain(stats.getSumGrad(), stats.getSumHess());
    }

    public double calcNodeValue(GradStats stats) {
        return calcNodeValue(stats.getSumGrad(), stats.getSumHess());
    }

    // calculate the cost of loss function
    private double calcGain(double sumGrad, double sumHess) {
        if (sumHess < minChildWeight)
            return 0.0;
        double gain;
        // no limit on leaf value
        if (maxAbsLeafVal <= 0) {
            if (regL1 == 0.0f) {
                gain = Math.pow(sumGrad, 2) / (sumHess + regL2);
            } else {
                gain = Math.pow(thresholdL1(sumGrad, regL1), 2) / (sumHess + regL2);
            }
        } else {
            double leafVal = calcNodeValue(sumGrad, sumHess);
            gain = -2 * (sumGrad * leafVal + 0.5 * (sumHess + regL2) * Math.pow(leafVal, 2) + regL1 * Math.abs(leafVal));
        }
        return gain;
    }

    // calculate weight given the statistics, weight: leaf node predction
    private double calcNodeValue(double sumGrad, double sumHess) {
        if (sumHess < minChildWeight)
            return 0.0;
        double val;
        if (regL1 == 0.0f) {
            val = -sumGrad / (sumHess + regL2);
        } else {
            val = -thresholdL1(sumGrad, regL1) / (sumHess + regL2);
        }
        if (maxAbsLeafVal > 0) {
            if (val > maxAbsLeafVal) {
                val = maxAbsLeafVal;
            } else if (val < -maxAbsLeafVal) {
                val = -maxAbsLeafVal;
            }
        }
        return val;
    }

    // functions for L1 cost, w is gradient， IF w < 0, then weight >0
    private static double thresholdL1(double w, double lambda) {
        if (w > lambda)
            return w - lambda;
        if (w < -lambda)
            return w + lambda;
        return 0.0f;
    }

}
