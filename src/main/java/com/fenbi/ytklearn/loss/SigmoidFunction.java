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

import com.fenbi.ytklearn.data.Constants;

import java.util.Map;

/**
 * @author xialong
 */

public final class SigmoidFunction implements ILossFunction {
    private double zMax = 0;
    private double negZMax = 0;

    @Override
    public final double loss(double score, double label) {
        if (score >= 0.0) {
            return Math.log(1 + Math.exp(-score)) + score * (1.0 - label);
        } else {
            return Math.log(1 + Math.exp(score)) - score * label;
        }
    }

    @Override
    // inverse sigmoid
    public double pred2Score(double pred) {
        return -Math.log(1.0 / pred - 1.0);
    }

    @Override
    public final double predict(double score) {
        if (score >= 0.0) {
            return 1.0 / (1.0 + Math.exp(-score));
        } else {
            double ez = Math.exp(score);
            return ez / (1.0 + ez);
        }
    }

    @Override
    public final double firstDerivative(double score, double label) {
        return predict(score) - label;
    }

    @Override
    public final double secondDerivative(double score, double label) {
        double p = predict(score);
        return p * (1.0 - p);
    }

    @Override
    public void getDerivativeFast(double pred, double label, double[] deri) {
        deri[0] = pred - label;  //grad
        deri[1] =  pred * (1.0 - pred); //hess
        if (zMax == 0) {
            return;
        }

        double z = 0;
        if (deri[1] != 0) {
            z = -(deri[0] / deri[1]);
        }

        //avoid |g/h| too large
        if (z > zMax) {
            deri[1] = -(deri[0] / zMax);
        } else if (z < negZMax) {
            deri[1] = -(deri[0] / negZMax);
        }
    }

    @Override
    public boolean checkLabel(float x) {
        return (x >= 0.0) && (x <= 1.0);
    }

    @Override
    public String getName() {
        return "sigmoid";
    }

    @Override
    public void setParam(Map<String, String> params) {
        Float val = Float.parseFloat(params.get("sigmoid_zmax"));
        if (val != null) {
            zMax = val;
            negZMax = -val;
        }
    }
}
