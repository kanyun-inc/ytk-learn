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

public class PoissonFunction implements ILossFunction {
    public final static double MAX_EXP = 30;
    public final static int yRange = 1000000;
    public final static double []logy = new double[yRange];
    static {
        logy[0] = 0.0;
        logy[1] = 0.0;
        for (int i = 2; i < yRange; i++) {
            logy[i] = logy[i - 1] + Math.log(i);
        }
    }

    private double logyfunc(int y) {
        if (y < yRange) {
            return logy[y];
        } else {
            double val = logy[logy.length - 1];
            for (int i = yRange; i < y; i++) {
                val += Math.log(i);
            }
            return val;
        }
    }
    @Override
    public final double loss(double score, double label) {
        return -label * score + Math.exp(Math.min(score, MAX_EXP)) + logyfunc((int)label);
    }

    @Override
    public final double predict(double score) {
        return Math.exp(Math.min(score, MAX_EXP));
    }

    @Override
    public final double pred2Score(double pred) {
        return Math.log(pred);
    }

    @Override
    public final double firstDerivative(double score, double label) {
        return Math.exp(Math.min(score, MAX_EXP)) - label;
    }

    @Override
    public final double secondDerivative(double score, double label) {
        return Math.exp(Math.min(score, MAX_EXP));
    }

    @Override
    public void getDerivativeFast(double pred, double label, double[] deri) {
        deri[0] = pred - label;
        deri[1] =  pred;

    }

    @Override
    public boolean checkLabel(float x) {
        return x >= 0;
    }

    @Override
    public String getName() {
        return "poisson";
    }

    public static void main(String []args) {
        PoissonFunction poissonFunction = new PoissonFunction();
        for (double i = 1.0; i < 10; i+=0.01) {
            System.out.println("score:" + i + ", e^score:" + Math.exp(Math.min(i, MAX_EXP)) + "-->" + poissonFunction.loss(i, Math.E));
        }
    }
}
