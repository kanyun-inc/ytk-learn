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

public class SmoothHingeFunction implements ILossFunction {
    @Override
    public final double loss(double score, double label) {
        double z = (2 * label - 1.0) * score;
        if (z <= 0.0) {
            return 0.5 - z;
        } else if (z > 0 && z < 1.0) {
            return 0.5 * (1.0 - z) * (1.0 - z);
        } else {
            return 0.0;
        }
    }

    @Override
    public final double predict(double score) {
        return score;
    }

    @Override
    public final double firstDerivative(double score, double label) {
        double z = (2 * label - 1.0) * score;
        if (z <= 0.0) {
            return 1.0 - 2 * label;
        } else if (z > 0 && z < 1.0) {
            return (1.0 - 2 * label) * (1.0 - z);
        } else {
            return 0.0;
        }
    }

    @Override
    public final double secondDerivative(double score, double label) {
        double z = (2 * label - 1.0) * score;
        if (z <= 0.0 || z >= 1.0) {
            return 0.0;
        } else {
            return (2 * label - 1.0) * (2 * label - 1.0);
        }
    }

    @Override
    public String getName() {
        return "smooth_hinge";
    }

}
