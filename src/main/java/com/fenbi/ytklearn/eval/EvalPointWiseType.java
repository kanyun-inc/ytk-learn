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

package com.fenbi.ytklearn.eval;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * point wise evaluation
 * @author wufan
 * @author xialong
 */

public enum EvalPointWiseType {

    RMSE {
        @Override
        public double evalRow(float label, float pred) {
            return Math.pow(label - pred, 2);
        }

        @Override
        public String getEvalName() {
            return "rmse";
        }

        @Override
        public double getFinal(double sum, double weightSum) {
            return Math.sqrt(sum / weightSum);
        }
    },

    MAE {
        @Override
        public double evalRow(float label, float pred) {
            return Math.abs(label - pred);
        }

        @Override
        public String getEvalName() {
            return "mae";
        }

    },

    MAPE {
        @Override
        public double evalRow(float label, float pred) {
            return Math.abs((label - pred) / label);
        }

        @Override
        public String getEvalName() {
            return "mape";
        }
    },

    SMAPE {
        @Override
        public double evalRow(float label, float pred) {
            return Math.abs(pred - label) / ((label + Math.abs(pred)) / 2.0);
        }

        @Override
        public String getEvalName() {
            return "smape";
        }
    };

    public static final Logger LOG = LoggerFactory.getLogger(EvalPointWiseType.class);

    public abstract double evalRow(float label, float pred);

    public abstract String getEvalName();

    public double getFinal(double sum, double weightSum) {
        return sum / weightSum;
    }

}
