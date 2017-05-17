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

package com.fenbi.ytklearn.feature.gbdt.approximate.normlization;

import com.fenbi.ytklearn.utils.CheckUtils;

/**
 * value is rounded to dot_precision decimal places
 * @author wufan
 * @author xialong
 */

public class PrecisionNorm implements INormalization {

    private int dotPrecision;
    private long baseNum;
    private boolean initialized;

    public PrecisionNorm() {
        initialized = false;
    }

    @Override
    public void setParam(String key, String val) {
        if (key.equals("dot_precision")) {
            dotPrecision = Integer.parseInt(val);
            baseNum = (long)Math.pow(10, dotPrecision);
            initialized = true;
        }
    }

    private boolean initialized() {
        return initialized;
    }

    @Override
    public float normalization(float data) {
        CheckUtils.check(initialized(), "[GBDT] inner error, precision norm not initialized!");
        return (float)(((long) (data * baseNum)) * 1.0 / baseNum);
    }

    @Override
    public float inverseTransform(float data) {
         return data;
    }

}



