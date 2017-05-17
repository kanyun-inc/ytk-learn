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

import com.fenbi.ytklearn.exception.YtkLearnException;
import com.fenbi.ytklearn.utils.CheckUtils;

/**
 * @author wufan
 * @author xialong
 */

public class MinMaxNorm implements INormalization {

    private float minV;
    private float maxV;
    private float interval;

    private boolean initialized;

    public MinMaxNorm() {
        minV = Float.MAX_VALUE;
        maxV = Float.MIN_VALUE;
        initialized = false;
    }

    @Override
    public void init(float[] info) {
        CheckUtils.check(info != null && info.length == 2 && info[0] <= info[1], "[GBDT] inner error, min-max norm init param error!");
        minV = info[0];
        maxV = info[1];
        interval = maxV - minV;
        initialized = true;
    }

    private boolean initialized() {
        return initialized;
    }

    @Override
    public float normalization(float origin) throws YtkLearnException {
        CheckUtils.check(initialized(), "[GBDT] inner error, min-max normalization has not be initialized!");
        if (interval < 1E-8f) {
            return minV;
        }
        if (origin <= minV) {
            return 0.f;
        } else if (origin > maxV) {
            return 1.f;
        }
        return (origin - minV) / interval;
    }

    // has precision loss
    public float inverseTransform(float origin) throws YtkLearnException {
        CheckUtils.check(initialized(), "[GBDT]inner error, min-max normalization has not be initialized!");
        return origin * interval + minV;
    }
}
