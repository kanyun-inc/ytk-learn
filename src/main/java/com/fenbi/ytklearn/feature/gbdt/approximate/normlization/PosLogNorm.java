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
 * convert value to value + min(all values), then log(1+value)
 * @author wufan
 * @author xialong
 */

public class PosLogNorm implements INormalization {

    private float minV;
    private boolean initialized;

    public PosLogNorm() {
        minV = 0.0f;
        initialized = false;
    }

    private boolean initialized() {
        return initialized;
    }

    @Override
    public void init(float[] info) {
        CheckUtils.check(info!= null && info.length == 1, "pos log(1+x) norm init param error!");
        minV = Math.min(info[0], 0.f);
        initialized = true;
    }

    @Override
    public float normalization(float origin) {
        CheckUtils.check(initialized(), "pos log(1+x) not initialized!");
        return (float) Math.log(1 + origin - minV);
    }

    @Override
    public float inverseTransform(float origin) {
        CheckUtils.check(initialized(), "pos log(1+x) not initialized!");
        return (float) (Math.exp(origin) + minV - 1);
    }
}
