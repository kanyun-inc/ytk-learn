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

package com.fenbi.ytklearn.feature.gbdt.approximate.sampler;

import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.exception.YtkLearnException;

/**
 * @author wufan
 * @author xialong
 */

public class SamplerFactory {

    // comm can be null which means it's local mode
    public static ISampler create(SampleType type, ThreadCommSlave comm) {
        if (type.equals(SampleType.CNT)) {
            return new SampleByCnt(comm);
        } else if (type.equals(SampleType.RATE)) {
            return new SampleByRate(comm);
        } else if (type.equals(SampleType.PRECISION)) {
            return new SampleByPrecision(comm);
        } else if (type.equals(SampleType.NO_SAMPLE)) {
            return new NoSample(comm);
        } else if (type.equals(SampleType.QUANTILE)) {
            return new SampleByQuantile(comm);
        } else {
            throw new YtkLearnException("unknown sampler type: " + type.getName());
        }
    }
}
