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

import com.fenbi.ytklearn.dataflow.GBDTCoreData;

import java.util.Map;
import java.util.Set;

/**
 * interface of feature approximation
 * @author wufan
 * @author xialong
 */

public interface ISampler {

    default public void init(Map<String, String> params){}

    // sample_by_cnt, sample_by_rate, sample_by_precision, no_sample: return Set<Float>
    // sample_by_quantile: return WeightApproximateQuantile.Summary
    public Object doSample(GBDTCoreData data, int fid) throws Exception;

    // called by sample_by_quantile
    default public Set<Float> getSamples(Object o) {return null; }

    // revert value to original value scale, only need to be implemented in SampleByPrecision
    default public float inverseTransform(float data) {return data; }

}
