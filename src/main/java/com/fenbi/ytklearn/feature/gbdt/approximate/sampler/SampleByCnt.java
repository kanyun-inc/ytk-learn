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

import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.fenbi.ytklearn.utils.NumConvertUtils;
import com.fenbi.ytklearn.utils.RandomUtils;

import java.util.*;

/**
 * sample by cnt in each thread of each slave
 * @author wufan
 * @author xialong
 */

public class SampleByCnt implements ISampler {

    private final ThreadCommSlave comm;

    private int maxFeatureCnt;

    public SampleByCnt(ThreadCommSlave aggregate) {
        this.comm = aggregate;
    }

    @Override
    public void init(Map<String, String> params) {
        maxFeatureCnt = Integer.parseInt(params.get("max_cnt"));
    }

    @Override
    public Object doSample(GBDTCoreData data, int fid) throws Mp4jException {
        Set<Float> feaVals = new HashSet<>();

        boolean needSample = false;
        for (int i = 0; i < data.sampleNum; i++) {
            feaVals.add(NumConvertUtils.int2float(data.getFeatureVal(i, fid)));
            if (feaVals.size() > maxFeatureCnt) {
                needSample = true;
                break;
            }
        }

        if (needSample) {
            int[] sampleOut = new int[maxFeatureCnt];
            CheckUtils.check(RandomUtils.reservoidSample(data.sampleNum, maxFeatureCnt, sampleOut),
                    "[GBDT] inner error, reservoir sample");

            feaVals.clear();
            for (int sid: sampleOut) {
                feaVals.add(NumConvertUtils.int2float(data.getFeatureVal(sid, fid)));
            }
        }
        data.LOG_UTILS.verboseInfo(String.format("samplebyCnt(max_cnt=%d) sample out %d values", maxFeatureCnt, feaVals.size()), false);
        return feaVals;
    }

}
