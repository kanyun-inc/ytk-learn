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
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.ytklearn.data.Constants;
import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.utils.NumConvertUtils;
import com.fenbi.ytklearn.utils.WeightApproximateQuantile;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * @author wufan
 * @author xialong
 */

public class SampleByQuantile implements ISampler {

    private ThreadCommSlave comm;

    private int maxBinCnt;
    private long globalCnt;
    private int quantileApprBinFactor;
    private boolean useSampleWeight;
    private double alpha;

    public SampleByQuantile(ThreadCommSlave aggregate) {
        this.comm = aggregate;
    }

    @Override
    public void init(Map<String, String> params) {
        maxBinCnt = Integer.parseInt(params.get("max_cnt"));
        globalCnt = Long.parseLong(params.get("global_cnt"));
        quantileApprBinFactor = Integer.parseInt(params.get("quantile_approximate_bin_factor"));
        useSampleWeight = Boolean.parseBoolean(params.get("use_sample_weight"));
        alpha = Double.parseDouble(params.get("alpha"));
    }

    public Object doSample(GBDTCoreData data, int fid) throws Exception {
        Map<Float, Double> feaValCnt = new HashMap<>(Constants.RESERVE_NUM);
        if (useSampleWeight) {
            for (int i = 0; i < data.sampleNum; i++) {
                float val = NumConvertUtils.int2float(data.getFeatureVal(i, fid));
                Double cnt = feaValCnt.get(val);
                if (cnt == null) {
                    feaValCnt.put(val, (double)data.getSampleWeight(i));
                } else {
                    feaValCnt.put(val, cnt + data.getSampleWeight(i));
                }
            }
        } else {
            for (int i = 0; i < data.sampleNum; i++) {
                float val = NumConvertUtils.int2float(data.getFeatureVal(i, fid));
                Double cnt = feaValCnt.get(val);
                if (cnt == null) {
                    feaValCnt.put(val, 1.);
                } else {
                    feaValCnt.put(val, cnt + 1);
                }
            }
        }

        long globalValCnt = feaValCnt.size();
        if (comm != null) {
            globalValCnt = comm.allreduce(globalValCnt, Operands.LONG_OPERAND(), Operators.Long.SUM);
        }
        //directly compute
        if (globalValCnt <= maxBinCnt) {
            Set<Float> feaSet = new HashSet<>(feaValCnt.size());
            feaSet.addAll(feaValCnt.keySet());
            return feaSet;

        } else {
            double eps = 1. / (quantileApprBinFactor * maxBinCnt);
            WeightApproximateQuantile quantile = new WeightApproximateQuantile(globalValCnt, eps, Constants.QUNANTILE_PRECISION_MAX_SAMPLE_CNT);
            for (Map.Entry<Float, Double> entry: feaValCnt.entrySet()) {
                quantile.update(entry.getKey(), (float)Math.pow(entry.getValue(), alpha));
            }

            WeightApproximateQuantile.Summary summaryAll = quantile.mergeAllAndCompress();
            return summaryAll;
        }
    }

    @Override
    public Set<Float> getSamples(Object o) {
        Set<Float> vals = new HashSet<>(maxBinCnt);
        WeightApproximateQuantile.Summary mergeSummary = (WeightApproximateQuantile.Summary)o;
        for (int i = 1; i <= maxBinCnt; i++) {
            vals.add(mergeSummary.query(i * 1.0 / maxBinCnt));
        }
        return vals;
    }

}
