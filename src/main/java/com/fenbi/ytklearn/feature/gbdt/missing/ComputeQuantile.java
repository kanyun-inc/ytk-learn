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

package com.fenbi.ytklearn.feature.gbdt.missing;

import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.IObjectOperator;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.ytklearn.data.Constants;
import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.fenbi.ytklearn.utils.LogUtils;
import com.fenbi.ytklearn.utils.NumConvertUtils;
import com.fenbi.ytklearn.utils.WeightApproximateQuantile;

import java.util.HashMap;
import java.util.Map;

/**
 * compute quantile
 * @author wufan
 * @author xialong
 */
public class ComputeQuantile implements IComputer {
    private ThreadCommSlave comm;
    private LogUtils LOG_UTILS;
    private float quantile;

    public ComputeQuantile(ThreadCommSlave comm, String type) {
        this.comm = comm;
        this.LOG_UTILS = new LogUtils(comm, true);

        String[] fields = type.split(Constants.CONFIG_NAME_PARAM_SPLIT);
        this.quantile = 0.5f;
        CheckUtils.check(fields.length == 1 || fields.length == 2, "[GBDT] feature missing value process config error! type(%s), you can use mean, quantile or value@m", type);
        if (fields.length == 2) {
            this.quantile = Float.parseFloat(fields[1]);
            CheckUtils.check(quantile >= 0.f && quantile <= 1.f, "[GBDT] feature missing value process config error! quantile(%f) should belong to [0, 1]", quantile);
        }

    }


    public float[] compute(GBDTCoreData data) throws Exception {
        Map<String, WeightApproximateQuantile.Summary> mapSummary = new HashMap<>(data.usefulFeatureDim);
        for (int fid = 0; fid < data.usefulFeatureDim; fid++) {
            WeightApproximateQuantile.Summary summary = getWeightApproximateQuantile(fid, data);
            mapSummary.put(String.valueOf(fid), summary);
        }
        mapSummary = comm.allreduceMap(mapSummary, Operands.OBJECT_OPERAND(new WeightApproximateQuantile.SummarySerializer(), WeightApproximateQuantile.Summary.class), new IObjectOperator<WeightApproximateQuantile.Summary>() {
            @Override
            public WeightApproximateQuantile.Summary apply(WeightApproximateQuantile.Summary o1, WeightApproximateQuantile.Summary o2) {
                try {
                    return o1.merge(o2);
                } catch (Exception e) {
                    try {
                        comm.exception(e);
                    } catch (Mp4jException e1) {
                        e1.printStackTrace();
                    }
                    return null;
                }
            }
        });

        CheckUtils.check(mapSummary.size() == data.usefulFeatureDim,
                "GBDT: compute quantile error! featureMedianMap size=%d, feature dim=%d",
                mapSummary.size(), data.usefulFeatureDim);

        float[] missVfill = new float[data.usefulFeatureDim];
        for (Map.Entry<String, WeightApproximateQuantile.Summary> entry : mapSummary.entrySet()) {
            int fid = Integer.parseInt(entry.getKey());
            float val = 0;
            try {
                val = entry.getValue().query(quantile);
            } catch (Exception e) {
                if (comm != null) {
                    LOG_UTILS.importantInfo(String.format("feature(index:%d, name:%s) are all missing values or has no feature values!", fid, data.fIndex2NameMap.get(fid)));
                }
                val = 0;
            }
            missVfill[fid] = val;
        }
        return missVfill;
    }

    private WeightApproximateQuantile.Summary getWeightApproximateQuantile(int fid, GBDTCoreData data) throws Exception {
        Map<Float, Double> feaValCnt = new HashMap<>(Constants.RESERVE_NUM);
        for (int i = 0; i < data.sampleNum; i++) {
            int intval = data.getFeatureVal(i, fid);
            if (intval == Constants.INT_MISSING_VALUE) {
                continue;
            }
            float val = NumConvertUtils.int2float(intval);
            Double cnt = feaValCnt.get(val);
            if (cnt == null) {
                feaValCnt.put(val, (double) data.getSampleWeight(i));
            } else {
                feaValCnt.put(val, cnt + data.getSampleWeight(i));
            }
        }

        long globalValCnt = feaValCnt.size();
        if (comm != null) {
            globalValCnt = comm.allreduce(globalValCnt, Operands.LONG_OPERAND(), Operators.Long.SUM);
        }

        WeightApproximateQuantile quantile = new WeightApproximateQuantile(globalValCnt,
                Constants.QUNANTILE_APPROXIMATE_EPS, Constants.QUNANTILE_PRECISION_MAX_SAMPLE_CNT);
        for (Map.Entry<Float, Double> entry : feaValCnt.entrySet()) {
            quantile.update(entry.getKey(), entry.getValue().floatValue());
        }

        WeightApproximateQuantile.Summary summaryAll = quantile.mergeAllAndCompress();
        return summaryAll;

    }

}
