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
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.IDoubleOperator;
import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.feature.gbdt.approximate.normlization.*;
import com.fenbi.ytklearn.utils.NumConvertUtils;

import java.util.*;

/**
 * @author wufan
 * @author xialong
 */

public class SampleByPrecision implements ISampler {

    private int dotPrecision;
    private boolean useLog;
    private boolean useMinMax;

    private List<INormalization> normList;

    private ThreadCommSlave comm;

    public SampleByPrecision(ThreadCommSlave aggregate) {
        comm = aggregate;
    }


    @Override
    public void init(Map<String, String> params) {
        dotPrecision = Integer.parseInt(params.get("dot_precision"));
        useLog = Boolean.parseBoolean(params.get("use_log"));
        useMinMax = Boolean.parseBoolean(params.get("use_min_max"));
        normList = new ArrayList<>();
    }

    // find global min-max of a feature col
    private float[] getMinMax(GBDTCoreData data, int fid) throws Mp4jException {
        float minV = Float.MAX_VALUE;
        float maxV = Float.MIN_VALUE;
        for (int i = 0; i < data.sampleNum; i++) {
            float cur = NumConvertUtils.int2float(data.getFeatureVal(i, fid));
            if (cur > maxV) {
                maxV = cur;
            } else if (cur < minV) {
                minV = cur;
            }
        }
        if (comm != null) {

            double minMax = putFloatInDouble(minV, maxV);
            minMax = comm.allreduce(minMax, Operands.DOUBLE_OPERAND(), new IDoubleOperator() {
                @Override
                public double apply(double v1, double v2) {
                    float minV1 = getFloatValHigh(v1);
                    float maxV1 = getFloatValLow(v1);
                    float minV2 = getFloatValHigh(v2);
                    float maxV2 = getFloatValLow(v2);

                    float minV = minV1 < minV2 ? minV1 : minV2;
                    float maxV = maxV1 > maxV2 ? maxV1 : maxV2;
                    return putFloatInDouble(minV, maxV);
                }
            });
            minV = getFloatValHigh(minMax);
            maxV = getFloatValLow(minMax);
        }

        float[] minMax = new float[]{minV, maxV};
        data.LOG_UTILS.verboseInfo(String.format("SampleByPrecision, min-max norm minV: %f, maxV: %f", minV, maxV), false);
        return minMax;
    }

    private static double putFloatInDouble(float highV, float lowV) {
        long L = (((long)Float.floatToRawIntBits(highV)) << 32) & 0xffffffff00000000L;
        L = L | (0x00000000ffffffffL & Float.floatToRawIntBits(lowV));
        return Double.longBitsToDouble(L);
    }

    private static float getFloatValLow(double d) {
        return Float.intBitsToFloat((int) ((java.lang.Double.doubleToRawLongBits(d) & 0x00000000ffffffffL)));
    }

    private static float getFloatValHigh(double d) {
        return Float.intBitsToFloat((int) ((java.lang.Double.doubleToRawLongBits(d) & 0xffffffff00000000L) >> 32));
    }

    private void initNormlizer(GBDTCoreData data ,int fid) throws Mp4jException {
        if (useLog || useMinMax) {
            float[] minMax = getMinMax(data, fid);
            if (useLog) {
                INormalization posLogNorm = new PosLogNorm();
                posLogNorm.init(new float[]{minMax[0]});
                normList.add(posLogNorm);
                for (int i = 0; i < minMax.length; i++) {
                    minMax[i] = posLogNorm.normalization(minMax[i]);
                }
            }
            if (useMinMax) {
                INormalization minMaxNorm = new MinMaxNorm();
                minMaxNorm.init(minMax);
                normList.add(minMaxNorm);
            }
        }
        INormalization precisionNorm = new PrecisionNorm();
        precisionNorm.setParam("dot_precision", String.valueOf(dotPrecision));
        normList.add(precisionNorm);
    }

    private void normalization(GBDTCoreData data, int fid) {
        if (data == null) {
            return;
        }
        // get meta data
        for (int i = 0; i < data.sampleNum; i++) {
            float cur = NumConvertUtils.int2float(data.getFeatureVal(i, fid));
            for (INormalization norm : normList) {
                cur = norm.normalization(cur);
            }
            data.setFeatureVal(i, fid, NumConvertUtils.float2int(cur));
        }
    }

    @Override
    public Object doSample(GBDTCoreData data, int fid) throws Mp4jException {
        initNormlizer(data, fid);
        // data convert
        normalization(data, fid);

        Set<Float> feaVals = new HashSet<>();
        for (int i = 0; i < data.sampleNum; i++) {
            feaVals.add(NumConvertUtils.int2float(data.getFeatureVal(i, fid)));
        }
        data.LOG_UTILS.verboseInfo(String.format("SampleByPrecision(useLog=%b, useMinMax=%b, dotPrecision=%d), sample out %d values",
                useLog, useMinMax, dotPrecision, feaVals.size()), false);
        return feaVals;
    }

    @Override
    public float inverseTransform(float data) {
        for (int i = normList.size() - 1; i >= 0; i--) {
            data = normList.get(i).inverseTransform(data);
        }
        return data;
    }
}
