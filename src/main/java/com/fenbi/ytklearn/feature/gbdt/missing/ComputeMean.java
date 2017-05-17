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
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.ytklearn.data.Constants;
import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.utils.LogUtils;
import com.fenbi.ytklearn.utils.NumConvertUtils;

/**
 * compute mean
 * @author wufan
 * @author xialong
 */

public class ComputeMean implements IComputer {
    private ThreadCommSlave comm;
    private LogUtils LOG_UTILS;

    public ComputeMean(ThreadCommSlave comm) {
        this.comm = comm;
        this.LOG_UTILS = new LogUtils(comm, true);
    }

    @Override
    public float[] compute(GBDTCoreData data) throws Mp4jException {
        double[] sumCnt = new double[data.usefulFeatureDim << 1];
        for (int i = 0; i < sumCnt.length; i++) {
            sumCnt[i] = 0;
        }

        for (int i = 0; i < data.sampleNum; i++) {
            for (int fid = 0; fid < data.usefulFeatureDim; fid++) {
                int intval = data.getFeatureVal(i, fid);
                if (intval == Constants.INT_MISSING_VALUE) {
                    continue;
                }
                float val = NumConvertUtils.int2float(intval);
                float wei = data.getSampleWeight(i);
                sumCnt[fid << 1] += val * wei;
                sumCnt[(fid << 1) + 1] += wei;
            }
        }

        if (comm != null) {
            sumCnt = comm.allreduceArray(sumCnt, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, sumCnt.length);
        }

        float[] missVfill = new float[data.usefulFeatureDim];
        for (int fid = 0; fid < data.usefulFeatureDim; fid++) {
            int j = (fid << 1);
            // if all feature val is missing or this feature has no sample, then fill with 0
            if (sumCnt[j + 1] == 0.0) {
                missVfill[fid] = 0;
                if (comm != null) {
                    LOG_UTILS.importantInfo(String.format("feature(index:%d, name:%s) are all missing values or has no feature values!", fid, data.fIndex2NameMap.get(fid)));
                }
            } else {
                missVfill[fid] = (float) (sumCnt[j] / sumCnt[j + 1]);
            }
        }
        return missVfill;
    }
}
