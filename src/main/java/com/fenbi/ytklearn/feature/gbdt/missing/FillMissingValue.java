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
import com.fenbi.ytklearn.data.Constants;
import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.exception.YtkLearnException;
import com.fenbi.ytklearn.param.gbdt.GBDTFeatureParams;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.fenbi.ytklearn.utils.NumConvertUtils;

/**
 * @author wufan
 * @author xialong
 */

public class FillMissingValue {
    private ThreadCommSlave comm;
    private final boolean isLocal;

    public FillMissingValue(ThreadCommSlave comm, boolean isLocal) {
        this.comm = comm;
        this.isLocal = isLocal;
    }

    // compute missing value according to conf
    public float[] computeMissingValueFill(GBDTFeatureParams featParams, GBDTCoreData data) throws Exception {
        String type = featParams.featureMissingParams;
        IComputer computer = ComputeFactory.create(type, comm);
        if (computer == null) {
            throw new YtkLearnException("process missing value config error!" + type);
        }
        // compute missing value for each feature
        float[] missV = computer.compute(data);
        return missV;
    }

    // used in local & distributed version
    public void fillMissingValue(float[] missingValueFill, GBDTCoreData data) {
        CheckUtils.check(missingValueFill == null || missingValueFill.length == 0 || missingValueFill.length == data.usefulFeatureDim,
                String.format("GBDT: feature dim inconsistent, useful_feature_dim=%d, miss_value_fill length=%d",
                        data.usefulFeatureDim, missingValueFill.length));
        if (missingValueFill == null || missingValueFill.length == 0) {
            return;
        }

        int[] intMissingValueFill = new int[missingValueFill.length];
        for (int i = 0; i < missingValueFill.length; i++) {
            intMissingValueFill[i] = NumConvertUtils.float2int(missingValueFill[i]);
        }

        for (int i = 0; i < data.sampleNum; i++) {
            for (int fid = 0; fid < data.usefulFeatureDim; fid++) {
                data.isEqualReplaceFeaVal(i, fid, Constants.INT_MISSING_VALUE, intMissingValueFill[fid]);
            }
        }

        // local & train data
        if (isLocal && data.xT != null) {
                int startXCol = data.xColRange[0];
                float[][] feaInstByCol = data.xT.feaInstByCol;
                for (int fid = 0; fid < feaInstByCol.length; fid++) {
                    for (int sid = 0; sid < feaInstByCol[fid].length; sid += 2) {
                        if (Float.isNaN(feaInstByCol[fid][sid])) {
                            feaInstByCol[fid][sid] = missingValueFill[startXCol + fid];
                        }
                    }
                }
            }
        }
}
