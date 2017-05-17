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
import com.fenbi.ytklearn.utils.CheckUtils;

/**
 * @author wufan
 * @author xialong
 */

public class ComputeValue implements IComputer {
    private float value;

    public ComputeValue(ThreadCommSlave comm, String type) {
        String[] fields = type.split(Constants.CONFIG_NAME_PARAM_SPLIT);
        value = 0f;
        CheckUtils.check(fields.length == 1 || fields.length == 2, "feature missing value process config error! type(%s), you can use mean, quantile or value@m", type);
        if (fields.length == 2) {
            value = Float.parseFloat(fields[1]);
        }
    }

    @Override
    public float[] compute(GBDTCoreData data) throws Exception {
        float[] missVfill = new float[data.usefulFeatureDim];
        for (int fid = 0; fid < data.usefulFeatureDim; fid++) {
            missVfill[fid] = value;
        }
        return missVfill;
    }
}
