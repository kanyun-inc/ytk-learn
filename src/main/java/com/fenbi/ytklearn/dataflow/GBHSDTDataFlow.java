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

package com.fenbi.ytklearn.dataflow;

import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.fs.IFileSystem;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.fenbi.ytklearn.utils.MathUtils;
import com.typesafe.config.Config;
import lombok.Data;

import java.util.*;

/**
 * @author xialong
 */

@Data
public class GBHSDTDataFlow extends GBSDTDataFlow {
    private final int L;
    public GBHSDTDataFlow(IFileSystem fs,
                          Config config,
                          ThreadCommSlave comm,
                          int threadNum,
                          boolean needPyTransform,
                          String pyTransformScript) throws Exception {
        super(fs, config, comm, threadNum, needPyTransform, pyTransformScript);
        int l = -1;
        for (int i = 1; i <= 31; i++) {
            if ((1 << i) >= K) {
                l = i;
                break;
            }
        }
        CheckUtils.check(l >= 1, "K must be >= 2, K:" + K);
        this.L = l;
        LOG_UTILS.importantInfo("K:" + K + ", L:" + L);
    }

    @Override
    public void accumulate(GBMLRCoreData coreData, BitSet featureMask, boolean current, float []w, double learningRate, int K) {

        int stride = K - 1;
        int vstart = K;
        double wx[] = new double[2 * K - 1];
        double mu[] = new double[2 * K - 1];
        for (int k = 0; k < coreData.cursor2d; k++) {
            int lsNumInt = coreData.realNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                // xv, xw
                for (int p = 0; p < stride; p++) {
                    wx[p] = 0.0;
                }
                for (int j = coreData.xidx[k][i]; j < coreData.xidx[k][i + 1]; j+=2) {
                    if (current && !featureMask.get(coreData.x[k][j])) {
                        continue;
                    }
                    double fval = Float.intBitsToFloat(coreData.x[k][j+1]);
                    int idx = vstart + coreData.x[k][j] * stride;
                    for (int p = 0; p < stride; p++) {
                        wx[p] += w[idx + p] * fval;
                    }
                }

                for (int j = 0; j < stride; j++) {
                    wx[j] = MathUtils.logistic(wx[j]);
                }

                for (int j = stride; j < 2 * K - 1; j++) {
                    wx[j] = w[j - stride];
                }

                // calc mu
                for (int j = stride; j < 2 * K - 1; j++) {
                    mu[j] = wx[j];
                    int prevIdx = j + 1;
                    int curIdx;
                    for (int p = 0; p < L; p++) {
                        curIdx = prevIdx >>> 1;
                        mu[j] *= ((prevIdx & 1) == 0 ? wx[curIdx - 1] : 1.0 - wx[curIdx - 1]);
                        prevIdx = curIdx;
                        if (curIdx == 1) {
                            break;
                        }
                    }
                }

                for (int j = stride - 1; j >= 0; j--) {
                    int idx = ((j + 1) << 1) - 1;
                    mu[j] = mu[idx] + mu[idx + 1];
                }

                coreData.z[k][i] += learningRate * mu[0];
            }
        }
    }
}
