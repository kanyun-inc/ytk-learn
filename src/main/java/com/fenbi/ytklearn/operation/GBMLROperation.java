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

package com.fenbi.ytklearn.operation;

import com.fenbi.ytklearn.dataflow.DataFlow;
import com.fenbi.ytklearn.dataflow.GBMLRDataFlow;
import com.fenbi.ytklearn.optimizer.IOptimizer;
import com.fenbi.ytklearn.optimizer.HoagOptimizer;
import com.fenbi.ytklearn.param.CommonParams;
import com.fenbi.mp4j.comm.ThreadCommSlave;

/**
 * @author xialong
 */

public class GBMLROperation implements ITrainOperation {
    @Override
    public void operate(DataFlow dataFlow, IOptimizer optimizer, ThreadCommSlave comm, int threadIdx) throws Exception {
        ((HoagOptimizer)optimizer).init();
        GBMLRDataFlow gbmlrDataFlow = (GBMLRDataFlow)dataFlow;
        HoagOptimizer hoagOptimizer = (HoagOptimizer) optimizer;
        int finishedNum = gbmlrDataFlow.getFinishedTreeNum();
        int tree = finishedNum;
        double prevLoss = Double.MAX_VALUE;
        int rank = comm.getRank();
        int K = gbmlrDataFlow.getK();

        CommonParams commonParams = gbmlrDataFlow.getCommonParams();
        boolean justEval = commonParams.lossParams.just_evaluate;

        if (tree >= gbmlrDataFlow.getTreeNum() && !justEval) {
            comm.info("finished tree num:" + finishedNum + " >= tree num:" + gbmlrDataFlow.getTreeNum() +
                            ", train finished!");
            return;
        }

        while(true) {

            if (threadIdx == 0) {
                comm.info("finished tree num:" + finishedNum + ", now constructing treeid:" + tree);
            }
            // optimizer
            double loss = hoagOptimizer.lbfgs(false);

            if (justEval) {
                comm.info("just evalate, return!");
                return;
            }

            if (threadIdx == 0) {
                comm.info("gradient boost cur loss:" + loss + ", prev loss:" + prevLoss + ", will construct next tree!");
            }

            comm.info("accumulate tree:" + tree + "...");
            GBMLRDataFlow.GBMLRCoreData trainCoreData = (GBMLRDataFlow.GBMLRCoreData)gbmlrDataFlow
                    .getThreadTrainCoreDatas()[threadIdx];
            // accumulate train data
            if (threadIdx == 0) {
                float [][]trainz = trainCoreData.getZ();
                comm.info(getSomeZInfo(trainz, "rank:" + rank + ", before accumulate, train z:"));
            }
            gbmlrDataFlow.accumulate(trainCoreData, trainCoreData.getFeatureMask(), true, gbmlrDataFlow.getW()[threadIdx], gbmlrDataFlow.getLearningRate(), K);
            if (threadIdx == 0) {
                float [][]trainz = trainCoreData.getZ();
                comm.info(getSomeZInfo(trainz, "rank:" + rank + ", after accumulate, train z:"));
            }

            // accumulate test data
            if (gbmlrDataFlow.isNeedTest()) {
                GBMLRDataFlow.GBMLRCoreData testCoreData = (GBMLRDataFlow.GBMLRCoreData)gbmlrDataFlow
                        .getThreadTestCoreDatas()[threadIdx];
                gbmlrDataFlow.accumulate(testCoreData, trainCoreData.getFeatureMask(), true, gbmlrDataFlow.getW()[threadIdx], gbmlrDataFlow.getLearningRate(), K);
            }

            // dump model info and init next tree
            if (threadIdx == 0) {
                comm.info("constructing treeid:" + tree + " finished!");
                gbmlrDataFlow.incrFinishedTreeNum();
                gbmlrDataFlow.dumpModelInfo(rank);
            }

            tree++;
            if (tree >= gbmlrDataFlow.getTreeNum()) {
                break;
            }

            prevLoss = loss;
            comm.threadBarrier();
            gbmlrDataFlow.initW(threadIdx);
            gbmlrDataFlow.randomNextSample(trainCoreData, gbmlrDataFlow.getRandomSampleRate(), gbmlrDataFlow.getRandomFeatureRate());
            comm.threadBarrier();

        }
    }

    private String getSomeZInfo(float [][]z, String prefix) {
        StringBuffer sb = new StringBuffer(prefix);
        for (int i = 0; i < Math.min(10, z[0].length); i++) {
            sb.append("z[").append(i).append("]=").append(z[0][i]).append(",");
        }
        return sb.toString();
    }
}
