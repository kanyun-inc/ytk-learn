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

package com.fenbi.ytklearn.data.gbdt;

import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.ytklearn.data.Tuple;
import com.fenbi.ytklearn.dataflow.GBDTCoreData;

/**
 * @author wufan
 * @author xialong
 */

public class HistogramBuilder {

    private final ThreadCommSlave comm;
    private final int rank;
    private final int threadId;
    private final int[] feaSplitValStartIndex;
    private final int[][] globalHistAssignFrom;
    private final int[][] globalHistAssignCount;

    public HistogramBuilder(ThreadCommSlave comm, FeatureApprData featureAprrData) {
        this.comm = comm;
        this.rank = comm.getRank();
        this.threadId = comm.getThreadId();
        this.feaSplitValStartIndex = featureAprrData.getFeaSplitValStartIndex();
        globalHistAssignFrom = featureAprrData.getGlobalHistAssignFrom();
        globalHistAssignCount = featureAprrData.getGlobalHistAssignCount();
    }

    public double[] buildHist(Tuple<Integer, Integer> posInterval,
                              int[] positionAndIndex,
                              GBDTCoreData trainData,
                              int[] feaIndex,
                              int groupId,
                              double[] localGradSum,
                              long[] cost) throws Mp4jException {
        int nidOffset;
        int sampleIdx, feaSlotIdx;
        int gradIdx, gradSumIdx;
        int index2D, index1D;
        int[][] features = trainData.getX();
        long start = System.currentTimeMillis();
        for (int i = 0; i < localGradSum.length; i++) {
            localGradSum[i] = 0;
        }
        for (int i = posInterval.v1; i < posInterval.v2; i++) {
            nidOffset = i << 1;
            // not sampled
            if (positionAndIndex[nidOffset] < 0)
                continue;
            sampleIdx = positionAndIndex[nidOffset + 1];
            index2D = sampleIdx / trainData.DENSE_MAX_1D_SAMPLE_CNT;
            index1D = sampleIdx % trainData.DENSE_MAX_1D_SAMPLE_CNT;

            // sampled features
            for (int fid : feaIndex) {
                feaSlotIdx = features[index2D][index1D * trainData.maxFeatureDim + fid];
                gradIdx = (index1D * trainData.numTreeInGroup + groupId) << 1;
                gradSumIdx = (feaSplitValStartIndex[fid] + feaSlotIdx) << 1;

                localGradSum[gradSumIdx] += trainData.gradPairs[index2D][gradIdx];
                localGradSum[gradSumIdx + 1] += trainData.gradPairs[index2D][gradIdx + 1];
            }
        }

        cost[0] += System.currentTimeMillis() - start;
        start = System.currentTimeMillis();

        localGradSum = comm.reduceScatterArray(localGradSum, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, globalHistAssignCount);
        cost[1] +=  System.currentTimeMillis() - start;
        return localGradSum;
    }


    public double[] substract(double[] parentHist, double[] siblingHist, double[] selfHist) throws Mp4jException {
        // len(selfHist)= len(globalGradSum), both len(parentHist) and len(siblingHist) = len(threadProcessGradSum)
        for (int i = 0; i < parentHist.length; i++) {
            selfHist[globalHistAssignFrom[rank][threadId] + i] = parentHist[i] - siblingHist[i];
        }
        return selfHist;
    }
}
