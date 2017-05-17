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

import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.param.gbdt.GBDTFeatureParams;
import com.fenbi.ytklearn.feature.gbdt.approximate.SampleManager;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.utils.BinarySearch;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.fenbi.ytklearn.utils.NumConvertUtils;
import com.fenbi.ytklearn.utils.StringUtils;

import java.util.*;

/**
 * info of features need to sync with other worker
 * process features approximate and convert in train set
 * the class will change feature values in train set
 * @author wufan
 * @author xialong
 */

public class FeatureApprData {

    private static final int MAX_INFO_LEN = 10000000;

    private final ThreadCommSlave comm;
    private final boolean owner;
    private GBDTCoreData data;
    private final boolean verbose;

    // sorted feature values, used to find value by slot index
    private Map<Integer, float[]> globalFeaSplitValsSorted;

    // global start index of each feature in 1D array, the last element is the total number of approximate feature values.
    private int[] feaSplitValStartIndex;
    private int[][] globalHistAssignFrom;
    private int[][] globalHistAssignCount;

    private int numBin;

    public FeatureApprData(ThreadCommSlave comm,
                           GBDTFeatureParams featParam,
                           GBDTCoreData trainData) throws Exception {
        this.comm = comm;
        this.owner = comm.getRank() == 0 && comm.getThreadId() == 0;
        this.data = trainData;
        this.verbose = featParam.verbose;
        genGlobalFeaBinAndConvert(featParam);
        updateGlobalHistAssign();
        this.numBin = feaSplitValStartIndex[feaSplitValStartIndex.length - 1];
    }

    public void info(String info) throws Mp4jException {
        if (comm != null && owner) {
            while (info.length() > MAX_INFO_LEN) {
                comm.info("[GBDT] " +  info.substring(0, MAX_INFO_LEN), false);
                info = info.substring(MAX_INFO_LEN);
            }
            if (info.length() > 0) {
                comm.info("[GBDT] " + info, false);
            }
        }
    }

    // convert original feature value to slot number
    private void genGlobalFeaBinAndConvert(GBDTFeatureParams featParams) throws Exception {
        SampleManager sampleManager = new SampleManager(data, comm);
        sampleManager.init(featParams);
        genApprFeaSorted(sampleManager);
        convertFeaVal2ApprFeaIndex();
        reverseApprFeaVal2OriFeaVal(sampleManager);
    }

    public void updateGlobalHistAssign() {
        int slaveNum = comm.getSlaveNum();
        int threadNum = comm.getThreadNum();
        globalHistAssignFrom = new int[slaveNum][threadNum];
        globalHistAssignCount = new int[slaveNum][threadNum];

        for (int i = 0; i < slaveNum; i++) {
            for (int j = 0; j < threadNum; j++) {
                globalHistAssignFrom[i][j] = feaSplitValStartIndex[data.globalFeatureAssignFrom[i][j]] << 1;
                globalHistAssignCount[i][j] = (feaSplitValStartIndex[data.globalFeatureAssignTo[i][j]] -
                        feaSplitValStartIndex[data.globalFeatureAssignFrom[i][j]])<< 1;
            }
        }
    }

    // sort global feature values
    private void genApprFeaSorted(SampleManager sampleManager) throws Exception {
        Map<String, Set<Float>> globalFeaSplitVals = sampleManager.doSample();

        int feaDim = globalFeaSplitVals.size();
        globalFeaSplitValsSorted = new HashMap<>(feaDim);

        // the last element is the total count of approximate features.
        feaSplitValStartIndex = new int[feaDim + 1];
        for (int i = 0; i < feaDim; i++) {
            feaSplitValStartIndex[i] = 0;
        }

        for (Map.Entry<String, Set<Float>> entry : globalFeaSplitVals.entrySet()) {
            Set<Float> feaSet = entry.getValue();
            float[] sortFeas = new float[feaSet.size()];
            int index = 0;
            for (Float fea : entry.getValue()) {
                sortFeas[index] = fea;
                index++;
            }

            Arrays.sort(sortFeas);
            int fid = Integer.parseInt(entry.getKey());

            globalFeaSplitValsSorted.put(fid, sortFeas);
            feaSplitValStartIndex[fid + 1] = sortFeas.length;
        }

        for (int i = 1; i < feaSplitValStartIndex.length; i++) {
            feaSplitValStartIndex[i] += feaSplitValStartIndex[i - 1];
        }

        boolean isFirst = true;
        StringBuffer sb = new StringBuffer("");
        Map<Integer, String> fIndex2NameMap = data.fIndex2NameMap;
        for (Map.Entry<Integer, float[]> entry : globalFeaSplitValsSorted.entrySet()) {
            int fid = entry.getKey();
            if (isFirst) {
                isFirst = false;
                sb.append(fIndex2NameMap.get(fid) + "=" + entry.getValue().length);
                if (verbose) {
                    sb.append(", vals:" + StringUtils.join(entry.getValue(), ","));
                }
            } else {
                if (!verbose) {
                    sb.append(",");
                    sb.append(fIndex2NameMap.get(fid) + "=" + entry.getValue().length);
                } else {
                    sb.append("\n");
                    sb.append(fIndex2NameMap.get(fid) + "=" + entry.getValue().length);
                    sb.append(", vals:" + StringUtils.join(entry.getValue(), ","));
                }
            }
        }

        String head = "generate sorted global feature bins complete! feature dim:" + globalFeaSplitValsSorted.size()
                + ", total feature bin cnt:" + feaSplitValStartIndex[feaSplitValStartIndex.length - 1];
        if (!verbose) {
            info(head + ", details(format:feature_name=bin_cnt):\n" + sb.toString());
        } else {
            info(head + ", details(format: feature_name=bin_cnt, bin_vals):\n" + sb.toString());
        }
    }

    // convert origin feature value to approximate feature slot num(feature bin)
    private void convertFeaVal2ApprFeaIndex() {
        int sampleNum = data.sampleNum;
        int featureDim = data.usefulFeatureDim;

        int feaApproIdx;
        float featVal;
        for (int fid = 0; fid < featureDim; fid++) {
            float[] globalApproFea = globalFeaSplitValsSorted.get(fid);
            CheckUtils.check(globalApproFea != null && globalApproFea.length > 0, "[GBDT] fid:" + fid + " has no global feature vals!");
            for (int sid = 0; sid < sampleNum; sid++) {
                if (globalApproFea.length == 1) {
                    feaApproIdx = 0;
                } else {
                    featVal = NumConvertUtils.int2float(data.getFeatureVal(sid, fid));
                    feaApproIdx = BinarySearch.findLastEqualOrUpper(globalApproFea, featVal);
                    if (feaApproIdx == -1) {
                        feaApproIdx = globalApproFea.length - 1;
                    } else if (feaApproIdx >= 1) {
                        if (featVal < (globalApproFea[feaApproIdx] + globalApproFea[feaApproIdx - 1]) * 0.5) {
                            feaApproIdx = feaApproIdx - 1;
                        }
                    }
                }
                data.setFeatureVal(sid, fid, feaApproIdx);
            }
        }
    }

    // reverse converted feature value to original value,
    // used to convert split values in tree to original feature value
    private void reverseApprFeaVal2OriFeaVal(SampleManager sampleManager) {
        for (Map.Entry<Integer, float[]> entry : globalFeaSplitValsSorted.entrySet()) {
            sampleManager.reverse(entry.getKey(), entry.getValue());
        }
    }


    public int[] getFeaSplitValStartIndex() {
        return feaSplitValStartIndex;
    }

    public int getNumBin() {
        return numBin;
    }

    public Map<Integer, float[]> getGlobalFeaSplitValsSorted() {
        return globalFeaSplitValsSorted;
    }

    public int[][] getGlobalHistAssignFrom() {
        return globalHistAssignFrom;
    }

    public int[][] getGlobalHistAssignCount() {
        return globalHistAssignCount;
    }

}
