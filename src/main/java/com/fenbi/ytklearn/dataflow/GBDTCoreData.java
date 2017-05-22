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
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.ytklearn.data.Constants;
import com.fenbi.ytklearn.exception.YtkLearnException;
import com.fenbi.ytklearn.loss.ILossFunction;
import com.fenbi.ytklearn.utils.CheckUtils;
import lombok.Data;
import org.apache.commons.lang.ArrayUtils;
import org.python.core.PyFunction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

/**
 * used for data-parallel tree maker
 * @author wufan
 * @author xialong
 */

@Data
public class GBDTCoreData extends CoreData {
    public final int DENSE_MAX_1D_SAMPLE_CNT;
    public final int DENSE_MAX_1D_LEN;

    public final int maxFeatureDim;
    public int usefulFeatureDim;

    public final int numTreeInGroup;

    public ILossFunction obj;
    private float baseScore;
    private boolean sampleDepdtBasePrediction;

    // set outside
    public Map<Integer, String> fIndex2NameMap;
    // set outside, compute once, used in thread data
    public int sampleNum;
    public double weightSum;

    // === used for train phase ===
    public int lastPredRound;

    public float[] TMP_INIT_SCORE;
    public float[][] initScore;

    // prediction buffer, save raw prediction(score before loss function) in last iter
    public float[][] score;
    // first order and second order gradient  eg: grad1, hess1, grad2, hess2,
    public float[][] gradPairs;

    // len(xColRange) = 2, [startXColIndex, endXColIndex)
    public int[] xColRange;
    // global feature range for each worker(thread)
    public int[][] globalFeatureAssignFrom;
    public int[][] globalFeatureAssignTo;

    // === used in local(feature-parallel) version ===
    // len(xRowRange) = 2, [startXRowIndex, endXRowIndex)
    public int[] xRowRange;
    public int[][] globalSampleAssignFrom;
    public int[][] globalSampleAssignTo;

    public int[][] globalGradAssignFrom;
    public int[][] globalGradAssignTo;
    public GBDTCoreData allData;
    // feature cols for thread data, only used in local version
    public FeatureColData xT;


    public GBDTCoreData(ThreadCommSlave comm,
                        DataFlow.CoreParams coreParams,
                        IFeatureMap featureMap,
                        PyFunction pyTransformFunc,
                        boolean needPyTransfor,
                        int maxFeatureDim,
                        int numTreeInGroup,
                        ILossFunction obj,
                        float baseScore,
                        boolean sampleDepdtBasePrediction) {
        super(comm, coreParams, featureMap, pyTransformFunc, needPyTransfor);
        this.initScore = new float[MAX_2D_LEN][];
        this.score = new float[MAX_2D_LEN][];

        this.DENSE_MAX_1D_SAMPLE_CNT = MAX_1D_LEN / maxFeatureDim;
        this.DENSE_MAX_1D_LEN = DENSE_MAX_1D_SAMPLE_CNT * maxFeatureDim;
        this.lastPredRound = 0;

        this.maxFeatureDim = maxFeatureDim;
        this.numTreeInGroup = numTreeInGroup;
        this.obj = obj;
        this.baseScore = baseScore;
        this.sampleDepdtBasePrediction = sampleDepdtBasePrediction;
    }

    // for feature parellel
    public static GBDTCoreData mergeThreadFeatData(CoreData[] threadCorData) {
        if (threadCorData == null || threadCorData.length == 0) {
            return null;
        }
        int total2D = 0;
        for (int t = 0; t < threadCorData.length; t++) {
            total2D += threadCorData[t].cursor2d;
        }
        GBDTCoreData tdata0 = (GBDTCoreData)threadCorData[0];
        GBDTCoreData allData = new GBDTCoreData(tdata0.comm, tdata0.coreParams, tdata0.featureMap, tdata0.pyTransformFunc, tdata0.needPyTransform,
                tdata0.maxFeatureDim, tdata0.numTreeInGroup, tdata0.obj, tdata0.baseScore, tdata0.sampleDepdtBasePrediction);
        allData.sampleNum = (int)tdata0.gRealNum;
        allData.cursor2d = total2D;
        allData.x = new int[total2D][];
        allData.realNum = new int[total2D];
        int index2D = 0;
        for (CoreData data: threadCorData) {
            for (int i = 0; i < data.cursor2d; i++) {
                allData.x[index2D] = data.x[i];
                allData.realNum[index2D] = data.realNum[i];
                index2D++;
            }
        }
        return allData;
    }

    //local(feature-parallel) version: reverse features, get features for thread data
    public void createFeatureColData() {
        xT = new FeatureColData(allData, xColRange);
    }


    @Override
    public void initAssistData() {
        LINE_LIST = new ArrayList<>();
        LINE_LIST.add("temp");

        TMP_Y = new float[DENSE_MAX_1D_SAMPLE_CNT * numTreeInGroup];
        TMP_WEIGHT = new float[DENSE_MAX_1D_SAMPLE_CNT];
        TMP_INIT_SCORE = new float[DENSE_MAX_1D_SAMPLE_CNT * numTreeInGroup];
    }

    public int getFeatureVal(int sid, int fid) {
        int index2D = sid / DENSE_MAX_1D_SAMPLE_CNT;
        int index1D = sid % DENSE_MAX_1D_SAMPLE_CNT;
        return x[index2D][index1D * maxFeatureDim + fid];
    }

    public void setFeatureVal(int sid, int fid, int val) {
        int index2D = sid / DENSE_MAX_1D_SAMPLE_CNT;
        int index1D = sid % DENSE_MAX_1D_SAMPLE_CNT;
        x[index2D][index1D * maxFeatureDim + fid] = val;
    }

    public boolean isEqualReplaceFeaVal(int sid, int fid, int refVal, int newVal) {
        int index2D = sid / DENSE_MAX_1D_SAMPLE_CNT;
        int index1D = (sid % DENSE_MAX_1D_SAMPLE_CNT) * maxFeatureDim + fid;
        if (x[index2D][index1D] == refVal) {
            x[index2D][index1D] = newVal;
            return true;
        } else {
            return false;
        }
    }

    public float getSampleWeight(int sid) {
        int index2D = sid / DENSE_MAX_1D_SAMPLE_CNT;
        int index1D = sid % DENSE_MAX_1D_SAMPLE_CNT;
        return weight[index2D][index1D];
    }

    // used in train phase, alloc space for gradPairs
    public void initGradPairs() {
        gradPairs = new float[cursor2d][];
        for (int i = 0; i < cursor2d; i++) {
            gradPairs[i] = new float[realNum[i] * 2 * numTreeInGroup];
            for (int j = 0; j < gradPairs[i].length; j++) {
                gradPairs[i][j] = 0.f;
            }
        }
    }

    @Override
    protected String[] trainDataSplit(String line) {
        String[] info = line.trim().split(coreParams.x_delim);
        CheckUtils.check(info.length >=3, "[GBDT] data format error! line:%s", line);
        return info;
    }

    @Override
    protected void updateY() {
        int yidx = count * numTreeInGroup;
        for (int i = 0; i < numTreeInGroup; i++) {
            TMP_Y[yidx + i] = label[i];
        }
    }

    @Override
    protected boolean yExtract(String line, String[] info) throws Exception {

        //regression & binary classification & ...
        if (numTreeInGroup == 1) {
            label[0] = Float.parseFloat(info[1]);
            CheckUtils.check(obj.checkLabel(label[0]), "[GBDT] label error, line: %s", line);
            labelIdx = (int) label[0];

        } else { // multiclass softmax
            String[] linfo = info[1].split(coreParams.y_delim);
            CheckUtils.check(linfo.length == numTreeInGroup || linfo.length == 1, "[GBDT] label num must equal %d or 1, line: %s", numTreeInGroup, line);

            if (linfo.length == 1) {
                for (int i = 0; i < numTreeInGroup; i++) {
                    label[i] = 0;
                }
                int clazz = Integer.parseInt(linfo[0]);
                if (clazz >= numTreeInGroup) {
                    throw new YtkLearnException("multi classification label must in range [0,K-1]!\n" + line);
                }
                label[clazz] = 1.0f;
            } else {
                for (int i = 0; i < numTreeInGroup; i++) {
                    label[i] = Float.parseFloat(linfo[i]);
                }
            }

            CheckUtils.check(obj.checkLabel(label), "[GBDT] all label sum must equal 1.0, line: %s", line);
            if (coreParams.needYStat) {
                labelIdx = -1;
                for (int i = 0; i < numTreeInGroup; i++) {
                    if (label[i] == 1.0) {
                        labelIdx = i;
                    }
                }
            }

        }

        if (!coreParams.needYSampling) {
            return true;
        }

        CheckUtils.check(labelIdx != -1, "[GBDT] label error! line: %s", line);
        float rate = coreParams.ySampling[labelIdx];
        if (rate <= 1.0f) {
            wei *= (1.0f / rate);
        } else {
            wei *= rate;
        }
        return rand.nextFloat() <= rate;
    }

    @Override
    protected boolean exceed1DRange() {
        return xindex >= DENSE_MAX_1D_LEN;
    }

    @Override
    protected void exceed1DHandle() throws Mp4jException {
        int localnum = realNum[cursor2d];

        if (localnum != count) {
            LOG_UTILS.verboseInfo(loadingPrefix + "----error! localnum:" + localnum + ", count:" + count, false);
        }

        weight[cursor2d] = new float[localnum];
        System.arraycopy(TMP_WEIGHT, 0, weight[cursor2d], 0, localnum);

        y[cursor2d] = new float[localnum * numTreeInGroup];
        System.arraycopy(TMP_Y, 0, y[cursor2d], 0, localnum * numTreeInGroup);

        initScore[cursor2d] = new float[localnum * numTreeInGroup];
        System.arraycopy(TMP_INIT_SCORE, 0, initScore[cursor2d], 0, localnum * numTreeInGroup);

        score[cursor2d] = new float[localnum * numTreeInGroup];
        predict[cursor2d] = new float[localnum * numTreeInGroup];

        xindex = 0;
        count = 0;
        cursor2d++;
    }

    @Override
    protected void alloc1D() {
        if (x[cursor2d] == null) {
            x[cursor2d] = new int[DENSE_MAX_1D_LEN];
            for (int i = 0; i < DENSE_MAX_1D_LEN; i++) {
                x[cursor2d][i] = Constants.INT_MISSING_VALUE;
            }
        }
    }

    @Override
    protected void exceed2DHandle() {
        int new_len = x.length * 2;
        int[][] new_x = new int[new_len][];
        float[][] new_y = new float[new_len][];
        float[][] new_weight = new float[new_len][];
        int[] new_realNum = new int[new_len];
        double[] new_weightNum = new double[new_len];
        float[][] new_margin = new float[new_len][];
        float[][] new_score = new float[new_len][];
        float[][] new_predict = new float[new_len][];

        for (int i = 0; i < x.length; i++) {
            new_x[i] = x[i];
            new_y[i] = y[i];
            new_weight[i] = weight[i];
            new_realNum[i] = realNum[i];
            new_weightNum[i] = weightNum[i];
            new_margin[i] = initScore[i];
            new_score[i] = score[i];
            new_predict[i] = predict[i];
        }

        x = new_x;
        y = new_y;
        weight = new_weight;
        realNum = new_realNum;
        weightNum = new_weightNum;
        initScore = new_margin;
        score = new_score;
        predict = new_predict;
    }

    @Override
    protected void updateXidx() throws Mp4jException {
        TMP_WEIGHT[count] = wei;
    }

    @Override
    protected boolean updateX(String line, String[] info) throws Mp4jException {

        try {
            Map<String, Float> fnvMap = line2FeatureMap(line, info);
            for (Map.Entry<String, Float> fnvMapEntry : fnvMap.entrySet()) {
                String fn = fnvMapEntry.getKey();
                Float fv = fnvMapEntry.getValue();

                if (loadingTrainData && !coreParams.need_dict) {
                    XLong xcnt = featureXCntMap.get(fn);
                    if (xcnt == null) {
                        xcnt = new XLong();
                        xcnt.val = 1;
                        featureXCntMap.put(fn, xcnt);
                    } else {
                        xcnt.val ++;
                    }
                }

                Integer findex = featureMap.getIndex(fn);
                if (findex == null) {
                    continue;
                }
                if (loadingTrainData) {
                    findex += biasDelta;
                }

                CheckUtils.check(findex < maxFeatureDim, "[GBDT] max_feature_dim(%d) smaller than real feature number in data set, local feature index is %d, sample:%s",
                        maxFeatureDim, findex, line);
                x[cursor2d][xindex + findex] = Float.floatToRawIntBits(fv);
            }

            xindex += maxFeatureDim;
            if (lineCnt < 10) {
                LOG_UTILS.verboseInfo("weight:" + wei + ", label:" + ArrayUtils.toString(label) + ", features:" + fnvMap, false);
            }

        } catch (Exception e) {
            errorNum++;
            LOG_UTILS.error(loadingPrefix + "[ERROR] error format:" + line +
                    ", local error total num:" + errorNum +
                    ", max error tol:" + maxErrorTolNum +
                    ", has read lines:" + lineCnt);
            if (errorNum > maxErrorTolNum) {
                LOG_UTILS.error("[ERROR] train error num:" + errorNum +
                        " > " + "max tol:" + maxErrorTolNum);
                throw e;
            }
            return false;
        }

        return true;
    }

    protected void lastSampleHandle() throws Mp4jException {
        int localnum = realNum[cursor2d];

        weight[cursor2d] = new float[localnum];
        System.arraycopy(TMP_WEIGHT, 0, weight[cursor2d], 0, localnum);

        y[cursor2d] = new float[localnum * numTreeInGroup];
        System.arraycopy(TMP_Y, 0, y[cursor2d], 0, localnum * numTreeInGroup);

        initScore[cursor2d] = new float[localnum * numTreeInGroup];
        System.arraycopy(TMP_INIT_SCORE, 0, initScore[cursor2d], 0, localnum * numTreeInGroup);

        score[cursor2d] = new float[localnum * numTreeInGroup];
        predict[cursor2d] = new float[localnum * numTreeInGroup];

        LOG_UTILS.verboseInfo(loadingPrefix + "finished read data, cursor2d:" + cursor2d +
                ", real num:" + ArrayUtils.toString(Arrays.copyOfRange(realNum, 0, cursor2d + 1)) +
                ", weight sum:" + ArrayUtils.toString(Arrays.copyOfRange(weightNum, 0, cursor2d + 1)), false);
    }

    @Override
    protected void otherHandle(String line, String[] info) {
        if (numTreeInGroup == 1) {
            TMP_INIT_SCORE[count] = baseScore;
            if (sampleDepdtBasePrediction) {
                TMP_INIT_SCORE[count] += (float)obj.pred2Score(Float.parseFloat(info[3]));
            }
        } else { // multi-class
            String[] linfo = null;
            if (sampleDepdtBasePrediction) {
                linfo = info[3].split(coreParams.y_delim);
                CheckUtils.check(linfo.length == numTreeInGroup,
                        "[GBDT] sample dependent score num must equal %d, %s", numTreeInGroup, line);
            }

            int offset = count * numTreeInGroup;
            for (int i = 0; i < numTreeInGroup; i++) {
                TMP_INIT_SCORE[offset + i] = baseScore;
                if (sampleDepdtBasePrediction) {
                    TMP_INIT_SCORE[offset + i] += (float)obj.pred2Score(Float.parseFloat(linfo[i]));
                }
            }
        }
    }

}
