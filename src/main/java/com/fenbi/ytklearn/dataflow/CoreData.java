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
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.IObjectOperator;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.mp4j.utils.KryoUtils;
import com.fenbi.ytklearn.param.TransformParams;
import com.fenbi.ytklearn.utils.LogUtils;
import com.fenbi.ytklearn.utils.NumConvertUtils;
import lombok.Data;
import org.apache.commons.lang.ArrayUtils;
import org.python.core.PyByteArray;
import org.python.core.PyFunction;
import org.python.core.PyList;

import java.io.UnsupportedEncodingException;
import java.util.*;

/**
 * @author xialong
 */

@Data
public class CoreData {
    public static final int INIT_MAP_SIZE = 500000;
    public static final int MAX_2D_LEN = 50000;
    public static final int MAX_1D_LEN = 2000000;
    public static final int ONE_BITS = Float.floatToRawIntBits(1.0f);

    // TODO: protected -> public
    public List<String> LINE_LIST;// = new ArrayList<>();
    public int [] TMP_XIDX;// = new int[MAX_1D_LEN + 1];
    public float []TMP_Y;// = new float[MAX_1D_LEN];
    public float []TMP_WEIGHT; // new float[MAX_1D_LEN];

    public int [][]x;
    public int [][]xidx;
    public float [][]y;
    public float [][]weight;
    public float [][]predict;
    public int []realNum;
    public double []weightNum;
    public long totalRealNum = 0;
    public double totalWeightNum = 0.0;
    public int cursor2d = 0;

    public float []label;
    public int labelIdx;
    public long []yRealNumStat;
    public double []yWeightNumStat;
    public float wei;
    public long lineCnt = 0;
    public int count = 0;
    public int xindex = 0;
    public long errorNum = 0;
    public int maxErrorTolNum;

    public final ThreadCommSlave comm;
    public final DataFlow.CoreParams coreParams;
    public final IFeatureMap featureMap;
    public final PyFunction pyTransformFunc;
    public final boolean needPyTransform;


    public int yNum;
    public String loadingPrefix;
    public boolean loadingTrainData;
    public Random rand = new Random();

    public int biasDelta;

    public long gErrorNum;
    public long gRealNum;
    public double gWeightNum;
    public long []gYRealNumStat;
    public double []gYWeightNumStat;

    public boolean needFeatureTransform;

    public LogUtils LOG_UTILS;

    public static class FeatureStat {
        public long cnt;
        public double sum;
        public double sum2;
        public double max;
        public double min;

        public FeatureStat(long cnt, double sum, double sum2, double max, double min) {
            this.cnt = cnt;
            this.sum = sum;
            this.sum2 = sum2;
            this.max = max;
            this.min = min;
        }

        public void update(float val) {
            cnt ++;
            sum += val;
            sum2 += val * val;
            max = max >= val ? max : val;
            min = min <= val ? min : val;
        }

        public TransformNode convert(TransformParams.Mode mode, double rangeMax, double rangeMin) {
            double mean = sum / cnt;
            double mean2 = sum2 / cnt;
            return new TransformNode(mode, mean, Math.sqrt(mean2 - mean * mean), max, min, rangeMax, rangeMin);
        }

        @Override
        public String toString() {
            return "FeatureStat{" +
                    "cnt=" + cnt +
                    ", sum=" + sum +
                    ", sum2=" + sum2 +
                    ", max=" + max +
                    ", min=" + min +
                    '}';
        }
    }

    public static class XLong {
        public long val;
    }

    protected Map<String, XLong> featureXCntMap = new HashMap<>();
    protected Map<String, Long> featureCntMap;

    public static class TransformNode {
        public TransformParams.Mode mode;
        public double mean;
        public double stdvar;
        public double max;
        public double min;
        public double rangeMax;
        public double rangeMin;


        public TransformNode(TransformParams.Mode mode,
                             double mean,
                             double stdvar,
                             double max,
                             double min,
                             double rangeMax,
                             double rangeMin) {
            this.mode = mode;
            this.mean = mean;
            this.stdvar = stdvar;
            this.max = max;
            this.min = min;
            this.rangeMax = rangeMax;
            this.rangeMin = rangeMin;
        }

        public float transform(float val) {
            if (mode == TransformParams.Mode.STANDARDIZATION) {
                if (stdvar < 0.000001) {
                    return val;
                }
                return (float)((val - mean) / stdvar);
            } else {
                if (Math.abs(max - min) < 0.000001) {
                    return 1.0f;
                }
                return (float)(rangeMin + ((rangeMax - rangeMin) * ((val - min) / (max - min))));
            }

        }

        @Override
        public String toString() {
            return  "mode=" + mode +
                    ", mean=" + mean +
                    ", stdvar=" + stdvar +
                    ", max=" + max +
                    ", min=" + min +
                    ", rangeMax=" + rangeMax +
                    ", rangeMin=" + rangeMin;

        }

        public static TransformNode fromString(String line) {
            String []info = line.split(",");
            TransformParams.Mode mode = TransformParams.Mode.getMode(info[0].split("=")[1].trim());
            double mean = Double.parseDouble(info[1].split("=")[1].trim());
            double stdvar = Double.parseDouble(info[2].split("=")[1].trim());
            double max = Double.parseDouble(info[3].split("=")[1].trim());
            double min = Double.parseDouble(info[4].split("=")[1].trim());
            double rangeMax = Double.parseDouble(info[5].split("=")[1].trim());
            double rangeMin = Double.parseDouble(info[6].split("=")[1].trim());

            TransformNode node = new TransformNode(mode, mean, stdvar, max, min, rangeMax, rangeMin);
            return node;
        }
    }

    public Map<String, FeatureStat> featureStat = new HashMap<>();

    public CoreData(ThreadCommSlave comm) {
        this.comm = comm;
        this.coreParams = null;
        this.featureMap = null;
        this.pyTransformFunc = null;
        this.needPyTransform = false;
    }

    public CoreData(ThreadCommSlave comm,
                    DataFlow.CoreParams coreParams,
                    IFeatureMap featureMap,
                    PyFunction pyTransformFunc,
                    boolean needPyTransform) {
        this.comm = comm;
        this.coreParams = coreParams;
        this.featureMap = featureMap;
        this.pyTransformFunc = pyTransformFunc;
        this.needPyTransform = needPyTransform;

        this.LOG_UTILS = new LogUtils(comm, coreParams.verbose);

        this.biasDelta = coreParams.need_bias ? 1 : 0;

        this.x = new int[MAX_2D_LEN][];
        this.xidx = new int[MAX_2D_LEN][];
        this.y = new float[MAX_2D_LEN][];
        this.weight = new float[MAX_2D_LEN][];

        this.realNum = new int[MAX_2D_LEN];
        this.weightNum = new double[MAX_2D_LEN];

        this.predict = new float[MAX_2D_LEN][];

        this.needFeatureTransform = coreParams.featureParams != null &&
                coreParams.featureParams.transform != null &&
                coreParams.featureParams.transform.switch_on;
    }

    public void releaseFeatureStatMap() {
        featureStat = null;
    }

    public void releaseFeatureCntMap() {
        featureCntMap = null;
        featureXCntMap = null;
    }

    public void initAssistData() {

        LINE_LIST = new ArrayList<>();
        LINE_LIST.add("temp");
        TMP_XIDX = new int[MAX_1D_LEN + 1];
        TMP_Y = new float[MAX_1D_LEN];
        TMP_WEIGHT = new float[MAX_1D_LEN];

    }

    public long getTotalRealNum() {
        totalRealNum = 0;
        for (int i = 0; i < cursor2d; i++) {
            totalRealNum += realNum[i];
        }
        return totalRealNum;
    }

    public double getTotalWeightNum() {
        totalWeightNum = 0.0;
        for (int i = 0; i < cursor2d; i++) {
            totalWeightNum += weightNum[i];
        }
        return totalWeightNum;
    }

    protected Iterator nextSamples(String line, boolean isTransform) throws UnsupportedEncodingException {
        Iterator iter;
        if (isTransform) {
            iter = transform(line, pyTransformFunc).iterator();
        } else {
            LINE_LIST.set(0, line);
            iter = LINE_LIST.iterator();
        }

        return iter;
    }

    private static PyList transform(String line, PyFunction pyTransformFunc) throws UnsupportedEncodingException {
        return (PyList) pyTransformFunc.__call__(new PyByteArray(line.getBytes("utf-8")));
    }

    protected String[] trainDataSplit(String line) {
        return line.trim().split(coreParams.x_delim);
    }

    protected void weightExtract(String line, String[] info) {
        wei = Float.parseFloat(info[0]);
    }

    protected boolean yExtract(String line, String[] info) throws Exception {
        label[0] = Float.parseFloat(info[1]);

        labelIdx = (int) label[0];

        if (!coreParams.needYSampling) {
            return true;
        }

        float rate = coreParams.ySampling[(int)label[0]];
        if (rate <= 1.0f) {
            wei *= (1.0f / rate);
        } else {
            wei *= rate;
        }

        return rand.nextFloat() <= rate;
    }

    protected boolean exceed1DRange() {
        return xindex >= CoreData.MAX_1D_LEN - 100000;
    }

    protected void exceed1DHandle() throws Mp4jException {
        TMP_XIDX[count] = xindex;
        int localnum = realNum[cursor2d];

        if (localnum != count) {
            LOG_UTILS.verboseInfo(loadingPrefix + "----error! localnum:" + localnum + ", count:" + count, false);
        }
        xidx[cursor2d] = new int[localnum + 1];
        System.arraycopy(TMP_XIDX, 0, xidx[cursor2d], 0, localnum + 1);

        y[cursor2d] = new float[localnum];
        System.arraycopy(TMP_Y, 0, y[cursor2d], 0, localnum);

        weight[cursor2d] = new float[localnum];
        System.arraycopy(TMP_WEIGHT, 0, weight[cursor2d], 0, localnum);

        predict[cursor2d] = new float[localnum];

        xindex = 0;
        count = 0;
        cursor2d++;
    }


    protected Map<String, Float> line2FeatureMap(String line, String []info) {
        Map<String, Float> X = new HashMap<>();
        String [] finfo = info[2].trim().split(coreParams.features_delim);
        for (String f : finfo) {
            String []fvinfo = f.split(coreParams.feature_name_val_delim);
            X.put(fvinfo[0].trim(), NumConvertUtils.parseFloat(fvinfo[1]));
        }

        return X;
    }

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

                if (loadingTrainData && needFeatureTransform) {
                    FeatureStat fstat = featureStat.get(fn);
                    if (fstat == null) {
                        fstat = new FeatureStat(1L, fv, fv * fv, fv, fv);
                        featureStat.put(fn, fstat);
                    } else {
                        fstat.update(fv);
                    }
                }

                Integer findex = featureMap.getIndex(fn);
                if (findex == null) {
                    continue;
                }
                if (loadingTrainData && !coreParams.need_dict) {
                    findex += biasDelta;
                }

                x[cursor2d][xindex++] = findex;
                x[cursor2d][xindex++] = Float.floatToRawIntBits(fv);
            }

            // 添加bias
            if (coreParams.need_bias) {
                x[cursor2d][xindex++] = 0;
                x[cursor2d][xindex++] = CoreData.ONE_BITS;
            }

            if (lineCnt < 10) {
                LOG_UTILS.verboseInfo("wei:" + wei + ", label:" + ArrayUtils.toString(label) + ", features:" + fnvMap, false);
            }

        } catch (Exception e) {
            errorNum++;
            LOG_UTILS.error(loadingPrefix + "[ERROR] error format:" + line +
                    ", local error total num:" + errorNum +
                    ", max error tol:" + maxErrorTolNum +
                    ", has read lines:" + lineCnt);
            if (errorNum > maxErrorTolNum) {
                LOG_UTILS.error("[ERROR] error num:" + errorNum +
                        " > " + "max tol:" + maxErrorTolNum);
                throw e;
            }
            return false;
        }


        return true;
    }

    protected boolean exceed2DRange() {
        return cursor2d >= x.length;
    }

    protected void exceed2DHandle() {

        int[][] new_x = new int[x.length * 2][];
        int[][] new_xidx = new int[x.length * 2][];
        float[][] new_y = new float[x.length * 2][];
        float[][] new_weight = new float[x.length * 2][];
        int[] new_realNum = new int[x.length * 2];
        double[] new_weightNum = new double[x.length * 2];
        float[][] new_predict = new float[x.length * 2][];

        for (int i = 0; i < x.length; i++) {
            new_x[i] = x[i];
            new_xidx[i] = xidx[i];
            new_y[i] = y[i];
            new_weight[i] = weight[i];
            new_realNum[i] = realNum[i];
            new_weightNum[i] = weightNum[i];
            new_predict[i] = predict[i];
        }

        x = new_x;
        xidx = new_xidx;
        y = new_y;
        weight = new_weight;
        realNum = new_realNum;
        weightNum = new_weightNum;
        predict = new_predict;
    }

    protected void alloc1D() {
        if (x[cursor2d] == null) {
            x[cursor2d] = new int[CoreData.MAX_1D_LEN];
        }
    }

    protected void updateXidx() throws Mp4jException {
        //int weiInt = Float.floatToRawIntBits(wei);
        TMP_XIDX[count] = xindex;
        //TMP_XIDX[(count << 1) + 1] = weiInt;
        TMP_WEIGHT[count] = wei;
    }

    protected void updateY() throws Exception {
        TMP_Y[count] = label[0];
    }

    protected void updateStat() throws Mp4jException {
        realNum[cursor2d]++;
        weightNum[cursor2d] += wei;
        lineCnt++;
        count++;

        if (coreParams.needYStat) {
            yRealNumStat[labelIdx] += 1;
            yWeightNumStat[labelIdx] += wei;
        }

        if (lineCnt % 10000 == 0) {
            LOG_UTILS.importantInfo("has readed lines:" + lineCnt, false);
        }
    }

    protected void lastSampleHandle() throws Mp4jException {
        TMP_XIDX[count] = xindex;
        int localnum = realNum[cursor2d];
        xidx[cursor2d] = new int[localnum + 1];
        System.arraycopy(TMP_XIDX, 0, xidx[cursor2d], 0, localnum + 1);

        y[cursor2d] = new float[localnum];
        System.arraycopy(TMP_Y, 0, y[cursor2d], 0, localnum);

        weight[cursor2d] = new float[localnum];
        System.arraycopy(TMP_WEIGHT, 0, weight[cursor2d], 0, localnum);

        predict[cursor2d] = new float[localnum];

        LOG_UTILS.verboseInfo(loadingPrefix + "finished read data, cursor2d:" + cursor2d +
                ", real num:" + ArrayUtils.toString(Arrays.copyOfRange(realNum, 0, cursor2d + 1)) +
                ", weight sum:" + ArrayUtils.toString(Arrays.copyOfRange(weightNum, 0, cursor2d + 1)), false);
    }

    protected void otherHandle(String line, String[] info) {}

    public void readData(Iterator<String> dataIter,
                         boolean loadingTrainData,
                         int yNum) throws Exception {

        if (loadingTrainData) {
            this.loadingTrainData = true;
            maxErrorTolNum = coreParams.train_max_error_tol;
            loadingPrefix = "[train data]";
        } else {
            this.loadingTrainData = false;
            maxErrorTolNum = coreParams.test_max_error_tol;
            loadingPrefix = "[test data]";
        }
        this.yNum = yNum;
        this.label = new float[yNum];

        if (coreParams.needYStat) {
            int classNum = yNum;
            if (classNum == 1) {
                classNum = 2;
            }
            this.yRealNumStat = new long[classNum];
            this.yWeightNumStat = new double[classNum];
            for (int i = 0; i < classNum; i++) {
                this.yRealNumStat[i] = 0;
                this.yWeightNumStat[i] = 0.0;
            }
        }

        long start = System.currentTimeMillis();
        while (dataIter.hasNext()) {
            String rawLine = dataIter.next();
            Iterator iter = nextSamples(rawLine, needPyTransform);

            while (iter.hasNext()) {
                String line = (String) iter.next();

                String[] info = trainDataSplit(line);

                weightExtract(line, info);

                if (!yExtract(line, info)) {
                    continue;
                }

                if (exceed1DRange()) {
                    exceed1DHandle();
                }

                if (exceed2DRange()) {
                    exceed2DHandle();
                }

                alloc1D();

                updateXidx();

                if (!updateX(line, info)) {
                    continue;
                }

                updateY();

                otherHandle(line, info);

                updateStat();
            }
        }

        lastSampleHandle();

        cursor2d++;
        long cost = System.currentTimeMillis() - start;
        LOG_UTILS.verboseInfo(loadingPrefix + "this slave read lines:" + lineCnt + ", read takes:" + cost / 1000. + "s", false);

    }

    public void globalSync() throws Mp4jException{
        gErrorNum = comm.allreduce(errorNum, Operands.LONG_OPERAND(), Operators.Long.SUM);
        gRealNum = comm.allreduce(getTotalRealNum(), Operands.LONG_OPERAND(), Operators.Long.SUM);
        gWeightNum = comm.allreduce(getTotalWeightNum(), Operands.DOUBLE_OPERAND(), Operators.Double.SUM);

        if (coreParams.needYStat) {
            gYRealNumStat = comm.allreduceArray(yRealNumStat, Operands.LONG_OPERAND(), Operators.Long.SUM, 0, yRealNumStat.length);
            gYWeightNumStat = comm.allreduceArray(yWeightNumStat, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, yWeightNumStat.length);
        }

        if (loadingTrainData && !coreParams.need_dict) {
            featureCntMap = new HashMap<>(featureXCntMap.size());
            for (Map.Entry<String, XLong> entry : featureXCntMap.entrySet()) {
                featureCntMap.put(entry.getKey(), entry.getValue().val);
            }
            featureCntMap = comm.allreduceMap(featureCntMap, Operands.LONG_OPERAND(), Operators.Long.SUM);
        }

        if (loadingTrainData && needFeatureTransform) {
            featureStat = comm.allreduceMap(featureStat, Operands.OBJECT_OPERAND(KryoUtils.getDefaultSerializer(FeatureStat.class), FeatureStat.class), new IObjectOperator<FeatureStat>() {
                @Override
                public FeatureStat apply(FeatureStat s1, FeatureStat s2) {
                    s1.cnt += s2.cnt;
                    s1.sum += s2.sum;
                    s1.sum2 += s2.sum2;
                    s1.max = s1.max >= s2.max ? s1.max : s2.max;
                    s1.min = s1.min <= s2.min ? s1.min : s2.min;
                    return s1;
                }
            });
            LOG_UTILS.verboseInfo(featureStat.toString());
        }
    }

}
