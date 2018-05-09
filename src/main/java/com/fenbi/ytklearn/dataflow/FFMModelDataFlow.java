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

import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.ytklearn.feature.FeatureHash;
import com.fenbi.ytklearn.fs.IFileSystem;
import com.fenbi.ytklearn.param.RandomParams;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.fenbi.ytklearn.utils.RandomParamsUtils;
import com.typesafe.config.Config;
import lombok.Data;
import org.apache.commons.lang.ArrayUtils;
import org.python.core.PyFunction;

import java.io.*;
import java.util.*;

/**
 * @author xialong
 */

@Data
public class FFMModelDataFlow extends ContinuousDataFlow {
    private int K[];
    //private int seed;
    private boolean needFirstOrder;
    private boolean needSecondOrder;
    private boolean biasNeedLatentFactor;

    private String fieldDelim;
    private String fieldDictPath;

    private Map<String, Integer> field2IndexMap = new HashMap<>();
    private int fieldSize;
    private int maxFeatureNum = -1;
    private int maxFeatureDim = 100;

    private RandomParams randomParams;



    public FFMModelDataFlow(IFileSystem fs,
                           Config config,
                           ThreadCommSlave comm,
                           int threadNum,
                           boolean needPyTransform,
                           String pyTransformScript) throws Exception {
        super(fs, config,
                comm,
                threadNum,
                needPyTransform,
                pyTransformScript);

        List<Integer> klist = config.getIntList("k");
        K = new int[klist.size()];
        for (int i = 0; i < klist.size(); i++) {
            K[i] = klist.get(i);
        }
        //seed = config.getInt("seed");
        biasNeedLatentFactor = config.getBoolean("bias_need_latent_factor");

        needFirstOrder = (K[0] >= 1);
        needSecondOrder = (K[1] >= 1);

        fieldDelim = config.getString("data.delim.field_delim");
        fieldDictPath = config.getString("model.field_dict_path");
        maxFeatureDim = config.getInt("data.max_feature_dim");

        randomParams = new RandomParams(config, "");

        LOG_UTILS.importantInfo("K:" + Arrays.toString(K));
        //LOG_UTILS.importantInfo("seed:" + seed);
        LOG_UTILS.importantInfo("random:" + randomParams);
        LOG_UTILS.importantInfo("bias_need_latent_factor:" + biasNeedLatentFactor);
        LOG_UTILS.importantInfo("need_first_order:" + needFirstOrder + ", need_second_order:" + needSecondOrder);
        LOG_UTILS.importantInfo("field_delim:" + fieldDelim + ", field_dict_path:" + fieldDictPath);

    }

    @Data
    public static class FFMCoreData extends ContinuousCoreData {
        private int maxFeatureNum;
        private int maxFeatureDim;
        private String fieldDelim;
        private Map<String, Integer> field2IndexMap;

        public FFMCoreData(ThreadCommSlave comm,
                           DataFlow.CoreParams coreParams,
                           IFeatureMap featureMap,
                           PyFunction pyTransformFunc,
                           boolean needPyTransform,
                           FeatureHash featureHash,
                           int maxFeatureNum,
                           String fieldDelim,
                           Map<String, Integer> field2IndexMap) {
            super(comm, coreParams, featureMap, pyTransformFunc, needPyTransform, featureHash);
            this.maxFeatureNum = maxFeatureNum;
            this.fieldDelim = fieldDelim;
            this.field2IndexMap = field2IndexMap;
        }


        @Override
        protected boolean updateX(String line, String[] info) throws Mp4jException {

            try {
                Map<String, Float> fnvMap = line2FeatureMap(line, info);
                if (fnvMap.size() > maxFeatureNum) {
                    maxFeatureNum = fnvMap.size();
                }
                for (Map.Entry<String, Float> fnvMapEntry : fnvMap.entrySet()) {
                    String fn = fnvMapEntry.getKey();
                    String field = fnvMapEntry.getKey().split(fieldDelim)[0];
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

                    Integer fieldIdx = field2IndexMap.get(field);
                    if (fieldIdx == null) {
                        continue;
                    }

                    x[cursor2d][xindex++] = findex;
                    x[cursor2d][xindex++] = Float.floatToRawIntBits(fv);
                    x[cursor2d][xindex++] = fieldIdx;
                }

                // 添加bias
                if (coreParams.need_bias) {
                    x[cursor2d][xindex++] = 0;
                    x[cursor2d][xindex++] = CoreData.ONE_BITS;
                    x[cursor2d][xindex++] = 0;
                }

                if (lineCnt < 10) {
                    LOG_UTILS.verboseInfo("wei:" + wei + ", label:" + ArrayUtils.toString(label) + ", features:" + fnvMap, false);
                }

            } catch (Exception e) {
                errorNum++;
                LOG_UTILS.error(loadingPrefix + "[ERROR] error format:" + line +
                        ", local error total num:" + errorNum +
                        ", max error tol:" + maxErrorTolNum +
                        ", has readed lines:" + lineCnt);
                if (errorNum > maxErrorTolNum) {
                    LOG_UTILS.error("[ERROR] train error num:" + errorNum +
                            " > " + "max tol:" + maxErrorTolNum);
                    throw e;
                }
                return false;
            }


            return true;
        }

        @Override
        public void globalSync() throws Mp4jException{
            super.globalSync();
            maxFeatureNum = comm.allreduce(maxFeatureNum, Operands.INT_OPERAND(), Operators.Int.MAX);
        }
    }

    @Override
    protected boolean needLaplace() {
        return false;
    }

    @Override
    protected CoreData getCoreData() {
        return new FFMCoreData(comm, coreParams, featureMap, pyTransformFunc, needPyTransform, featureHash, maxFeatureNum, fieldDelim, field2IndexMap);
    }

    @Override
    protected void loadDict() throws IOException, Mp4jException {
        // load dict
        super.loadDict();

        // load field dict
        if (coreParams.need_bias) {
            field2IndexMap.put(coreParams.bias_feature_name, 0);
        }

        CheckUtils.check(fs.exists(fieldDictPath), "ffm model must contain field dict, set model.field_dict_path");

        DataUtils.travel(line -> field2IndexMap.put(line.trim(), field2IndexMap.size()),
                fs.read(Arrays.asList(fieldDictPath)));

        fieldSize = field2IndexMap.size();
        LOG_UTILS.importantInfo("field dict size:" + fieldSize + "\n" +
                field2IndexMap);

    }

    @Override
    protected void loadModel() throws IOException, Mp4jException {
        // 初始化为0
        //Random rand = new Random(seed);
        RandomParamsUtils randomParamsUtils = new RandomParamsUtils(randomParams);
        w = new float[threadNum][];
        precision = new float[threadNum][];
        int secondOrderIndexStart = secondOrderIndexStart();
        for (int t = 0; t < threadNum; t++) {
            w[t] = new float[dim];
        }
        for (int i = 0; i < secondOrderIndexStart; i++) {
            w[0][i] = 0;
        }
        for (int i = secondOrderIndexStart; i < dim; i++) {
            w[0][i] = (float) randomParamsUtils.next();
        }

        if (coreParams.need_bias) {
            for (int i = 0; i < K[1] * fieldSize; i++) {
                w[0][secondOrderIndexStart + i] = 0.0f;
            }
        }

        for (int t = 1; t < threadNum; t++) {
            System.arraycopy(w[0], 0, w[t], 0, w[t].length);
        }

        if (!modelParams.continue_train && !commonParams.lossParams.just_evaluate) {
            return;
        }

        if (!fs.exists(modelParams.data_path)) {
            LOG_UTILS.importantInfo("old model doesn't exist, new model...");
            return;
        }

        int cnt = 0;
        List<Iterator<String>> iterators = fs.read(Arrays.asList(modelParams.data_path));
        for (Iterator<String> it : iterators) {
            while (it.hasNext()) {
                String line = it.next();
                if (line.trim().length() == 0) {
                    LOG_UTILS.verboseInfo("[ERROR] invalid model line:" + line);
                    continue;
                }
                String[] info = line.trim().split(modelParams.delim);

                Integer gidx = fName2IndexMap.get(info[0]);
                cnt++;
                if (gidx == null) {
                    continue;
                }

                float firstOrderWei = Float.parseFloat(info[1]);
                w[0][gidx] = firstOrderWei; // maybe include bias

                int startidx = secondOrderIndexStart + gidx * K[1] * fieldSize;
                for (int f = 0; f < fieldSize * K[1]; f++) {
                    w[0][startidx + f] = Float.parseFloat(info[f + 2]);
                }
            }
        }

        for (int t = 1; t < threadNum; t++) {
            System.arraycopy(w[0], 0, w[t], 0, w[t].length);
        }

        LOG_UTILS.importantInfo("load model finished, old model feature cnt:" + cnt);
        for (int i = 0; i < Math.min(10, w[0].length); i++) {
            LOG_UTILS.verboseInfo("w[" + i + "]=" + w[0][i]);
        }
        LOG_UTILS.verboseInfo("w last=" + w[0][w[0].length - 1]);
    }

    @Override
    protected void handleOtherTrainInfo() throws Mp4jException {
        this.maxFeatureNum = ((FFMCoreData)threadTrainCoreDatas[0]).getMaxFeatureNum();
        LOG_UTILS.importantInfo("train line max feature num:" + maxFeatureNum + ", config max feature dim:" + maxFeatureDim);
    }

    @Override
    protected void handleOtherTestInfo() throws Mp4jException {
        this.maxFeatureNum = ((FFMCoreData)threadTestCoreDatas[0]).getMaxFeatureNum();
        LOG_UTILS.importantInfo("train & test line max feature num:" + maxFeatureNum);
    }

    @Override
    protected void setDim() throws Mp4jException {
        int size = fName2IndexMap.size();
        dim = size + size * fieldSize * K[1];
        LOG_UTILS.importantInfo("dim:" + dim);
    }



    @Override
    public void dumpModel() throws IOException, Mp4jException {
        PrintWriter writer = null;
        PrintWriter dictWriter = null;

        int featureNum = fName2IndexMap.size();
        int avg = featureNum / slaveNum;
        int start = rank * avg;
        int end = (rank + 1) * avg;

        if (rank == slaveNum - 1) {
            end = featureNum;
        }

        LOG_UTILS.verboseInfo("dump from:" + start + ", to:" + end);

        int secondOrderIndexStart = secondOrderIndexStart();
        try {
            String modelPartPath = modelParams.data_path + "/model-" + String.format("%05d", rank);
            String dictPartPath = modelParams.data_path + "_dict/dict-" + String.format("%05d", rank);
            writer = new PrintWriter(fs.getWriter(modelPartPath));
            dictWriter = new PrintWriter(fs.getWriter(dictPartPath));

            for (Map.Entry<String, Integer> entry : fName2IndexMap.entrySet()) {

                if (!entry.getKey().equalsIgnoreCase(modelParams.bias_feature_name)) {

                    int idx = entry.getValue();
                    if (!(idx >= start && idx < end)) {
                        continue;
                    }

                    double firstOrderWei = w[0][idx];

                    int sidx = secondOrderIndexStart + idx * K[1] * fieldSize;
                    StringBuffer sb = new StringBuffer();
                    for (int i = 0; i < K[1] * fieldSize; i++) {
                        sb.append(w[0][sidx + i]);
                        if (i < K[1] * fieldSize - 1) {
                            sb.append(modelParams.delim);
                        }
                    }

                    String str = String.format("%s%s%f%s%s",
                            entry.getKey(), modelParams.delim,
                            firstOrderWei, modelParams.delim,
                            sb.toString());
                    writer.println(str);
                    dictWriter.println(entry.getKey());
                } else {

                    int idx = entry.getValue();
                    if (!(idx >= start && idx < end)) {
                        continue;
                    }

                    double firstOrderWei = w[0][idx];

                    int sidx = secondOrderIndexStart + idx * K[1] * fieldSize;
                    StringBuffer sb = new StringBuffer();
                    for (int i = 0; i < K[1] * fieldSize; i++) {
                        sb.append(w[0][sidx + i]);
                        if (i < K[1] * fieldSize - 1) {
                            sb.append(modelParams.delim);
                        }
                    }

                    writer.println(entry.getKey() + modelParams.delim + firstOrderWei + modelParams.delim +
                            sb.toString());
                }
            }
            LOG_UTILS.importantInfo("model is written to " + modelPartPath);
            LOG_UTILS.importantInfo("model-dict is written to " + dictPartPath);
        } finally {
            if (writer != null) {
                writer.close();
            }
            if (dictWriter != null) {
                dictWriter.close();
            }
        }
    }

    @Override
    public void handleLocalIdx(int tidx, Map<Integer, String> fIndex2NameMap) {
        if (replacedIdx) {
            return;
        }

        CoreData coreData = threadTrainCoreDatas[tidx];
        int x[][] = coreData.x;
        int[] realNum = coreData.realNum;
        int xidx[][] = coreData.xidx;

        for (int k = 0; k < coreData.cursor2d; k++) {
            int lsNumInt = realNum[k];
            int suballMissCnt = 0;
            for (int i = 0; i < lsNumInt; i++) {
                int start = xidx[k][i];
                int end = xidx[k][i + 1];

                xidx[k][i] -= suballMissCnt;
                for (int j = start; j < end; j += 3) {
                    int localIdx = x[k][j];
                    String fname = fIndex2NameMap.get(localIdx);
                    Integer gidx = fName2IndexMap.get(fname);
                    if (gidx != null) {
                        x[k][j] = gidx;
                        x[k][j - suballMissCnt] = x[k][j];
                        x[k][j + 1 - suballMissCnt] = x[k][j + 1];
                        x[k][j + 2 - suballMissCnt] = x[k][j + 2];
                    } else {
                        suballMissCnt += 3;
                    }
                }
            }
            xidx[k][lsNumInt] -= suballMissCnt;
        }
    }

    @Override
    public void replaceFeatureTransform(CoreData coreData) {

        int x[][] = coreData.x;
        int[] realNum = coreData.realNum;
        int xidx[][] = coreData.xidx;
        for (int k = 0; k < coreData.cursor2d; k++) {
            int lsNumInt = realNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                for (int j = xidx[k][i]; j < xidx[k][i + 1]; j += 3) {
                    if (x[k][j] == 0 && coreParams.need_bias) {
                        continue;
                    }
                    CoreData.TransformNode node = transformNodeMap.get(x[k][j]);
                    x[k][j + 1] = Float.floatToRawIntBits(node.transform(Float.intBitsToFloat(x[k][j + 1])));
                }
            }
        }
    }


    public int firstOrderIndexStart() {
        return coreParams.need_bias ? 1 : 0;
    }

    public int secondOrderIndexStart() {
        return fName2IndexMap.size();
    }
}
