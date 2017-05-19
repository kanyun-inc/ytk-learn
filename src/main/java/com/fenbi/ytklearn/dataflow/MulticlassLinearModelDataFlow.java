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

import com.fenbi.ytklearn.exception.YtkLearnException;
import com.fenbi.ytklearn.feature.FeatureHash;
import com.fenbi.ytklearn.fs.IFileSystem;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.typesafe.config.Config;
import lombok.Getter;
import org.apache.commons.lang.ArrayUtils;
import org.python.core.PyFunction;

import java.io.*;
import java.util.*;

/**
 * @author xialong
 */

public class MulticlassLinearModelDataFlow extends ContinuousDataFlow {
    @Getter
    private final int K;

    public MulticlassLinearModelDataFlow(IFileSystem fs,
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
        this.K = config.getInt("k");
        LOG_UTILS.importantInfo("K:" + K);
    }

    public static class MulticlassLinearCoreData extends ContinuousCoreData {
        private int K;
        public MulticlassLinearCoreData(ThreadCommSlave comm,
                                        CoreParams coreParams,
                                        IFeatureMap featureMap,
                                        PyFunction pyTransformFunc,
                                        boolean needPyTransform,
                                        FeatureHash featureHash,
                                        int K) {
            super(comm, coreParams, featureMap, pyTransformFunc, needPyTransform, featureHash);
            this.K = K;
        }


        @Override
        public void initAssistData() {

            LINE_LIST = new ArrayList<>();
            LINE_LIST.add("temp");
            TMP_XIDX = new int[MAX_1D_LEN + 1];
            int size = Math.min(Integer.MAX_VALUE, MAX_1D_LEN * K);
            TMP_Y = new float[size];
            TMP_WEIGHT = new float[MAX_1D_LEN];
        }

        @Override
        protected void updateY() throws Exception {
            int yidx = count * K;
            double sum = 0.0;
            for (int p = 0; p < K; p++) {
                TMP_Y[yidx + p] = label[p];
                sum += TMP_Y[yidx + p];
            }

            if (Math.abs(sum - 1.0f) > 0.01) {
                throw new Exception("all label sum must equal 1.0!");
            }
        }



        @Override
        protected boolean yExtract(String line, String[] info) throws Exception {
            String []linfo = info[1].split(coreParams.y_delim);

            if (linfo.length != K && linfo.length != 1) {
                throw new Exception("label num must = " + K + ", or = 1, line:" + line);
            }

            if (linfo.length == 1) {
                for (int i = 0; i < K; i++) {
                    label[i] = 0;
                }
                int clazz = Integer.parseInt(linfo[0]);
                if (clazz >= K) {
                    throw new YtkLearnException("multi classification label must in range [0,K-1]!\n" + line);
                }
                label[clazz] = 1.0f;
            } else {
                for (int i = 0; i < K; i++) {
                    label[i] = Float.parseFloat(linfo[i]);
                }
            }


            if (coreParams.needYStat) {
                labelIdx = -1;
                for (int i = 0; i < K; i++) {
                    if (label[i] == 1.0) {
                        labelIdx = i;
                    }
                }
            }

            if (!coreParams.needYSampling) {
                return true;
            }


            if (labelIdx == -1) {
                throw new Exception("label error for y sampling! line:" + line);
            }
            float rate = coreParams.ySampling[labelIdx];
            if (rate <= 1.0f) {
                wei *= (1.0f / rate);
            } else {
                wei *= rate;
            }
            return rand.nextFloat() <= rate;
        }

        @Override
        protected void exceed1DHandle() throws Mp4jException {
            TMP_XIDX[count] = xindex;
            int localnum = realNum[cursor2d];

            if (localnum != count) {
                LOG_UTILS.verboseInfo(loadingPrefix + "----error! localnum:" + localnum + ", count:" + count, false);
            }
            xidx[cursor2d] = new int[localnum + 1];
            System.arraycopy(TMP_XIDX, 0, xidx[cursor2d], 0, localnum + 1);

            y[cursor2d] = new float[localnum * K];
            System.arraycopy(TMP_Y, 0, y[cursor2d], 0, localnum * K);

            weight[cursor2d] = new float[localnum];
            System.arraycopy(TMP_WEIGHT, 0, weight[cursor2d], 0, localnum);

            predict[cursor2d] = new float[localnum * K];

            xindex = 0;
            count = 0;
            cursor2d++;
        }

        @Override
        protected void lastSampleHandle() throws Mp4jException {
            TMP_XIDX[count] = xindex;
            int localnum = realNum[cursor2d];
            xidx[cursor2d] = new int[localnum + 1];
            System.arraycopy(TMP_XIDX, 0, xidx[cursor2d], 0, localnum + 1);

            y[cursor2d] = new float[localnum * K];
            System.arraycopy(TMP_Y, 0, y[cursor2d], 0, localnum * K);

            weight[cursor2d] = new float[localnum];
            System.arraycopy(TMP_WEIGHT, 0, weight[cursor2d], 0, localnum);

            predict[cursor2d] = new float[localnum * K];

            LOG_UTILS.verboseInfo(loadingPrefix + "finished read data, cursor2d:" + cursor2d +
                    ", real num:" + ArrayUtils.toString(Arrays.copyOfRange(realNum, 0, cursor2d + 1)) +
                    ", weight sum:" + ArrayUtils.toString(Arrays.copyOfRange(weightNum, 0, cursor2d + 1)), false);
        }
    }

    @Override
    protected void setDim() throws Mp4jException {
        dim = fName2IndexMap.size() * (K - 1);
        LOG_UTILS.importantInfo("dim:" + dim);
    }

    @Override
    protected boolean needLaplace() {
        return false;
    }

    @Override
    protected CoreData getCoreData() {
        return new MulticlassLinearCoreData(comm, coreParams, featureMap, pyTransformFunc, needPyTransform, featureHash, K);
    }

    @Override
    protected void loadModel() throws IOException, Mp4jException {
        // 初始化为0
        w = new float[threadNum][];
        w[0] = new float[dim];
        precision = new float[threadNum][];

        for (int i = 0; i < dim; i++) {
            w[0][i] = 0;
        }

        for (int t = 1; t < threadNum; t++) {
            w[t] = new float[dim];
            System.arraycopy(w[0], 0, w[t], 0, dim);
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
                    LOG_UTILS.importantInfo("[ERROR] invalid model line:" + line);
                    continue;
                }
                String []info = line.trim().split(modelParams.delim);

                if (info.length < 2) {
                    LOG_UTILS.importantInfo("[ERROR] invalid model line:" + line);
                    continue;
                }

                Integer idx = fName2IndexMap.get(info[0]);
                cnt ++;
                if (idx == null) {
                    continue;
                }

                int gidx = idx * (K - 1);
                for (int k = 0; k < K - 1; k++) {
                    w[0][gidx + k] = Float.parseFloat(info[1 + k]);
                }
            }
        }

        for (int t = 1; t < threadNum; t++) {
            System.arraycopy(w[0], 0, w[t], 0, dim);
        }

        LOG_UTILS.importantInfo("load model finished, old model feature cnt:" + cnt);
        for (int i = 0; i < Math.min(10, w[0].length); i++) {
            LOG_UTILS.verboseInfo("w[" + i + "]=" + w[0][i]);
        }
        LOG_UTILS.verboseInfo("w last=" + w[0][w[0].length - 1]);
    }

    @Override
    protected void handleOtherTrainInfo() {

    }

    @Override
    protected void handleOtherTestInfo() {

    }

    @Override
    protected int getYnum() {
        return K;
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

        String modelDataPath = modelParams.data_path;
        String modelDataDelim = modelParams.delim;
        try {
            String modelPartPath = modelDataPath + "/model-" + String.format("%05d", rank);
            String dictPartPath = modelDataPath + "_dict/dict-" + String.format("%05d", rank);
            writer = new PrintWriter(fs.getWriter(modelPartPath));
            dictWriter = new PrintWriter(fs.getWriter(dictPartPath));

            for (Map.Entry<String, Integer> entry : fName2IndexMap.entrySet()) {

                int idx = entry.getValue();
                if (!(idx >= start && idx < end)) {
                    continue;
                }
                StringBuffer sb = new StringBuffer();
                int gidx = idx * (K - 1);
                for (int i = 0; i < K - 2; i++) {
                    sb.append(w[0][gidx + i]).append(modelDataDelim);
                }
                sb.append(w[0][gidx + K - 2]);

                String str = String.format("%s%s%s",
                        entry.getKey(), modelDataDelim,
                        sb.toString());
                writer.println(str);
                dictWriter.println(entry.getKey());

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
}
