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

import com.fenbi.ytklearn.fs.IFileSystem;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.utils.RandomParamsUtils;
import com.typesafe.config.Config;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

/**
 * @author xialong
 */

public class GBSDTDataFlow extends GBMLRDataFlow {
    private List<Double> leafRange;
    public GBSDTDataFlow(IFileSystem fs,
                         Config config,
                         ThreadCommSlave comm,
                         int threadNum,
                         boolean needPyTransform,
                         String pyTransformScript) throws Exception {
        super(fs, config, comm, threadNum, needPyTransform, pyTransformScript);
        this.leafRange = config.getDoubleList("leaf_random_init_range");
        LOG_UTILS.importantInfo("leaf_random_init_range:" + leafRange);
    }

    @Override
    protected void setDim() throws Mp4jException {
        dim = K + fName2IndexMap.size() * (K - 1);
        LOG_UTILS.importantInfo("dim:" + dim);
    }

    @Override
    public void initW(int tidx) throws Mp4jException {
        //Random rand = new Random(getSeed());
        RandomParamsUtils randomParamsUtils = new RandomParamsUtils(randomParams, getSeed());
        for (int i = 0; i < dim; i++) {
            w[tidx][i] = (float) randomParamsUtils.next();
        }

        for (int i = 0; i < K; i++) {
            w[tidx][i] = (float) randomParamsUtils.nextUniform(leafRange.get(0), leafRange.get(1));
        }

        if (coreParams.need_bias) {
            for (int i = K; i < 2 * K - 1; i++) {
                w[tidx][i] = 0.0f;
            }
        }

//        for (int t = 1; t < threadNum; t++) {
//            System.arraycopy(w[0], 0, w[t], 0, w[t].length);
//        }

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Math.min(2 * K - 1, w[tidx].length); i++) {
            sb.append("w[").append(tidx).append("][").append(i).append("]=").append(w[tidx][i]).append("\n");
        }
        LOG_UTILS.verboseInfo("init w:" + sb, false);
    }

    @Override
    protected void loadModel() throws IOException, Mp4jException, InterruptedException {

        boolean haveOldModel = true;
        // check model data path
        if (!fs.exists(modelParams.data_path)) {
            LOG_UTILS.importantInfo("old model doesn't exist, new model...");
            haveOldModel = false;
        }

        // load model info
        String modelInfoPath = modelParams.data_path + "/tree-info";
        if (!fs.exists(modelInfoPath)) {
            LOG_UTILS.importantInfo("have no model info data, old model doesn't exist, new model..." + modelInfoPath);
            haveOldModel = false;
        }

        // alloc w
        w = new float[threadNum][];
        precision = new float[threadNum][];
        for (int p = 0; p < threadNum; p++) {
            w[p] = new float[dim];
        }

        if (!haveOldModel || (!modelParams.continue_train && !commonParams.lossParams.just_evaluate)) {
            for (int t = 0; t < threadNum; t++) {
                initW(t);
            }

            // random sample and feature
            for (int t = 0; t < threadNum; t++) {
                ((GBMLRCoreData)threadTrainCoreDatas[t]).featureMask = new BitSet(fName2IndexMap.size());
                randomNextSample((GBMLRCoreData)threadTrainCoreDatas[t], randomSampleRate, randomFeatureRate);
            }
            return;
        }

        BufferedReader infoReader = null;
        try {
            String line;
            List<String> lineList = new ArrayList<>();
            infoReader = new BufferedReader(fs.getReader(modelInfoPath));
            while((line = infoReader.readLine()) != null) {
                if (line.trim().length() > 0) {
                    lineList.add(line.trim());
                }
            }
            if (lineList.size() != 4) {
                throw new IOException("model info must have 4 lines!");
            }

            int oldK = Integer.parseInt(lineList.get(0).split(":")[1]);
            int oldTreeNum = Integer.parseInt(lineList.get(1).split(":")[1]);
            finishedTreeNum = Integer.parseInt(lineList.get(2).split(":")[1]);
            double oldBias = Double.parseDouble(lineList.get(3).split(":")[1]);
            if (K != oldK) {
                throw new IOException("model info K != config K" + ", model info K:" + oldK + ", config K:" + K);
            }

            if (oldTreeNum != treeNum) {
                LOG_UTILS.importantInfo("[WARNING] old tree num:" + oldTreeNum + " != tree num:" + treeNum);
            }

            if (finishedTreeNum >= treeNum && !commonParams.lossParams.just_evaluate) {
                LOG_UTILS.importantInfo("finished tree num:" + finishedTreeNum + " >= " + "tree num:" + treeNum + ", finished directly!");
                return;
            }

            LOG_UTILS.importantInfo("load old model info, K:" + oldK +
                    ", old tree num:" + oldTreeNum +
                    ", tree num:" + treeNum +
                    ", finished tree num:" + finishedTreeNum +
                    ", old bias:" + oldBias +
                    ", uniform base score:" + uniformBaseScore);
        } finally {
            if (infoReader != null) {
                infoReader.close();
            }
        }

        // load finished tress
        for (int tree = 0; tree < finishedTreeNum + 1; tree ++) {

            if (commonParams.lossParams.just_evaluate) {
                if (tree >= finishedTreeNum) {
                    // random sample and feature
                    for (int t = 0; t < threadNum; t++) {
                        ((GBMLRCoreData)threadTrainCoreDatas[t]).featureMask = new BitSet(fName2IndexMap.size());
                        randomNextSample((GBMLRCoreData)threadTrainCoreDatas[t], 1.0, 1.0);
                    }
                    LOG_UTILS.importantInfo("just evaluate, only load finished model!");
                    return;
                }
            }

            if (tree == finishedTreeNum) {
                if (!fs.exists(modelParams.data_path + "/tree-" + String.format("%05d", tree))) {
                    for (int t = 0; t < threadNum; t++) {
                        initW(t);
                    }

                    // random sample and feature
                    for (int t = 0; t < threadNum; t++) {
                        ((GBMLRCoreData)threadTrainCoreDatas[t]).featureMask = new BitSet(fName2IndexMap.size());
                        randomNextSample((GBMLRCoreData)threadTrainCoreDatas[t], randomSampleRate, randomFeatureRate);
                    }
                    LOG_UTILS.importantInfo("unfinished tree not exited!");
                    return;
                } else {
                    // random sample and feature
                    for (int t = 0; t < threadNum; t++) {
                        ((GBMLRCoreData)threadTrainCoreDatas[t]).featureMask = new BitSet(fName2IndexMap.size());
                        randomNextSample((GBMLRCoreData)threadTrainCoreDatas[t], randomSampleRate, randomFeatureRate);
                    }
                    LOG_UTILS.importantInfo("unfinished tree existed! will be readed ...");
                }
            }

            List<String> files = fs.recurGetPaths(Arrays.asList(modelParams.data_path +
                    "/tree-" + String.format("%05d", tree)));

            int cnt = 0;
            for (String  file : files) {
                BufferedReader reader = null;
                try {
                    reader = new BufferedReader(fs.getReader(file));
                    LOG_UTILS.importantInfo("loading old w params:" + file);
                    // skip first, second line
                    String line = reader.readLine();
                    if (K != Integer.parseInt(line.split(":")[1])) {
                        throw new IOException("old model k != config's K = " + K);
                    }
                    LOG_UTILS.importantInfo(line);
                    line = reader.readLine();
                    String []leafInfo = line.split(modelParams.delim);
                    for (int p = 0; p < K; p++) {
                        float leafWei = Float.parseFloat(leafInfo[p]);
                        for (int t = 0; t < threadNum; t++) {
                            w[t][p] = leafWei;
                        }
                    }
                    LOG_UTILS.importantInfo(line);

                    while ((line = reader.readLine()) != null) {
                        if (line.trim().length() == 0) {
                            LOG_UTILS.error("invalid model line:" + line);
                            continue;
                        }
                        String[] info = line.trim().split(modelParams.delim);

                        if (info.length < 2) {
                            LOG_UTILS.error("invalid model line:" + line);
                            continue;
                        }

                        Integer gidx = fName2IndexMap.get(info[0]);
                        cnt++;
                        if (gidx == null) {
                            continue;
                        }

                        gidx = K + gidx * (K - 1);
                        for (int k = 0; k < K - 1; k++) {
                            float wei = Float.parseFloat(info[1 + k]);
                            for (int t = 0; t < threadNum; t++) {
                                w[t][gidx + k] = wei;
                            }
                        }
                    }
                }  finally {
                    if (reader != null)
                        try {
                            reader.close();
                        } catch (IOException e) {
                            LOG.error("load model data path error!", e);
                        }
                }
            }

//            // replace local index
//            if (!replacedIdx && !coreParams.need_dict) {
//                for (int p = 0; p < threadNum; p++) {
//                    handleLocalIdx(p, fIndex2NameMap);
//                }
//                replacedIdx = true;
//            }

            int bound = commonParams.lossParams.just_evaluate ? finishedTreeNum - 1 : finishedTreeNum;
            if (tree < bound) {
                LOG_UTILS.importantInfo("accumulate tree:" + tree + "....");
                Thread []threads = new Thread[threadNum];
                for (int t = 0; t < threadNum; t++) {
                    final int tidx = t;
                    threads[t] = new Thread() {
                        @Override
                        public void run() {
                            GBMLRCoreData gbmlrCoreData = (GBMLRCoreData)threadTrainCoreDatas[tidx];
                            accumulate(gbmlrCoreData, gbmlrCoreData.featureMask, false, w[tidx], learningRate, K);

                            BitSet featureMask = ((GBMLRCoreData)threadTrainCoreDatas[0]).featureMask;
                            if (needTest) {
                                GBMLRCoreData gbmlrCoreDataTest = (GBMLRCoreData)threadTestCoreDatas[tidx];
                                accumulate(gbmlrCoreDataTest, featureMask, false, w[tidx], learningRate, K);

                            }
                        }
                    };
                    threads[t].start();
                }

                for (int t = 0; t < threadNum; t++) {
                    threads[t].join();
                }
//                for (int t = 0; t < threadTrainCoreDatas.length; t++) {
//                    GBMLRCoreData gbmlrCoreData = (GBMLRCoreData)threadTrainCoreDatas[t];
//                    accumulate(gbmlrCoreData, gbmlrCoreData.featureMask, false, w[t], learningRate, K);
//                }
//
//                BitSet featureMask = ((GBMLRCoreData)threadTrainCoreDatas[0]).featureMask;
//                if (needTest) {
//                    for (int t = 0; t < threadTestCoreDatas.length; t++) {
//                        GBMLRCoreData gbmlrCoreData = (GBMLRCoreData)threadTestCoreDatas[t];
//                        accumulate(gbmlrCoreData, featureMask, false, w[t], learningRate, K);
//                    }
//                }
            }

            LOG_UTILS.importantInfo("tree:" + tree + ", load model finished, old model feature cnt:" + cnt);
            StringBuilder sbo = new StringBuilder();
            for (int i = 0; i < 2 * K - 1; i++) {
                sbo.append("w[").append(i).append("]=").append(w[0][i]).append(",");
            }
            LOG_UTILS.verboseInfo("tree:" + tree + " ,old w:" + sbo);

        }

    }

    @Override
    public void accumulate(GBMLRCoreData coreData, BitSet featureMask, boolean current, float []w, double learningRate, int K) {

        int stride = K - 1;
        int vstart = K;
        double wx[] = new double[K - 1];
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
                // softmax
                double maxXv = wx[0];
                for (int j = 1; j < stride; j++) {
                    if (wx[j] > maxXv) {
                        maxXv = wx[j];
                    }
                }
                double esum = 0.0;
                for (int j = 0; j < stride; j++) {
                    wx[j] = Math.exp(wx[j] - maxXv);
                    esum += wx[j];
                }

                // g(x), fx
                double gk_1 = 1.0;
                double fx = 0.0;
                double inv = 1.0 / (Math.exp(-maxXv) + esum);
                for (int j = 0; j < stride; j++) {
                    wx[j] = wx[j] * inv;
                    gk_1 -= wx[j];
                    fx += wx[j] * w[j];
                }
                fx += gk_1 * w[vstart - 1];

                coreData.z[k][i] += learningRate * fx;
            }
        }
    }

    @Override
    public void dumpModel() throws IOException, Mp4jException {
        PrintWriter writer = null;
        PrintWriter dictWriter = null;

        int featureNum = (dim - K) / (K - 1);
        int avg = featureNum / slaveNum;
        int start = rank * avg;
        int end = (rank + 1) * avg;

        if (rank == slaveNum - 1) {
            end = featureNum;
        }

        GBMLRCoreData coreData = (GBMLRCoreData)threadTrainCoreDatas[0];

        LOG_UTILS.verboseInfo("dump from:" + start + ", to:" + end + ", finished tree num:" + finishedTreeNum);
        int curtreeid = finishedTreeNum;
        String modelDataPath = modelParams.data_path;
        String modelDataDelim = modelParams.delim;
        try {

            String modelPartPath = modelDataPath + "/tree-" + String.format("%05d/model-", curtreeid) + String.format("%05d", rank);
            String dictPartPath = modelDataPath + "_dict/dict-" + String.format("%05d", rank);
            writer = new PrintWriter(fs.getWriter(modelPartPath));
            dictWriter = new PrintWriter(fs.getWriter(dictPartPath));

            writer.println("k:" + K);
            StringBuffer leafsb = new StringBuffer();
            for (int p = 0; p < K - 1; p++) {
                leafsb.append(w[0][p]).append(modelDataDelim);
            }
            leafsb.append(w[0][K - 1]);
            writer.println(leafsb.toString());
            for (Map.Entry<String, Integer> entry : fName2IndexMap.entrySet()) {

                if (!entry.getKey().equalsIgnoreCase(coreParams.bias_feature_name)) {

                    int idx = entry.getValue();
                    if (!(idx >= start && idx < end)) {
                        continue;
                    }

                    StringBuilder sb = new StringBuilder();
                    if (coreData.featureMask.get(idx)) {
                        int realidx = K + idx * (K - 1);
                        for (int i = 0; i < K - 1; i++) {
                            sb.append(w[0][realidx + i]).append(modelDataDelim);
                        }
                    } else {
                        for (int i = 0; i < K - 1; i++) {
                            sb.append(0.0).append(modelDataDelim);
                        }
                    }

                    String str = String.format("%s%s%s",
                            entry.getKey(), modelDataDelim,
                            sb);
                    writer.println(str);
                    dictWriter.println(entry.getKey());
                } else {


                    int idx = entry.getValue();
                    if (!(idx >= start && idx < end)) {
                        continue;
                    }
                    StringBuilder sb = new StringBuilder();
                    int realidx = K + idx * (K - 1);
                    for (int i = 0; i < K - 1; i++) {
                        sb.append(w[0][realidx + i]).append(modelDataDelim);
                    }
                    writer.println(entry.getKey() + modelDataDelim + sb);
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

        // dump model info
        dumpModelInfo(rank);
    }



}
