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

import com.fenbi.ytklearn.feature.FeatureHash;
import com.fenbi.ytklearn.fs.IFileSystem;
import com.fenbi.ytklearn.loss.LossFunctions;
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
public class GBMLRDataFlow extends ContinuousDataFlow {

    protected int K;
    //@Getter
    //protected int seed;

    protected int treeNum;
    protected volatile int finishedTreeNum = 0;
    protected float uniformBaseScore;
    protected boolean sampleDepdtBaseScore;

    protected double learningRate;
    protected double randomSampleRate;
    protected double randomFeatureRate;

    protected RandomParams randomParams;
    protected boolean gradientBoosting;

    public enum Type {
        GB("gradient_boosting"),
        RF("random_forest");
        private String name;
        private Type(String name) {
            this.name = name;
        }

        @Override
        public String toString() {
            return name;
        }

        public static Type getType(String name) throws Exception {
            for (Type type : values()) {
                if (name.equalsIgnoreCase(type.toString())) {
                    return type;
                }
            }
            throw new Exception("unknow type:" + name + ", must be gradient_boosting or random_forest");
        }
    }
    protected Type type;

    @Data
    public static class GBMLRCoreData extends ContinuousCoreData {
        public float [][]z;
        public BitSet []randMask;
        public float []TMP_Z;
        public BitSet featureMask;

        public float uniformBaseScore;
        public boolean sampleDepdtBaseScore;

        public GBMLRCoreData(ThreadCommSlave comm,
                             CoreParams coreParams,
                             IFeatureMap featureMap,
                             PyFunction pyTransformFunc,
                             boolean needPyTransform,
                             FeatureHash featureHash,
                             float uniformBaseScore,
                             boolean sampleDepdtBaseScore) {
            super(comm, coreParams, featureMap, pyTransformFunc, needPyTransform, featureHash);
            this.z = new float[MAX_2D_LEN][];
            this.randMask = new BitSet[MAX_2D_LEN];
            this.uniformBaseScore = uniformBaseScore;
            this.sampleDepdtBaseScore = sampleDepdtBaseScore;


        }

        @Override
        public void initAssistData() {
            super.initAssistData();
            this.TMP_Z = new float[MAX_1D_LEN];
        }

        @Override
        protected void otherHandle(String line, String[] info) {
            TMP_Z[count] = uniformBaseScore;
            if (sampleDepdtBaseScore) {
                TMP_Z[count] += coreParams.lossFunction.pred2Score(Float.parseFloat(info[3]));
            }
        }

        protected void exceed1DHandle() throws Mp4jException {
            TMP_XIDX[count] = xindex;
            int localnum = realNum[cursor2d];

            if (localnum != count) {
                LOG_UTILS.importantInfo(loadingPrefix + "----error! localnum:" + localnum + ", count:" + count, false);
            }
            xidx[cursor2d] = new int[localnum + 1];
            System.arraycopy(TMP_XIDX, 0, xidx[cursor2d], 0, localnum + 1);

            y[cursor2d] = new float[localnum];
            System.arraycopy(TMP_Y, 0, y[cursor2d], 0, localnum);

            weight[cursor2d] = new float[localnum];
            System.arraycopy(TMP_WEIGHT, 0, weight[cursor2d], 0, localnum);

            z[cursor2d] = new float[localnum];
            System.arraycopy(TMP_Z, 0, z[cursor2d], 0, localnum);

            predict[cursor2d] = new float[localnum];

            randMask[cursor2d] = new BitSet(localnum);

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

            y[cursor2d] = new float[localnum];
            System.arraycopy(TMP_Y, 0, y[cursor2d], 0, localnum);

            weight[cursor2d] = new float[localnum];
            System.arraycopy(TMP_WEIGHT, 0, weight[cursor2d], 0, localnum);

            z[cursor2d] = new float[localnum];
            System.arraycopy(TMP_Z, 0, z[cursor2d], 0, localnum);

            predict[cursor2d] = new float[localnum];

            randMask[cursor2d] = new BitSet(localnum);

            LOG_UTILS.verboseInfo(loadingPrefix + "finished read data, cursor2d:" + cursor2d +
                    ", real num:" + ArrayUtils.toString(Arrays.copyOfRange(realNum, 0, cursor2d + 1)) +
                    ", weight sum:" + ArrayUtils.toString(Arrays.copyOfRange(weightNum, 0, cursor2d + 1)), false);
        }

        @Override
        protected void exceed2DHandle() {
            super.exceed2DHandle();
            float [][]new_z = new float[x.length * 2][];

            for (int i = 0; i < x.length; i++) {
                new_z[i] = z[i];
            }
            z = new_z;

        }
    }


    public GBMLRDataFlow(IFileSystem fs,
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
        //this.seed = config.getInt("seed");
        this.randomSampleRate = config.getDouble("instance_sample_rate");
        this.randomFeatureRate = config.getDouble("feature_sample_rate");
        this.uniformBaseScore = (float) LossFunctions.createLossFunction(commonParams.lossParams.loss_function).
                                pred2Score(config.getDouble("uniform_base_prediction"));
        this.sampleDepdtBaseScore = config.getBoolean("sample_dependent_base_prediction");
        this.treeNum = config.getInt("tree_num");
        this.learningRate = config.getDouble("learning_rate");
        this.randomParams = new RandomParams(config, "");

        this.type = Type.getType(config.getString("type"));
        if (type == Type.RF) {
            this.learningRate = 1.0;
        }

        CheckUtils.check(K >= 2, "K:%d must >= 2", K);

        LOG_UTILS.importantInfo("K:" + K);
        //comm.LOG_UTILS.importantInfo("seed:" + seed);
        LOG_UTILS.importantInfo("random:" + randomParams);
        LOG_UTILS.importantInfo("instance_sample_rate:" + randomSampleRate);
        LOG_UTILS.importantInfo("feature_sample_rate:" + randomFeatureRate);
        LOG_UTILS.importantInfo("uniform_base_prediction:" + uniformBaseScore);
        LOG_UTILS.importantInfo("tree_num:" + treeNum);
        LOG_UTILS.importantInfo("learning_rate:" + learningRate);
        LOG_UTILS.importantInfo("type:" + type);
    }

    @Override
    protected void setDim() throws Mp4jException {
        dim = fName2IndexMap.size() * (2 * K - 1);
        LOG_UTILS.importantInfo("dim:" + dim);
    }


    @Override
    protected boolean needLaplace() {
        return false;
    }

    @Override
    protected CoreData getCoreData() {
        return new GBMLRCoreData(comm, coreParams, featureMap, pyTransformFunc, needPyTransform, featureHash, uniformBaseScore, sampleDepdtBaseScore);
    }



    public int getSeed() throws Mp4jException {
        int newseed = 99999 + finishedTreeNum * randomParams.seed;
        LOG_UTILS.importantInfo("new seed:" + newseed + ", finished tree num:" + finishedTreeNum);
        return newseed;
    }

    public void initW(int tidx) throws Mp4jException {
        //Random rand = new Random(getSeed());
        RandomParamsUtils randomParamsUtils = new RandomParamsUtils(randomParams, getSeed());
        // 初始化为[-0.5, 0.5]的均匀分布, bias初始化为0
        //sigma = new double[threadNum][];
        if (modelParams.need_bias) {
            for (int i = 0; i < 2 * K - 1; i++) {
                w[tidx][i] = 0.0f;
            }
        }

        int idxstart = modelParams.need_bias ? 2 * K - 1 : 0;
        for (int i = idxstart; i < dim; i++) {
            w[tidx][i] = (float) randomParamsUtils.next();
        }

//        for (int t = 1; t < threadNum; t++) {
//            System.arraycopy(w[0], 0, w[t], 0, w[t].length);
//        }

        StringBuilder sb = new StringBuilder();
        int start = coreParams.need_bias ? 2 * K - 1 : 0;
        int end = coreParams.need_bias ? 2 * (2 * K - 1) : 2 * K - 1;
        for (int i = start; i < end; i++) {
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
            double oldUniformBaseScore = Double.parseDouble(lineList.get(3).split(":")[1]);
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

            if (oldUniformBaseScore != uniformBaseScore) {
                throw new IOException("old uniform_base_prediction != uniform_base_prediction, old:" + oldUniformBaseScore + ", new:" + uniformBaseScore);
            }

            LOG_UTILS.importantInfo("load old model info, K:" + oldK +
                    ", old tree num:" + oldTreeNum +
                    ", tree num:" + treeNum +
                    ", finished tree num:" + finishedTreeNum +
                    ", old uniform base score:" + oldUniformBaseScore +
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
                    // skip first line
                    String line = reader.readLine();
                    if (K != Integer.parseInt(line.split(":")[1])) {
                        throw new IOException("old model k != config's K = " + K);
                    }
                    LOG_UTILS.importantInfo(line);
                    while ((line = reader.readLine()) != null) {
                        if (line.trim().length() == 0) {
                            LOG_UTILS.importantInfo("[ERROR] invalid model line:" + line);
                            continue;
                        }
                        String []info = line.trim().split(modelParams.delim);

                        if (info.length < 2) {
                            LOG_UTILS.importantInfo("[ERROR] invalid model line:" + line);
                            continue;
                        }

                        Integer gidx = fName2IndexMap.get(info[0]);
                        cnt ++;
                        if (gidx == null) {
                            continue;
                        }

                        gidx *= (2 * K - 1);
                        for (int k = 0; k < 2 * K - 1; k++) {
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
//            if (!coreParams.need_dict && !replacedIdx) {
//                for (int p = 0; p < threadNum; p++) {
//                    handleLocalIdx(p, fIndex2NameMap);
//                }
//                replacedIdx = true;
//            }

            // replace feature transform
            if (needFeatureTransform && !trainReplacedFeatureTransform) {
                for (int p = 0; p < threadNum; p++) {
                    replaceFeatureTransform(threadTrainCoreDatas[p]);
                }
                trainReplacedFeatureTransform = true;
            }

            if (needFeatureTransform && needTest && !testReplacedFeatureTransform) {
                for (int p = 0; p < threadNum; p++) {
                    replaceFeatureTransform(threadTestCoreDatas[p]);
                }
                testReplacedFeatureTransform = true;
            }

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

    public void accumulate(GBMLRCoreData coreData, BitSet featureMask, boolean current, float []w, double learningRate, int K) {

        double wx[] = new double[2 * K - 1];
        int stride = 2 * K - 1;
        int vstride = K - 1;
        for (int k = 0; k < coreData.cursor2d; k++) {
            int lsNumInt = coreData.realNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                // xv, xw
                for (int p = 0; p < stride; p++) {
                    wx[p] = 0.0;
                }
                for (int j = coreData.xidx[k][i]; j < coreData.xidx[k][i + 1]; j+=2) {
                    double fval = Float.intBitsToFloat(coreData.x[k][j+1]);
                    int idx = coreData.x[k][j] * stride;
                    int pstart = (!current || featureMask.get(coreData.x[k][j])) ? 0 : vstride;
                    for (int p = pstart; p < stride; p++) {
                        wx[p] += w[idx + p] * fval;
                    }
                }
                // softmax
                double maxXv = wx[0];
                for (int j = 1; j < vstride; j++) {
                    if (wx[j] > maxXv) {
                        maxXv = wx[j];
                    }
                }
                double esum = 0.0;
                for (int j = 0; j < vstride; j++) {
                    wx[j] = Math.exp(wx[j] - maxXv);
                    esum += wx[j];
                }

                // g(x), fx
                double gk_1 = 1.0;
                double fx = 0.0;
                double inv = 1.0 / (Math.exp(-maxXv) + esum);
                for (int j = 0; j < vstride; j++) {
                    wx[j] = wx[j] * inv;
                    gk_1 -= wx[j];
                    fx += wx[j] * wx[vstride + j];
                }
                fx += gk_1 * wx[stride - 1];

                coreData.z[k][i] += learningRate * fx;
            }
        }
    }

    public void randomNextSample(GBMLRCoreData coreData, double randomSampleRate, double randomFeatureRate) throws Mp4jException {

        // sample random
        Random sampleRand = new Random();
        for (int k = 0; k < coreData.cursor2d; k++) {
            int lsNumInt = coreData.realNum[k];
            coreData.randMask[k].clear();
            for (int i = 0; i < lsNumInt; i++) {
                if (sampleRand.nextDouble() <= randomSampleRate) {
                    coreData.randMask[k].set(i);
                }
            }
        }

        // feature random
        Random frand = new Random(getSeed());
        coreData.featureMask.clear();
        int bstart = 0;
        if (coreParams.need_bias) {
            bstart = 1;
            coreData.featureMask.set(0);
        }

        for (int idx = bstart; idx < fName2IndexMap.size(); idx++) {
            if (frand.nextDouble() <= randomFeatureRate) {
                coreData.featureMask.set(idx);
            }
        }

        if (coreData.cursor2d > 0) {
            LOG_UTILS.importantInfo("finished tree num:" + finishedTreeNum);
            LOG_UTILS.verboseInfo("sample random mask:" + coreData.randMask[0].get(0, 30).toString());
            LOG_UTILS.verboseInfo("feature random mask:" + coreData.featureMask.get(0, 30).toString());
        } else {
            LOG_UTILS.importantInfo("this thread have no data, so have no mask!");
        }


    }


    @Override
    protected void handleOtherTrainInfo() {


    }

    @Override
    protected void handleOtherTestInfo() {

    }

    @Override
    public void dumpModel() throws IOException, Mp4jException {
        PrintWriter writer = null;
        PrintWriter dictWriter = null;

        int featureNum = dim / (2 * K - 1);
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
            for (Map.Entry<String, Integer> entry : fName2IndexMap.entrySet()) {

                if (!entry.getKey().equalsIgnoreCase(coreParams.bias_feature_name)) {
                    int idx = entry.getValue();
                    if (!(idx >= start && idx < end)) {
                        continue;
                    }

                    StringBuilder sb = new StringBuilder();
                    int realidx = idx * (2 * K - 1);
                    if (coreData.featureMask.get(idx)) {
                        for (int i = 0; i < 2 * K - 1; i++) {
                            sb.append(w[0][realidx + i]).append(modelDataDelim);
                        }
                    } else {
                        for (int i = 0; i < K - 1; i++) {
                            sb.append(0.0).append(modelDataDelim);
                        }
                        for (int i = K - 1; i < 2 * K - 1; i++) {
                            sb.append(w[0][realidx + i]).append(modelDataDelim);
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
                    int realidx = idx * (2 * K - 1);
                    for (int i = 0; i < 2 * K - 1; i++) {
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

    public void dumpModelInfo(int rank) throws IOException, Mp4jException {
        if (rank != 0) return;

        String modelInfoPath = modelParams.data_path + "/tree-info";
        LOG_UTILS.importantInfo("begin dumping tree info, k:" + K +
                ", tree_num:" + treeNum +
                ", finished_tree_num:" + finishedTreeNum +
                ", uniform_base_prediction:" + uniformBaseScore);
        PrintWriter writer = null;
        try {
            writer = new PrintWriter(fs.getWriter(modelInfoPath));
            writer.println("K:" + K);
            writer.println("tree_num:" + treeNum);
            writer.println("finished_tree_num:" + finishedTreeNum);
            writer.println("uniform_base_prediction:" + uniformBaseScore);
        } finally {
            if (writer != null) {
                writer.close();
            }
        }
    }

    public void incrFinishedTreeNum() throws Mp4jException {
        finishedTreeNum ++;
        LOG_UTILS.importantInfo("finished num:" + finishedTreeNum);
    }
}
