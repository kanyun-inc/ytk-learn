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
import com.fenbi.ytklearn.param.RandomParams;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.utils.RandomParamsUtils;
import com.typesafe.config.Config;
import lombok.Data;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

/**
 * @author xialong
 */

@Data
public class FMModelDataFlow extends ContinuousDataFlow {
    private int K[];
    //private int seed;
    private boolean needFirstOrder;
    private boolean needSecondOrder;
    private boolean biasNeedLatentFactor;

    private RandomParams randomParams;

    public FMModelDataFlow(IFileSystem fs,
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
        randomParams = new RandomParams(config, "");
        biasNeedLatentFactor = config.getBoolean("bias_need_latent_factor");

        needFirstOrder = (K[0] >= 1);
        needSecondOrder = (K[1] >= 1);

        LOG_UTILS.importantInfo("K:" + Arrays.toString(K));
        //comm.LOG_UTILS.importantInfo("seed:" + seed);
        LOG_UTILS.importantInfo("random:" + randomParams);
        LOG_UTILS.importantInfo("bias_need_latent_factor:" + biasNeedLatentFactor);
        LOG_UTILS.importantInfo("need_first_order:" + needFirstOrder + ", need_second_order:" + needSecondOrder);
    }


    @Override
    protected boolean needLaplace() {
        return false;
    }

    @Override
    protected CoreData getCoreData() {
        return new ContinuousCoreData(comm, coreParams, featureMap, pyTransformFunc, needPyTransform, featureHash);
    }

    @Override
    protected void loadModel() throws IOException, Mp4jException {
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
//        for (int i = secondOrderIndexStart; i < dim; i++) {
//            w[0][i] = (float)(rand.nextGaussian() * 0.01);
//        }

        if (coreParams.need_bias) {
            for (int i = 0; i < K[1]; i++) {
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
            LOG_UTILS.importantInfo("old model doesn't exist, new model...model path:" + modelParams.data_path);
            return;
        }


        int cnt = 0;
        List<Iterator<String>> iterators = fs.read(Arrays.asList(modelParams.data_path));
        for (Iterator<String> it : iterators) {
            while (it.hasNext()) {
                String line = it.next();
                if (line.trim().length() == 0) {
                    LOG_UTILS.error("invalid model line:" + line);
                    continue;
                }
                String []info = line.trim().split(modelParams.delim);

                Integer gidx = fName2IndexMap.get(info[0]);
                cnt ++;
                if (gidx == null) {
                    continue;
                }

                float firstOrderWei = Float.parseFloat(info[1]);
                w[0][gidx] = firstOrderWei; // maybe include bias

                int startidx = secondOrderIndexStart + gidx * K[1];
                for (int i = 0; i < K[1]; i++) {
                    w[0][startidx + i] = Float.parseFloat(info[i + 2]);
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
    protected void handleOtherTrainInfo() {

    }

    @Override
    protected void handleOtherTestInfo() {

    }

    @Override
    protected void setDim() throws Mp4jException {
        dim = (1 + K[1]) * fName2IndexMap.size();
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

                    int sidx = secondOrderIndexStart + idx * K[1];
                    StringBuffer sb = new StringBuffer();
                    for (int i = 0; i < K[1]; i++) {
                        sb.append(w[0][sidx + i]);
                        if (i < K[1] - 1) {
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

                    StringBuffer sb = new StringBuffer();
                    int sidx = secondOrderIndexStart + idx * K[1];
                    for (int i = 0; i < K[1]; i++) {
                        sb.append(w[0][sidx + i]);
                        if (i < K[1] - 1) {
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

    public int firstOrderIndexStart() {
        return coreParams.need_bias ? 1 : 0;
    }

    public int secondOrderIndexStart() {
        return fName2IndexMap.size();
    }
}
