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
import com.fenbi.ytklearn.fs.IFileSystem;
import com.typesafe.config.Config;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * @author xialong
 */

public class LinearModelDataFlow extends ContinuousDataFlow {

    public LinearModelDataFlow(IFileSystem fs,
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
    }

    @Override
    protected CoreData getCoreData() {
        return new ContinuousCoreData(comm, coreParams, featureMap, pyTransformFunc, needPyTransform, featureHash);
    }

    @Override
    protected boolean needLaplace() {
        return true;
    }

    @Override
    protected void loadModel() throws IOException, Mp4jException {
        w = new float[threadNum][];
        precision = new float[threadNum][];
        for (int t = 0; t < threadNum; t++) {
            w[t] = new float[dim];
            precision[t] = new float[dim];
            for (int i = 0; i < dim; i++) {
                w[t][i] = 0;
            }
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
                    LOG_UTILS.error("invalid model line:" + line);
                    continue;
                }
                String []info = line.trim().split(modelParams.delim);

                if (info.length < 2) {
                    LOG_UTILS.error("invalid model line:" + line);
                    continue;
                }

                Integer gidx = fName2IndexMap.get(info[0]);
                cnt ++;
                if (gidx == null) {
                    continue;
                }

                double wei = Double.parseDouble(info[1]);
                for (int t = 0; t < threadNum; t++)
                    w[t][gidx] = (float)wei;
            }
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
    public void dumpModel() throws IOException, Mp4jException {
        PrintWriter writer = null;
        PrintWriter dictWriter = null;

        int avg = dim / slaveNum;
        int start = rank * avg;
        int end = (rank + 1) * avg;

        if (rank == slaveNum - 1) {
            end = dim;
        }

        LOG_UTILS.verboseInfo("dump from:" + start + ", to:" + end);

        try {
            int nonzeroNum = 0;
            String modelPartPath = modelParams.data_path + "/model-" + String.format("%05d", rank);
            String dictPartPath = modelParams.data_path + "_dict/dict-" + String.format("%05d", rank);
            writer = new PrintWriter(fs.getWriter(modelPartPath));
            dictWriter = new PrintWriter(fs.getWriter(dictPartPath));

            for (Map.Entry<String, Integer> entry : fName2IndexMap.entrySet()) {

                if (!entry.getKey().equalsIgnoreCase(modelParams.bias_feature_name)) {

                    if (Math.abs(w[0][entry.getValue()]) > 0.0) {
                        nonzeroNum ++;
                    } else {
                        continue;
                    }

                    int idx = entry.getValue();
                    if (!(idx >= start && idx < end)) {
                        continue;
                    }

                    String str = String.format("%s%s%f%s%f",
                            entry.getKey(), modelParams.delim,
                            w[0][idx], modelParams.delim,
                            precision[0][idx]);
                    writer.println(str);
                    dictWriter.println(entry.getKey());
                } else {
                    if (Math.abs(w[0][entry.getValue()]) > 0.0) {
                        nonzeroNum ++;
                    }

                    int idx = entry.getValue();
                    if (!(idx >= start && idx < end)) {
                        continue;
                    }

                    writer.println(entry.getKey() + modelParams.delim + w[0][entry.getValue()]
                            + modelParams.delim + "null");
                }
            }
            LOG_UTILS.importantInfo("model is written to " + modelPartPath + ", all nonzero num:" + nonzeroNum +
                    ", dim:" + dim +
                    ", prop:" + (nonzeroNum * 1.0 / dim));

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
