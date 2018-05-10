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

package com.fenbi.ytklearn.predictor;

import com.fenbi.ytklearn.dataflow.GBMLRDataFlow;
import lombok.Data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

/**
 * @author xialong
 */

@Data
public class GBMLROnlinePredictor extends ContinuousOnlinePredictor<float[]> implements ITreePredictor {
    protected final int K;
    protected final float uniformBaseScore;
    protected final boolean sampleDepdtBaseScore;
    protected int treeNum;
    protected final double learningRate;

    protected final ThreadLocal<double[]> wxbuffer = new ThreadLocal<>();
    protected final ThreadLocal<double[]> gatingbuffer = new ThreadLocal<>();
    protected final GBMLRDataFlow.Type type;

    public GBMLROnlinePredictor(String configPath) throws Exception {
        super(configPath);

        this.K = config.getInt("k");
        this.uniformBaseScore = (float)lossFunction.pred2Score(config.getDouble("uniform_base_prediction"));
        this.sampleDepdtBaseScore = config.getBoolean("sample_dependent_base_prediction");
        this.treeNum = config.getInt("tree_num");
        this.type = GBMLRDataFlow.Type.getType(config.getString("type"));
        if (type == GBMLRDataFlow.Type.RF) {
            this.learningRate = 1.0;
        } else {
            this.learningRate = config.getDouble("learning_rate");
        }
        loadModel();
    }

    public GBMLROnlinePredictor(Reader configReader) throws Exception {
        super(configReader);

        this.K = config.getInt("k");
        this.uniformBaseScore = (float)config.getDouble("uniform_base_prediction");
        this.sampleDepdtBaseScore = config.getBoolean("sample_dependent_base_prediction");
        this.treeNum = config.getInt("tree_num");
        this.type = GBMLRDataFlow.Type.getType(config.getString("type"));
        if (type == GBMLRDataFlow.Type.RF) {
            this.learningRate = 1.0;
        } else {
            this.learningRate = config.getDouble("learning_rate");
        }
        loadModel();
    }

    @Override
    protected OnlinePredictor loadModel() throws Exception {
        // check model data path
        if (!fs.exists(modelParams.data_path)) {
            throw new Exception("gbmlr old model doesn't exist! path:" + modelParams.data_path);

        }

        // load model info
        String modelInfoPath = modelParams.data_path + "/tree-info";
        if (!fs.exists(modelInfoPath)) {
            throw new Exception("have no gbmlr model info data, old model doesn't exist! path:" + modelInfoPath);
        }

        List<String> lineList = new ArrayList<>();
        List<Iterator<String>> iterators = fs.read(Arrays.asList(modelInfoPath));
        for (Iterator<String> it : iterators) {
            while (it.hasNext()) {
                String line = it.next();
                if (line.trim().length() > 0) {
                    lineList.add(line.trim());
                }
            }
        }

        if (lineList.size() != 4) {
            throw new Exception("model info must have 4 lines!");
        }

        int oldK = Integer.parseInt(lineList.get(0).split(":")[1]);
        int oldTreeNum = Integer.parseInt(lineList.get(1).split(":")[1]);
        int finishedTreeNum = Integer.parseInt(lineList.get(2).split(":")[1]);
        double oldUniformBaseScore = Double.parseDouble(lineList.get(3).split(":")[1]);
        if (K != oldK) {
            throw new Exception("model info K != param K" + ", model info K:" + oldK + ", param K:" + K);
        }

        if (treeNum > finishedTreeNum) {
            LOG.info("treeNum:" + treeNum + " > finishedTreeNum:" + finishedTreeNum + ", use only finished trees!");
        }
        treeNum = Math.min(treeNum, finishedTreeNum);

        if (oldUniformBaseScore != uniformBaseScore) {
            throw new IOException("old uniform_base_prediction != uniform_base_prediction, old:" + oldUniformBaseScore + ", new:" + uniformBaseScore);
        }

        LOG.info("load old model info, K:" + oldK +
                ", old tree num:" + oldTreeNum +
                ", tree num:" + treeNum +
                ", finished tree num:" + finishedTreeNum +
                ", old uniform base score:" + oldUniformBaseScore +
                ", uniform base score:" + uniformBaseScore);

        // load finished tress
        int stride = 2 * K - 1;
        for (int tree = 0; tree < treeNum; tree ++) {

            List<String> files = fs.recurGetPaths(Arrays.asList(modelParams.data_path +
                    "/tree-" + String.format("%05d", tree)));

            int cnt = 0;
            for (String  file : files) {
                BufferedReader reader = null;
                try {
                    reader = new BufferedReader(fs.getReader(file));
                    // skip first line
                    String line = reader.readLine();
                    if (K != Integer.parseInt(line.split(":")[1])) {
                        throw new Exception("old model k != config's K = " + K);
                    }
                    LOG.info(line);
                    while ((line = reader.readLine()) != null) {
                        if (line.trim().length() == 0) {
                            LOG.info("[ERROR] invalid model line:" + line);
                            continue;
                        }
                        String []info = line.trim().split(modelParams.delim);

                        if (info.length < 2) {
                            LOG.info("[ERROR] invalid model line:" + line);
                            continue;
                        }

                        float []w = modelMap.get(info[0]);
                        cnt ++;
                        if (w == null) {
                            w = new float[stride * treeNum];
                            modelMap.put(info[0], w);
                        }

                        int gidx = tree * stride;
                        for (int k = 0; k < stride; k++) {
                            float wei = Float.parseFloat(info[1 + k]);
                            w[gidx + k] = wei;
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
            LOG.info("tree:" + tree + ", load model finished, old model feature cnt:" + cnt);

        }

        return this;
    }

    @Override
    public double score(Map<String, Float> features, Object other) {
        features.remove(modelParams.bias_feature_name);

        if (commonParams.featureParams.feature_hash.need_feature_hash) {
            features = featureHash.hashMap2Map(features);
        }

        int stride = 2 * K - 1;
        int vstride = K - 1;
        int size = stride * treeNum;

        double []wx = wxbuffer.get();
        if (wx == null) {
            wx = new double[size];
            wxbuffer.set(wx);
        }

        double []gating = gatingbuffer.get();
        if (gating == null) {
            gating = new double[K * treeNum];
            gatingbuffer.set(gating);
        }

        // xv, xw
        // bias
        float []w = modelMap.get(modelParams.bias_feature_name);
        if (w != null && modelParams.need_bias) {
            for (int p = 0; p < size; p++) {
                wx[p] = w[p];
            }
        } else {
            for (int p = 0; p < size; p++) {
                wx[p] = 0;
            }
        }

        for (Map.Entry<String, Float> f : features.entrySet()) {
            w = modelMap.get(f.getKey());
            if (w == null) {
                continue;
            }
            for (int i = 0; i < size; i++) {
                wx[i] += w[i] * transform(f.getKey(), f.getValue());
            }
        }

        int idx = 0;
        double fx = 0.0;
        for (int tree = 0; tree < treeNum; tree++) {
            // softmax
            double maxXv = wx[idx];
            for (int j = 1; j < vstride; j++) {
                if (wx[idx + j] > maxXv) {
                    maxXv = wx[idx + j];
                }
            }
            double esum = 0.0;
            for (int j = 0; j < vstride; j++) {
                wx[idx + j] = Math.exp(wx[idx + j] - maxXv);
                esum += wx[idx + j];
            }

            // g(x), fx
            double gk_1 = 1.0;
            double lfx = 0.0;
            double inv = 1.0 / (Math.exp(-maxXv) + esum);
            for (int j = 0; j < vstride; j++) {
                wx[idx + j] = wx[idx + j] * inv;
                gk_1 -= wx[idx + j];
                lfx += wx[idx + j] * wx[idx + vstride + j];
                gating[tree * K + j] = wx[idx + j];
            }
            lfx += gk_1 * wx[idx + stride - 1];
            gating[tree * K + K - 1] = gk_1;

            if (tree < treeNum - 1) {
                fx += learningRate * lfx;
            } else {
                fx += lfx;
            }


            idx += stride;
        }
        double lbias = uniformBaseScore;
        if (sampleDepdtBaseScore) {
            lbias += lossFunction.pred2Score(((Float)other).floatValue());
        }

        if (type == GBMLRDataFlow.Type.RF) {
            fx /= treeNum;
        }
        return lbias + fx;
    }

    @Override
    public double[] predictLeaf(Map<String, Float> features) {
        score(features, new Float(1.0f));
        return gatingbuffer.get();
    }
}
