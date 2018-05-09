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

import lombok.Data;

import java.io.Reader;
import java.util.*;

/**
 * @author xialong
 */

@Data
public class FFMOnlinePredictor extends ContinuousOnlinePredictor<float[]> {

    private final int K;

    private final String fieldDelim;
    private final String fieldDictPath;

    private final Map<String, Integer> field2IndexMap = new HashMap<>();
    private int fieldSize;

    private final ThreadLocal<float[]> assistbuffer = new ThreadLocal<>();
    private final ThreadLocal<int[]> fieldbuffer = new ThreadLocal<>();
    private final ThreadLocal<float[]> valbuffer = new ThreadLocal<>();

    private final int maxFeatureNum;

    public FFMOnlinePredictor(String configPath) throws Exception {
        super(configPath);

        List<Integer> klist = config.getIntList("k");
        K = klist.get(1);

        fieldDelim = config.getString("data.delim.field_delim");
        fieldDictPath = config.getString("model.field_dict_path");
        maxFeatureNum = config.getInt("data.max_feature_dim") + 1;

        loadModel();
    }

    public FFMOnlinePredictor(Reader configReader) throws Exception {
        super(configReader);

        List<Integer> klist = config.getIntList("k");
        K = klist.get(1);

        fieldDelim = config.getString("data.delim.field_delim");
        fieldDictPath = config.getString("model.field_dict_path");
        maxFeatureNum = config.getInt("data.max_feature_dim") + 1;

        loadModel();
    }

    @Override
    protected OnlinePredictor loadModel() throws Exception {
        // load field dict
        if (!fs.exists(fieldDictPath)) {
            throw new Exception("field dict path doesn't exist! path:" + fieldDictPath);
        }

        if (modelParams.need_bias) {
            field2IndexMap.put(modelParams.bias_feature_name, 0);
        }

        List<Iterator<String>> iterators = fs.read(Arrays.asList(fieldDictPath));
        for (Iterator<String> it : iterators) {
            while (it.hasNext()) {
                String line = it.next();
                int size = field2IndexMap.size();
                field2IndexMap.put(line.trim(), size);
            }
        }
        LOG.info("field dict size:" + field2IndexMap.size() + "\n" +
                field2IndexMap);
        fieldSize = field2IndexMap.size();

        // load field dict
        if (!fs.exists(modelParams.data_path)) {
            throw new Exception("model data path doesn't exist! path:" + modelParams.data_path);
        }

        int cnt = 0;
        iterators = fs.read(Arrays.asList(modelParams.data_path));
        for (Iterator<String> it : iterators) {
            while (it.hasNext()) {
                String line = it.next();
                if (line.trim().length() == 0) {
                    LOG.error("invalid model line(length=0):" + line);
                    continue;
                }
                String []info = line.trim().split(modelParams.delim);
//                if (fieldSize != (info.length - 5) / K) {
//                    LOG.info("invalid model line:" + line);
//                    continue;
//                }

//                if (info.length < 2) {
//                    LOG.error("[invalid model line:" + line);
//                    continue;
//                }


                float []w = modelMap.get(info[0]);
                if (w == null) {
                    w = new float[1 + K * fieldSize];
                    modelMap.put(info[0], w);
                }

                for (int k = 0; k < K * fieldSize + 1; k++) {
                    float wei = Float.parseFloat(info[1 + k]);
                    w[k] = wei;
                }

                if (cnt < 10) {
                    LOG.info("feature:" + info[0] + ", wei:" + Arrays.toString(w));
                }
            }
        }

        LOG.info("load model finished! feature cnt:" + cnt);

        return this;
    }

    @Override
    public double score(Map<String, Float> features, Object other) {
        features.remove(modelParams.bias_feature_name);

        if (commonParams.featureParams.feature_hash.need_feature_hash) {
            features = featureHash.hashMap2Map(features);
        }

        int stride = fieldSize * K;

        float []assist = assistbuffer.get();
        if (assist == null) {
            assist = new float[K * fieldSize * (maxFeatureNum + 1)];
            assistbuffer.set(assist);
        }

        int []fieldIdxArr = fieldbuffer.get();
        if (fieldIdxArr == null) {
            fieldIdxArr = new int[maxFeatureNum + 1];
            fieldbuffer.set(fieldIdxArr);
        }

        float []valArr = valbuffer.get();
        if (valArr == null) {
            valArr = new float[maxFeatureNum + 1];
            valbuffer.set(valArr);
        }

        double wx = 0.0;
        if (modelParams.need_bias) {
            features.put(modelParams.bias_feature_name, 1.0f);
        }

        int cidx = 0;
        for (Map.Entry<String, Float> feature : features.entrySet()) {

            // field idx
            Integer fieldIdx = field2IndexMap.get(feature.getKey().split(fieldDelim)[0]);
            if (fieldIdx == null) {
                continue;
            }
            fieldIdxArr[cidx] = fieldIdx;

            float val = transform(feature.getKey(), feature.getValue());
            // val
            valArr[cidx] = val;

            // weight
            float []w = modelMap.get(feature.getKey());
            wx += w[0] * val;
            System.arraycopy(w, 1, assist, cidx * stride, stride);
            cidx ++;
        }

        int size = cidx;
        double fx = 0.0;
        int pidx = 0;
        for (int p = 0; p < size; p ++) {
            float pval = valArr[p];
            int pfieldstart = fieldIdxArr[p] * K;
            int qidx = pidx + 1;

            int pstartIdx = pidx * stride;
            for (int q = p + 1; q < size; q ++) {
                float qval = valArr[q];
                int qfieldstart = fieldIdxArr[q] * K;
                int qstartIdx = qidx * stride;
                double wTw = 0.0;
                for (int f = 0; f < K; f++) {
                    wTw += assist[pstartIdx + qfieldstart + f] * assist[qstartIdx + pfieldstart + f];
                }
                wTw *= pval * qval;
                fx += wTw;
                qidx ++;
            }
            pidx ++;
        }
        fx += wx;

        return fx;
    }

}
