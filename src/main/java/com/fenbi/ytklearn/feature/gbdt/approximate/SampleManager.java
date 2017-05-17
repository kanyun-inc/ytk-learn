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

package com.fenbi.ytklearn.feature.gbdt.approximate;

import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.IObjectOperator;
import com.fenbi.mp4j.utils.KryoUtils;
import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.exception.YtkLearnException;
import com.fenbi.ytklearn.feature.gbdt.approximate.sampler.SampleType;
import com.fenbi.ytklearn.param.gbdt.GBDTFeatureParams;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.fenbi.ytklearn.feature.gbdt.approximate.sampler.ISampler;
import com.fenbi.ytklearn.feature.gbdt.approximate.sampler.SamplerFactory;
import com.fenbi.ytklearn.utils.WeightApproximateQuantile;

import java.util.*;

/**
 * feature approximate sample manager
 * @author wufan
 * @author xialong
 */

public class SampleManager {

    private static final SampleType DEFAULT_SAMPLER_TYPE = SampleType.getDefault();

    private HashMap<Integer, ISampler> featureSamplerMap;

    private ThreadCommSlave comm;

    private GBDTCoreData data;
    private int featureNum;

    public SampleManager(GBDTCoreData data, ThreadCommSlave comm) {
        this.data = data;
        this.featureNum = data.usefulFeatureDim;
        featureSamplerMap = new HashMap<>(featureNum);
        this.comm = comm;
    }

    public void init(GBDTFeatureParams featParams) throws Mp4jException {
        SampleType defaultType = null;
        Map<String, String> defaultParams = null;

        List<GBDTFeatureParams.FeatureApproximateParams> featParamList = featParams.featureApproximateParamList;
        for (GBDTFeatureParams.FeatureApproximateParams featParam : featParamList) {
            SampleType type = featParam.type;
            int[] cols = featParam.cols;
            Map<String, String> params = featParam.params;

            // default sampler
            if (cols.length == 1 && cols[0] == -1) {
                defaultType = type;
                defaultParams = params;

            } else {
                for (int col : cols) {
                    ISampler sampler = SamplerFactory.create(type, comm);
                    sampler.init(params);
                    if (featureSamplerMap.containsKey(col)) {
                        throw new YtkLearnException(String.format("feature col: %s configs more than once in config file!", data.fIndex2NameMap.get(col)));
                    }
                    featureSamplerMap.put(col, sampler);
                }
            }
        }
        if (defaultType == null) {
            defaultType = DEFAULT_SAMPLER_TYPE;
        }

        for (int i = 0; i < featureNum; i++) {
            if (featureSamplerMap.get(i) == null) {
                ISampler sampler = SamplerFactory.create(defaultType, comm);
                sampler.init(defaultParams);
                featureSamplerMap.put(i, sampler);

            }
        }
    }


    public Map<String, Set<Float>> doSample() throws Exception {
        Map<String, Set<Float>> notQuantileApprFeaMap = new HashMap<>();
        Map<String, WeightApproximateQuantile.Summary> quantileSummayMap = new HashMap<>(featureNum);

        for (Map.Entry<Integer, ISampler> entry: featureSamplerMap.entrySet()) {
            int fid = entry.getKey();
            ISampler sampler = entry.getValue();
            CheckUtils.check(sampler != null, "Sample Manager do sample for feature: %s error!", data.fIndex2NameMap.get(fid));
            Object sampleResult = sampler.doSample(data, fid);
            CheckUtils.check(sampleResult != null, "[GBDT] feature: %s has no global feature vals!", data.fIndex2NameMap.get(fid));
            if (sampleResult instanceof Set) {
                notQuantileApprFeaMap.put(String.valueOf(fid), (Set<Float>) sampleResult);

            } else if (sampleResult instanceof WeightApproximateQuantile.Summary) {
                quantileSummayMap.put(String.valueOf(fid), (WeightApproximateQuantile.Summary)sampleResult);

            } else {
                throw new YtkLearnException("[GBDT] do sample result err, feature:" + data.fIndex2NameMap.get(fid));
            }
        }

        notQuantileApprFeaMap = comm.allreduceMapSetUnion(notQuantileApprFeaMap, KryoUtils.getDefaultSerializer(Float.class), Float.class);
        quantileSummayMap = comm.allreduceMap(quantileSummayMap, Operands.OBJECT_OPERAND(new WeightApproximateQuantile.SummarySerializer(), WeightApproximateQuantile.Summary.class), new IObjectOperator<WeightApproximateQuantile.Summary>() {
            @Override
            public WeightApproximateQuantile.Summary apply(WeightApproximateQuantile.Summary o1, WeightApproximateQuantile.Summary o2) {
                try {
                    return o1.merge(o2);
                } catch (Exception e) {
                    try {
                        comm.exception(e);
                    } catch (Mp4jException e1) {
                        e1.printStackTrace();
                    }
                    return null;
                }
            }
        });

        // merge
        Map<String, Set<Float>> globalApprFeaVals = new HashMap<>(featureNum);
        globalApprFeaVals.putAll(notQuantileApprFeaMap);
        for (Map.Entry<String, WeightApproximateQuantile.Summary> entry: quantileSummayMap.entrySet()) {
            String fid = entry.getKey();
            Set<Float> vals = featureSamplerMap.get(Integer.parseInt(fid)).getSamples(entry.getValue());
            globalApprFeaVals.put(fid, vals);
        }
        CheckUtils.check(globalApprFeaVals.size() == featureNum, "[GBDT] inner error, get approximate feature val error, globalFeaSplitVals size(%d) != maxFeatureDim(%d)", globalApprFeaVals.size(), featureNum);
        return globalApprFeaVals;
    }


    // inverse transform, convert feature value to it's origin scale, do it in place
    public void reverse(int fid, float[] converData) {
        ISampler sampler = featureSamplerMap.get(fid);
        CheckUtils.check(sampler != null, "Sample Manager doesn't have sampler for feature: %s error!", data.fIndex2NameMap.get(fid));
        for (int i = 0; i < converData.length; i++) {
            converData[i] = sampler.inverseTransform(converData[i]);
        }
    }

}
