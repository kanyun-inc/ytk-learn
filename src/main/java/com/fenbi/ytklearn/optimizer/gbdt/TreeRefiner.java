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

package com.fenbi.ytklearn.optimizer.gbdt;

import com.fenbi.mp4j.operator.IObjectOperator;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.ytklearn.data.Constants;
import com.fenbi.ytklearn.data.Tuple;
import com.fenbi.ytklearn.data.gbdt.TreeMakerType;
import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.utils.*;
import com.fenbi.ytklearn.param.gbdt.GBDTCommonParams;
import com.fenbi.ytklearn.data.gbdt.Tree;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.ytklearn.data.gbdt.SamplePositionData;

import java.util.*;

/**
 * refine value of tree leaves after tree is builed, it's used in first order approximate
 * @author wufan
 * @author xialong
 */

public class TreeRefiner {

    public LogUtils LOG_UTILS;
    private String objName;
    private final TreeMakerType treeMakerType;
    private final float learningRate;
    private Boolean ladRefineAppr;

    private ThreadCommSlave comm;

    private float[] residuals;
    private float[] weights;

    public TreeRefiner(ThreadCommSlave comm, GBDTCommonParams commonParams) {
        this.comm = comm;
        this.objName = commonParams.optimizationParams.objective;
        this.learningRate = commonParams.optimizationParams.regularization.learningRate;
        this.treeMakerType = commonParams.optimizationParams.tree_maker_type;
        this.ladRefineAppr = commonParams.optimizationParams.lad_refine_appr;
        this.LOG_UTILS = new LogUtils(comm, commonParams.verbose);
    }

    // compute leaf refine value for tree in feature parallel version
    // xRowRange: left close, right open
    private Map<String, Float> getLeafRefineValForLADPrecise(int[] xRowRange,
                                                             int[] positionGlobal,
                                                             Tree tree) throws Mp4jException {
        Map<Integer, List<Integer>> leaf2SampleIndex;
        List<Integer> leafIndex = tree.getLeafNodes();
        leaf2SampleIndex = new HashMap<>(leafIndex.size());
        for (int leaf : leafIndex) {
            leaf2SampleIndex.put(leaf, new ArrayList<>());
        }

        for (int i = xRowRange[0]; i < xRowRange[1]; i++) {
            int nodeId = positionGlobal[i];
            leaf2SampleIndex.get(nodeId).add(i - xRowRange[0]);
        }

        PreciseQuantile weiMedian = new PreciseQuantile(comm, true);
        Map<String, Float> leafRefineValMap = new HashMap<>(leaf2SampleIndex.size());
        for (int leafId : leafIndex) {
            List<Integer> sampleList = leaf2SampleIndex.get(leafId);
            Float leafValue = weiMedian.getWeightedQuantile(sampleList, residuals, weights);
            leafRefineValMap.put(String.valueOf(leafId), leafValue);
        }

        leafRefineValMap = comm.allreduceMap(leafRefineValMap, Operands.FLOAT_OPERAND(), Operators.Float.MIN);
        CheckUtils.check(leafRefineValMap.size() == leaf2SampleIndex.size(),
                "TreeRefiner: after reduce leaf node number error! leafRefineValMap size=%d, leaf2SampleIndex size=%d",
                leafRefineValMap.size(), leaf2SampleIndex.size());
        return leafRefineValMap;
    }


    // compute leaf refine value for tree in data parallel version
    private Map<String, Float> getLeafRefineValForLADPrecise(SamplePositionData samplePosition,
                                                             Tree tree) throws Mp4jException {

        PreciseQuantile weiMedian = new PreciseQuantile(comm, false);
        int[] posAndSampleIndex = samplePosition.getPosAndSampleIndex();
        List<Tuple<Integer, Integer>> nodeSampleInterval = samplePosition.getNodeSampleInterval();

        List<Integer> leafIndex = tree.getLeafNodes();
        Map<String, Float> leafRefineValMap = new HashMap<>(leafIndex.size());
        for (int leafId : leafIndex) {
            Tuple<Integer, Integer> interval = nodeSampleInterval.get(leafId);
            Float leafValue = weiMedian.getWeightedQuantile(interval, posAndSampleIndex, residuals, weights);
            leafRefineValMap.put(String.valueOf(leafId), leafValue);
        }
        leafRefineValMap = comm.allreduceMap(leafRefineValMap, Operands.FLOAT_OPERAND(), Operators.Float.MIN);
        CheckUtils.check(leafRefineValMap.size() == leafIndex.size(),
                "TreeRefiner: after reduce leaf node number error! leafRefineValMap size=%d, leaf2SampleIndex size=%d",
                leafRefineValMap.size(), leafIndex.size());
        return leafRefineValMap;
    }


    private Map<String, Float> getLeafRefineValForLADAppr(int[] xRowRange,
                                                          int[] positionGlobal,
                                                          Tree tree) throws Exception {
        List<Integer> leafIndex = tree.getLeafNodes();
        Map<Integer, WeightApproximateQuantile> mapQuantile = new HashMap<>(leafIndex.size());
        for (int leafId : leafIndex) {
            WeightApproximateQuantile quantile = new WeightApproximateQuantile(tree.getNodeStat(leafId).getNodeSampleCnt(),
                    Constants.QUNANTILE_APPROXIMATE_EPS, Constants.QUNANTILE_PRECISION_MAX_SAMPLE_CNT);
            mapQuantile.put(leafId, quantile);
        }

        for (int i = xRowRange[0]; i < xRowRange[1]; i++) {
            int nodeId = positionGlobal[i];
            int localIdex = i - xRowRange[0];
            mapQuantile.get(nodeId).update(residuals[localIdex], weights[localIdex]);
        }
        Map<String, WeightApproximateQuantile.Summary> mapSummary = new HashMap<>(leafIndex.size());
        for (Map.Entry<Integer, WeightApproximateQuantile> entry : mapQuantile.entrySet()) {
            WeightApproximateQuantile.Summary summary = entry.getValue().mergeAllAndCompress();
            mapSummary.put(String.valueOf(entry.getKey()), summary);
        }

        Map<String, Float> leafRefineValMap = getLeafMedian(mapSummary);
        CheckUtils.check(leafRefineValMap.size() == leafIndex.size(),
                "TreeRefiner: after reduce leaf node number error! leafRefineValMap size=%d, leaf2SampleIndex size=%d",
                leafRefineValMap.size(), leafIndex.size());
        return leafRefineValMap;

    }

    private Map<String, Float> getLeafRefineValForLADAppr(SamplePositionData samplePosition, Tree tree) throws Exception {
        int[] posAndSampleIndex = samplePosition.getPosAndSampleIndex();
        List<Tuple<Integer, Integer>> nodeSampleInterval = samplePosition.getNodeSampleInterval();

        List<Integer> leafIndex = tree.getLeafNodes();
        Map<String, WeightApproximateQuantile.Summary> mapSummary = new HashMap<>(leafIndex.size());
        for (int leafId : leafIndex) {
            Tuple<Integer, Integer> interval = nodeSampleInterval.get(leafId);
            WeightApproximateQuantile quantile = new WeightApproximateQuantile(tree.getNodeStat(leafId).getNodeSampleCnt(),
                    Constants.QUNANTILE_APPROXIMATE_EPS, Constants.QUNANTILE_PRECISION_MAX_SAMPLE_CNT);

            for (int i = interval.v1; i < interval.v2; i++) {
                int sampleIdx = posAndSampleIndex[(i << 1) + 1];
                quantile.update(residuals[sampleIdx], weights[sampleIdx]);
            }

            WeightApproximateQuantile.Summary summary = quantile.mergeAllAndCompress();
            mapSummary.put(String.valueOf(leafId), summary);
        }

        Map<String, Float> leafRefineValMap = getLeafMedian(mapSummary);
        CheckUtils.check(leafRefineValMap.size() == leafIndex.size(),
                "TreeRefiner: after reduce leaf node number error! leafRefineValMap size=%d, leaf2SampleIndex size=%d",
                leafRefineValMap.size(), leafIndex.size());
        return leafRefineValMap;
    }

    private Map<String, Float> getLeafMedian(Map<String, WeightApproximateQuantile.Summary> mapSummary) throws Mp4jException {
        mapSummary = comm.allreduceMap(mapSummary, Operands.OBJECT_OPERAND(new WeightApproximateQuantile.SummarySerializer(), WeightApproximateQuantile.Summary.class), new IObjectOperator<WeightApproximateQuantile.Summary>() {
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
        }); // WeightApproximateQuantile.Summary is read only later


        Map<String, Float> leafRefineValMap = new HashMap<>(mapSummary.size());
        for (Map.Entry<String, WeightApproximateQuantile.Summary> entry : mapSummary.entrySet()) {
            float median = entry.getValue().query(0.5);
            leafRefineValMap.put(entry.getKey(), median);
        }
        return leafRefineValMap;
    }

    public void refine(GBDTCoreData data,
                       SamplePositionData samplePosition,
                       int[] positionGlobal,
                       Tree tree) throws Exception {
        if (objName.equals("l1")) {
            int sampleNum = data.sampleNum;
            // for reuse
            if (residuals == null || residuals.length != sampleNum) {
                residuals = new float[sampleNum];
                weights = new float[sampleNum];
            }

            int cursor = data.getCursor2d();
            float[][] labels = data.getY();
            float[][] scores = data.score;
            float[][] weight2D = data.getWeight();
            int sid = 0;
            for (int k = 0; k < cursor; k++) {
                for (int i = 0; i < labels[k].length; i++) {
                    residuals[sid] = labels[k][i] - scores[k][i];
                    weights[sid] = weight2D[k][i];
                    sid++;
                }
            }

            Map<String, Float> leafRefineValMap;
            if (treeMakerType.equals(TreeMakerType.FEATURE_PARALLEL)) {
                if (ladRefineAppr) {
                    leafRefineValMap = getLeafRefineValForLADAppr(data.xRowRange, positionGlobal, tree);
                } else {
                    leafRefineValMap = getLeafRefineValForLADPrecise(data.xRowRange, positionGlobal, tree);
                }
            } else {
                if (ladRefineAppr) {
                    leafRefineValMap = getLeafRefineValForLADAppr(samplePosition, tree);
                } else {
                    leafRefineValMap = getLeafRefineValForLADPrecise(samplePosition, tree);
                }
            }

            for (Map.Entry<String, Float> entry : leafRefineValMap.entrySet()) {
                int nodeId = Integer.parseInt(entry.getKey());
                tree.getNode(nodeId).setLeaf(entry.getValue() * learningRate);
            }
        }
    }
}
