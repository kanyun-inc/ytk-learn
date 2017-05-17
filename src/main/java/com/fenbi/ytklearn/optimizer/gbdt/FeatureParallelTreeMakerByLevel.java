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

import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.IObjectOperator;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.ytklearn.data.Constants;
import com.fenbi.ytklearn.data.gbdt.*;
import com.fenbi.ytklearn.dataflow.FeatureColData;
import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.param.gbdt.GBDTOptimizationParams;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.fenbi.ytklearn.utils.LogUtils;
import com.fenbi.ytklearn.utils.RandomUtils;

import java.util.*;

/**
 * feature-parallel local tree maker
 * @author wufan
 * @author xialong
 */

public class FeatureParallelTreeMakerByLevel implements ITreeMaker {

    // statistics for node entry, used for tree construction
    private class TreeMakerNodeStats {
        public GradStats sumGradStats;
        public GradStats leftBranchGradStats;

        public float rootGain;
        public float nodeValue;
        public long sampleCnt;

        public SplitInfo best;
        public float lastFeaValue;

        public boolean toExpand;

        public TreeMakerNodeStats() {
            sumGradStats = new GradStats();
            leftBranchGradStats = new GradStats();
            rootGain = 0.0f;
            nodeValue = 0.0f;
            sampleCnt = 0;
            best = new SplitInfo();
            lastFeaValue = 0.0f;
            toExpand = true;
        }

        public void clear() {
            sumGradStats.clear();
            leftBranchGradStats.clear();
            rootGain = 0.f;
            nodeValue = 0.0f;
            sampleCnt = 0;
            best.clear();
            lastFeaValue = 0.0f;
            toExpand = true;
        }
    }

    public LogUtils LOG_UTILS;
    private final ThreadCommSlave comm;

    private final GBDTOptimizationParams params;
    private UpdateStrategy updateStrategy;
    private Tree tree;

    // save the shuffle and sample index of each feature index, for col sample
    private int[] feaIndex;

    // data set info
    private GBDTCoreData trainData;
    private int localSampleNum;
    private int globalSampleNum;
    private int featureDim;
    private FeatureColData featureColData;

    // global sample position
    private int[] globalPosition;

    private int groupId;
    private float[] globalGradPairs;

    // statistics for each constructed node, used in building tree
    private List<TreeMakerNodeStats> nodeStats;
    private int curStatsNum;

    // tmp var, for computing lossChg
    private GradStats rightStats;

    // queue of nodeId to be expanded
    private List<Integer> expandNodes;
    private int leafCnt;

    public FeatureParallelTreeMakerByLevel(ThreadCommSlave comm,
                                           GBDTOptimizationParams params,
                                           GBDTCoreData trainData,
                                           int[] position) {

        this.comm = comm;
        this.LOG_UTILS = new LogUtils(comm, params.verbose);

        this.params = params;
        this.updateStrategy = new UpdateStrategy(params);

        this.trainData = trainData;
        this.featureDim = trainData.usefulFeatureDim;
        this.localSampleNum = trainData.sampleNum;
        this.globalSampleNum = position.length;
        this.featureColData = trainData.xT;

        this.globalGradPairs = new float[globalSampleNum << 1];;
        this.globalPosition = position;

        this.rightStats = new GradStats();
        this.nodeStats = new ArrayList<>(Constants.RESERVE_NUM);
        // expand new node, add root
        this.expandNodes = new ArrayList<>(Constants.RESERVE_NUM);
    }

    public Tree make(int groupId) throws Mp4jException {
        this.groupId = groupId;

        initAssistData();
        tree = new Tree();
        tree.init();

        expandNodes.add(0);
        leafCnt = 1;

        for (int depth = 0; depth < params.max_depth; depth++) {
            if (params.max_leaf_cnt > 0 && leafCnt >= params.max_leaf_cnt) {
                break;
            }
            // compute the sum of gradient of each expand node
            initNodeStats();

            // traverse all values(or approximate) for all features and find the best split value
            findSplit();

            // reset sample globalPosition and sort samples by node id
            resetPosition();

            updateExpandQueue();
            if (expandNodes.size() == 0)
                break;
        }

        // update leaf value
        initNodeStats();
        for (int nid : expandNodes) {
            tree.getNode(nid).setLeaf(nodeStats.get(nid).nodeValue * params.regularization.learningRate);
        }
        // update tree node stats
        for (int nid = 0; nid < tree.getNodeNum(); nid++) {
            updateTreeNodeStat(nid);
        }
        return tree;
    }

    // do instance sampling and feature sampling, add root to expandNodes
    private void initAssistData() throws Mp4jException {
        // tmp var, for computing lossChg
        rightStats.clear();
        expandNodes.clear();
        curStatsNum = 0;

        // all node init to root
        for (int i = 0; i < globalPosition.length; i++) {
            globalPosition[i] = 0;
        }

        // get global gradient
        syncGlobalGrads();

        long seed = System.nanoTime();
        seed = comm.allreduce(seed, Operands.LONG_OPERAND(), Operators.Long.MAX);
        Random rand = new Random(seed);
        //subsample
        int sampleCnt = 0;
        if (params.subsample < 1.0f) {
            for (int i = 0; i < globalSampleNum; i++) {
                if (!RandomUtils.sampleBinary(rand, params.subsample)) {
                    globalPosition[i] = -1;
                } else {
                    sampleCnt++;
                }
            }
            // this worker may have no samples
            LOG_UTILS.verboseInfo(String.format("do subsample, subsampleRate=%f, sample out %d of all %d instances", params.subsample, sampleCnt, globalSampleNum), false);
        } else {
            LOG_UTILS.verboseInfo("no subsample, use all instances");
        }

        // init feature index
        int[] localFeaIndex = new int[featureDim];
        for (int i = 0; i < featureDim; i++) {
            localFeaIndex[i] = i;
        }

        // do feature sample
        if (params.feature_sample_rate < 1.0f) {
            // get a global seed
            long seed2 = System.nanoTime();
            seed2 = comm.allreduce(seed2, Operands.LONG_OPERAND(), Operators.Long.MAX);

            // sample out features
            RandomUtils.shuffle(seed2, localFeaIndex);
            int nFeaSam = Math.round(params.feature_sample_rate * featureDim);
            CheckUtils.check(nFeaSam > 0,
                    "[GBDT] feature_sample_rate=%f is too small, and no feature can be included", params.feature_sample_rate);
            feaIndex = new int[nFeaSam];
            for (int i = 0; i < feaIndex.length; i++) {
                feaIndex[i] = localFeaIndex[i];
            }
            Arrays.sort(feaIndex);
            LOG_UTILS.verboseInfo("do feature sample complete! feature_sample_rate=" + params.feature_sample_rate + ", sample out " + nFeaSam + " features");

        } else {
            feaIndex = localFeaIndex;
            LOG_UTILS.verboseInfo("no featureSample, use all features");

        }
    }

    private void updateTreeNodeStat(int nid) {
        TreeNodeStat tstat = tree.getNodeStat(nid);
        TreeMakerNodeStats mstat = nodeStats.get(nid);
        tstat.setLossChg(mstat.best.getLossChg());
        tstat.setHessSum((float) (mstat.sumGradStats.getSumHess()));
        tstat.setNodeSampleCnt(mstat.sampleCnt);
    }

    // compute and sync global grads
    private void syncGlobalGrads() throws Mp4jException {
        int gradIdx;
        int index2D, index1D;
        int index = trainData.xRowRange[0] << 1;
        for (int sid = 0; sid < localSampleNum; sid++) {
            index2D = sid / trainData.DENSE_MAX_1D_SAMPLE_CNT;
            index1D = sid % trainData.DENSE_MAX_1D_SAMPLE_CNT;
            gradIdx = (index1D * trainData.numTreeInGroup + groupId) * 2;
            globalGradPairs[index] = trainData.gradPairs[index2D][gradIdx];
            globalGradPairs[index + 1] = trainData.gradPairs[index2D][gradIdx + 1];
            index += 2;
        }
        // sync global gradient
        globalGradPairs = comm.allgatherArray(globalGradPairs, Operands.FLOAT_OPERAND(), trainData.globalGradAssignFrom, trainData.globalGradAssignTo);
    }

    private void initNodeStats() throws Mp4jException {
        // construct node stats for each to expand node
        int nodeNum = tree.getNodeNum();
        CheckUtils.check(curStatsNum <= nodeNum,
                "[GBDT] inner error, nodeStats num(%d) is larger than %d", nodeStats.size(), nodeNum);
        for (int i = curStatsNum; i < nodeNum; i++) {
            if (nodeStats.size() <= i) {
                nodeStats.add(new TreeMakerNodeStats()); //stats value all set to 0
            } else {
                nodeStats.get(i).clear();
            }
        }
        curStatsNum = nodeNum;

        // compute gradient sum
        TreeMakerNodeStats curNodeStats;
        for (int i = 0; i < globalSampleNum; i++) {
            // not sampled
            if (globalPosition[i] < 0) {
                continue;
            }
            curNodeStats = nodeStats.get(globalPosition[i]);
            if (!curNodeStats.toExpand) {
                continue;
            }
            curNodeStats.sumGradStats.add(globalGradPairs, i);
            curNodeStats.sampleCnt++;
        }

        // compute root grain and weight for each node
        for (int j : expandNodes) {
            curNodeStats = nodeStats.get(j);
            curNodeStats.rootGain = (float) updateStrategy.calcGain(curNodeStats.sumGradStats);
            curNodeStats.nodeValue = (float) updateStrategy.calcNodeValue(curNodeStats.sumGradStats);
        }
    }

    // find splits at current level, traverse feature approximate matrix
    private void findSplit() throws Mp4jException {
        for (int nid : expandNodes) {
            TreeMakerNodeStats tstats = nodeStats.get(nid);
            if (!updateStrategy.canSplit(tstats.sumGradStats.getSumHess(), tstats.sampleCnt)) {
                tstats.toExpand = false;
            }
        }

        for (int fid : feaIndex) {
            if (fid >= trainData.xColRange[0] && fid < trainData.xColRange[1]) {
                enumerateSplit(fid);
            }
        }

        // sync global best split
        syncBestSplit();
        // update tree nodes
        SplitInfo spBest;
        for (int nid : expandNodes) {
            spBest = nodeStats.get(nid).best;
            if ((params.max_leaf_cnt < 0 || leafCnt < params.max_leaf_cnt) && spBest.getLossChg() > params.min_split_loss) {
                tree.addChilds(nid);
                leafCnt++;
                tree.getNode(nid).setSplit(spBest.getSplitIndex(), spBest.getSplitValue());
            } else {
                tree.getNode(nid).setLeaf(nodeStats.get(nid).nodeValue * params.regularization.learningRate);
            }
        }
    }

    // enumerate split values of specific feature
    private void enumerateSplit(int fid) throws Mp4jException {
        int feaColIdx = fid - trainData.xColRange[0];
        float[] feaInst = featureColData.feaInstByCol[feaColIdx];

        float lossChg, feaValue;
        int instIdx, nid;

        TreeMakerNodeStats snode;
        // tmp var, for computing lossChg
        rightStats.clear();

        for (int i : expandNodes) {
            snode = nodeStats.get(i);
            snode.leftBranchGradStats.clear();
        }

        // traverse data to find the best split
        for (int j = 0; j < featureColData.globalSampleCnt; j++) {
            feaValue = feaInst[j << 1];
            instIdx = (int) feaInst[(j << 1) + 1];

            nid = globalPosition[instIdx];
            if (nid < 0) {
                continue;
            }

            snode = nodeStats.get(nid);
            if (!snode.toExpand) {
                continue;
            }

            // first hit, init info
            if (snode.leftBranchGradStats.empty()) {
                snode.leftBranchGradStats.add(globalGradPairs, instIdx);
                snode.lastFeaValue = feaValue;

            } else {
                // the new node try to split
                if (Math.abs(feaValue - snode.lastFeaValue) > Constants.MIN_FEA_SPLIT_GAP &&
                        snode.leftBranchGradStats.getSumHess() >= params.min_child_hessian_sum) {
                    rightStats.setSubstract(snode.sumGradStats, snode.leftBranchGradStats);

                    if (rightStats.getSumHess() >= params.min_child_hessian_sum) {
                        lossChg = (float) (updateStrategy.calcGain(snode.leftBranchGradStats) + updateStrategy.calcGain(rightStats) - snode.rootGain);
                        snode.best.update(lossChg, fid, (feaValue + snode.lastFeaValue) * 0.5f);
                    }
                }
                // update the left branch sum
                snode.leftBranchGradStats.add(globalGradPairs, instIdx);
                snode.lastFeaValue = feaValue;
            }
        }
    }

    // sync best split result with other workers
    private void syncBestSplit() throws Mp4jException {
        Map<String, SplitInfo> bestSplitMap = new HashMap<>(expandNodes.size());
        for (int nid : expandNodes) {   //
            bestSplitMap.put(nid + "", nodeStats.get(nid).best);
        }

        bestSplitMap = comm.allreduceMap(bestSplitMap, Operands.OBJECT_OPERAND(new SplitInfo.SplitInfoSerializer(), SplitInfo.class), new IObjectOperator<SplitInfo>() {
            @Override
            public SplitInfo apply(SplitInfo t1, SplitInfo t2) {
                if (t1.needReplace(t2.getLossChg(), t2.getSplitIndex())) {
                    return t2;
                } else {
                    return t1;
                }
            }
        });

        for (int nid : expandNodes) {
            nodeStats.get(nid).best = bestSplitMap.get(nid + "").deepClone();
        }
    }

    // reset globalPosition of each data points after split is created in the tree
    private void resetPosition() throws Mp4jException {
        int nid, sampleOffset, fid;
        TreeNode treeNode;
        for (int sid = trainData.xRowRange[0]; sid < trainData.xRowRange[1]; sid++) {
            nid = globalPosition[sid];
            if (nid < 0) {
                continue;
            }
            treeNode = tree.getNode(nid);
            if (!treeNode.isLeaf()) {
                sampleOffset = sid - trainData.xRowRange[0];
                fid = treeNode.getSplitFeatureIndex();
                if (Float.intBitsToFloat(trainData.getFeatureVal(sampleOffset, fid)) < treeNode.getSplitCond()) {
                    globalPosition[sid] = treeNode.getLeftChild();
                } else {
                    globalPosition[sid] = treeNode.getRightChild();
                }
            }
        }
        globalPosition = comm.allgatherArray(globalPosition, Operands.INT_OPERAND(), trainData.globalSampleAssignFrom, trainData.globalSampleAssignTo);
    }

    // update queue expand add in new leaves
    private void updateExpandQueue() {
        List<Integer> newNodes = new ArrayList<>();
        TreeNode node;
        for (int nid : expandNodes) {
            node = tree.getNode(nid);
            if (!node.isLeaf()) {
                newNodes.add(node.getLeftChild());
                newNodes.add(node.getRightChild());
            }
            nodeStats.get(nid).toExpand = false;
        }
        expandNodes = newNodes;
    }

}
