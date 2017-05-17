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
import com.fenbi.ytklearn.data.Tuple;
import com.fenbi.ytklearn.data.gbdt.*;
import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.exception.YtkLearnException;
import com.fenbi.ytklearn.param.gbdt.GBDTOptimizationParams;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.fenbi.ytklearn.utils.LogUtils;
import com.fenbi.ytklearn.utils.RandomUtils;

import java.util.*;

/**
 * data-parallel tree maker
 * @author wufan
 * @author xialong
 */

public class DataParallelTreeMaker implements ITreeMaker {

    // statics for node entry, used for tree construction
    private static class TreeMakerNodeStats {
        public GradStats sumGradStats;
        public GradStats leftBranchGradStats;
        public int lastFeaValue;

        public float rootGain;
        public float nodeValue;
        public long sampleCnt;
        public SplitInfo best;

        public TreeMakerNodeStats() {
            sumGradStats = new GradStats();
            leftBranchGradStats = new GradStats();
            rootGain = 0.0f;
            nodeValue = 0.0f;
            sampleCnt = 0;
            lastFeaValue = 0;
            best = new SplitInfo();
        }

        public void clear() {
            sumGradStats.clear();
            leftBranchGradStats.clear();
            rootGain = 0.0f;
            nodeValue = 0.0f;
            sampleCnt = 0;
            lastFeaValue = 0;
            best.clear();
        }
    }

    private static class ExpandNode {
        int nid;
        float lossChg;
        int depth;
        int seq;

        ExpandNode(int nid, float lossChg, int depth, int seq) {
            this.nid = nid;
            this.lossChg = lossChg;
            this.depth = depth;
            this.seq = seq;
        }

        public String toString() {
            return "nid=" + nid
                    + ", lossChg=" + lossChg
                    + ", depth=" + depth
                    + ", seq=" + seq;
        }
    }

    // compare by lossChg, loss-wise growing(large lossChg first)
    private static Comparator<ExpandNode> lossChgComparator = (n1, n2) -> {
        int res = Float.compare(n2.lossChg, n1.lossChg);
        if (res != 0) {
            return res;
        } else {
            return Integer.compare(n1.seq, n2.seq);
        }
    };

    // compare by sequenceId, level-wise growing
    private static Comparator<ExpandNode> sequenceComparator = (n1, n2) -> Integer.compare(n1.seq, n2.seq);

    public LogUtils LOG_UTILS;
    private final ThreadCommSlave comm;
    private final int rank;
    private final int threadId;
    private final boolean owner;

    private final GBDTOptimizationParams params;
    private TimeStats timeStats;
    private TimeStats totalTimeStats;

    private UpdateStrategy updateStrategy;
    private Tree tree;

    // save the shuffled and sample index of each feature index, for col sample
    private int[] feaIndex;
    private int firstSampledFeaIndexInThread;

    // used in sorting samples by node index
    private SamplePositionData position;

    // data set info
    private GBDTCoreData trainData;
    private long sampleNum;
    private int featureDim;

    // used in sync grad sum
    private int groupId;
    private float[][] localGradPairs;
    private int[] featureStartIndex;
    private HistogramBuilder histBuilder;
    private HistogramPool histPool;

    // statistics for each constructed node
    private List<TreeMakerNodeStats> nodeStats;
    // tmp var, for computing lossChg
    private GradStats rightStats;
    // queue of nodeId to be expanded
    private Queue<ExpandNode> expandQueue;

    // build histogram
    private int histMissCnt;
    private long[] leafSampleCnt;
    private int smallLeafNid;
    private int largeLeafNid;
    private double[] smallLeafGradSum;
    private double[] largeLeafGradSum;

    private double[] smallLeafThreadFeatGradSum;
    private double[] largeLeafThreadFeatGradSum;

    public DataParallelTreeMaker(ThreadCommSlave comm,
                                 GBDTOptimizationParams params,
                                 GBDTCoreData trainData,
                                 SamplePositionData position,
                                 FeatureApprData feaApprData
    ) throws Mp4jException {

        this.comm = comm;
        this.rank = comm.getRank();
        this.threadId = comm.getThreadId();
        this.owner = (rank == 0 && threadId == 0);
        this.LOG_UTILS = new LogUtils(comm, params.verbose);

        this.params = params;
        this.updateStrategy = new UpdateStrategy(params);

        this.trainData = trainData;
        this.sampleNum = trainData.sampleNum;
        this.featureDim = trainData.usefulFeatureDim;

        this.localGradPairs = trainData.gradPairs;

        this.featureStartIndex = feaApprData.getFeaSplitValStartIndex();
        this.histBuilder = new HistogramBuilder(comm, feaApprData);

        // init histPool
        int capacity;
        if (params.histogram_pool_capacity > 0) {
            capacity = (int) (params.histogram_pool_capacity * Constants.MB / HistogramPool.UNIT / feaApprData.getNumBin());
            capacity = Math.max(2, capacity);
            capacity = Math.min(capacity, params.max_leaf_cnt);
        } else {
            capacity = params.max_leaf_cnt;
        }
        int numGrad = feaApprData.getGlobalHistAssignCount()[rank][threadId];
        int[] gradIndexRange = new int[]{feaApprData.getGlobalHistAssignFrom()[rank][threadId], feaApprData.getGlobalHistAssignFrom()[rank][threadId] + numGrad};
        this.histPool = new HistogramPool(capacity, numGrad, gradIndexRange);
        LOG_UTILS.verboseInfo("histogram pool size:" + capacity + ", grad num:" + numGrad, false);

        this.smallLeafGradSum = new double[feaApprData.getNumBin() * 2];
        this.largeLeafGradSum = new double[feaApprData.getNumBin() * 2];

        this.position = position;
        this.leafSampleCnt = new long[2];
        this.histMissCnt = 0;
        this.totalTimeStats = new TimeStats();
        this.timeStats = new TimeStats();

        // tmp var, for computing lossChg
        this.rightStats = new GradStats();
        this.nodeStats = new ArrayList<>(Constants.RESERVE_NUM);
        // expand new node
        if (params.tree_grow_policy.equals(TreeGrowPolicy.LEVEL_WISE)) {
            this.expandQueue = new PriorityQueue<>(Constants.RESERVE_NUM, sequenceComparator);
        } else if (params.tree_grow_policy.equals(TreeGrowPolicy.LOSSCHG_WISE)) {
            this.expandQueue = new PriorityQueue<>(Constants.RESERVE_NUM, lossChgComparator);
        } else {
            throw new YtkLearnException("tree maker policy(" + params.tree_grow_policy.getName() + ") invalid!");
        }

    }

    public Tree make(int groupId) throws Mp4jException {
        this.groupId = groupId;
        long start = System.currentTimeMillis();

        initAssistData();
        tree = new Tree();
        tree.init();

        // init root node and find best split for root
        int seq = 0;
        int rootId = 0;
        initTreeMakerNodeStats(rootId);
        findBestSplit(rootId);
        expandQueue.add(new ExpandNode(rootId, nodeStats.get(rootId).best.getLossChg(), 0, seq++));

        int numLeaf = 1;
        while (!expandQueue.isEmpty()) {
            ExpandNode expandNode = expandQueue.poll();
            TreeNode treeNode = tree.getNode(expandNode.nid);

            if (expandNode.lossChg <= params.min_split_loss
                    || (params.max_depth >= 0 && params.max_depth == expandNode.depth)
                    || (params.max_leaf_cnt > 0 && params.max_leaf_cnt == numLeaf)
                    || (params.min_split_samples > 0 && nodeStats.get(expandNode.nid).sampleCnt < params.min_split_samples)) {
                treeNode.setLeaf(nodeStats.get(expandNode.nid).nodeValue * params.regularization.learningRate);

            } else {
                tree.addChilds(expandNode.nid);
                // each split make leaf cnt inc by 1v
                numLeaf++;
                // reset sample position and sort samples by nodeid after split is created in the tree
                position.resetPosition(trainData, expandNode.nid, tree);

                int leftChild = treeNode.getLeftChild();
                int rightChild = treeNode.getRightChild();
                initTreeMakerNodeStats(leftChild, rightChild);

                if ((params.max_depth >= 0 && params.max_depth == expandNode.depth + 1)
                        || (params.max_leaf_cnt > 0 && params.max_leaf_cnt == numLeaf)
                        || (params.min_split_samples > 0
                        && nodeStats.get(leftChild).sampleCnt < params.min_split_samples
                        && nodeStats.get(rightChild).sampleCnt < params.min_split_samples)) {
                    computeLeafStats(expandNode.nid, leftChild, rightChild);
                    tree.getNode(leftChild).setLeaf(nodeStats.get(leftChild).nodeValue * params.regularization.learningRate);
                    tree.getNode(rightChild).setLeaf(nodeStats.get(rightChild).nodeValue * params.regularization.learningRate);

                } else {
                    findBestSplit(expandNode.nid, leftChild, rightChild);
                    expandQueue.add(new ExpandNode(leftChild, nodeStats.get(leftChild).best.getLossChg(), expandNode.depth + 1, seq++));
                    expandQueue.add(new ExpandNode(rightChild, nodeStats.get(rightChild).best.getLossChg(), expandNode.depth + 1, seq++));
                }
            }
        }

        // update node stats for debug
        for (int nid = 0; nid < tree.getNodeNum(); nid++) {
            updateTreeNodeStat(nid);
        }
        timeStats.totalTime = System.currentTimeMillis() - start;
        timeStats.findBestSplit -= timeStats.syncBestSplit;
        totalTimeStats.add(timeStats);

        LOG_UTILS.verboseInfo("build histHistogram, miss count=" + histMissCnt);
        LOG_UTILS.verboseInfo(timeStats.getStats());

        return tree;
    }

    public TimeStats getTotalTimeStats() {
        return totalTimeStats;
    }

    // init a new node and compute best split
    private void findBestSplit(int nid) throws Mp4jException {
        long start = System.currentTimeMillis();
        buildHist(nid);
        long cost = System.currentTimeMillis() - start;
        timeStats.buildHist += cost;

        // compute the sum of gradient, compute once, use the first feature to compute stats
        start = System.currentTimeMillis();
        updateTreeMakerNodeStats(nid, smallLeafThreadFeatGradSum, true);
        cost = System.currentTimeMillis() - start;
        timeStats.initStats += cost;

        start = System.currentTimeMillis();
        findBestSplit(nid, smallLeafThreadFeatGradSum);
        cost = System.currentTimeMillis() - start;
        timeStats.findBestSplit += cost;
    }

    private void findBestSplit(int parent, int leftChild, int rightChild) throws Mp4jException {
        long start = System.currentTimeMillis();
        buildHist(parent, leftChild, rightChild);
        long cost = System.currentTimeMillis() - start;
        timeStats.buildHist += cost;

        // compute the sum of gradient, compute once, use the first feature to compute stats
        start = System.currentTimeMillis();
        updateTreeMakerNodeStats(smallLeafNid, smallLeafThreadFeatGradSum, true);
        updateTreeMakerNodeStats(largeLeafNid, largeLeafThreadFeatGradSum, true);
        cost = System.currentTimeMillis() - start;
        timeStats.initStats += cost;

        start = System.currentTimeMillis();
        findBestSplit(smallLeafNid, smallLeafThreadFeatGradSum);
        findBestSplit(largeLeafNid, largeLeafThreadFeatGradSum);
        cost = System.currentTimeMillis() - start;
        timeStats.findBestSplit += cost;

    }

    private void computeLeafStats(int parent, int leftChild, int rightChild) throws Mp4jException {
        long start = System.currentTimeMillis();
        // left child has less samples
        if (nodeStats.get(leftChild).sampleCnt < nodeStats.get(rightChild).sampleCnt) {
            smallLeafNid = leftChild;
            largeLeafNid = rightChild;
        } else {
            smallLeafNid = rightChild;
            largeLeafNid = leftChild;
        }

        // compute smallChild gram sum
        int[] positionAndIndex = position.getPosAndSampleIndex();
        Tuple<Integer, Integer> posInterval = position.getNodeSampleInterval().get(smallLeafNid);
        double[] totalGlobalGradSum = new double[]{0, 0};

        int nidOffset;
        int sampleIdx, gradIdx;
        int index2D, index1D;
        for (int i = posInterval.v1; i < posInterval.v2; i++) {
            nidOffset = i << 1;
            // not sampled
            if (positionAndIndex[nidOffset] < 0)
                continue;
            sampleIdx = positionAndIndex[nidOffset + 1];
            index2D = sampleIdx / trainData.DENSE_MAX_1D_SAMPLE_CNT;
            index1D = sampleIdx % trainData.DENSE_MAX_1D_SAMPLE_CNT;
            gradIdx = (index1D * trainData.numTreeInGroup + groupId) << 1;

            totalGlobalGradSum[0] += localGradPairs[index2D][gradIdx];
            totalGlobalGradSum[1] += localGradPairs[index2D][gradIdx + 1];
        }

        // sync grad sum with all workers
        totalGlobalGradSum = comm.allreduceArray(totalGlobalGradSum, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, totalGlobalGradSum.length);
        long cost = System.currentTimeMillis() - start;
        timeStats.buildHist += cost;

        updateTreeMakerNodeStats(smallLeafNid, totalGlobalGradSum, false);

        TreeMakerNodeStats stats = nodeStats.get(largeLeafNid);
        stats.sumGradStats.setSubstract(nodeStats.get(parent).sumGradStats, nodeStats.get(smallLeafNid).sumGradStats);
        stats.rootGain = (float) updateStrategy.calcGain(stats.sumGradStats);
        stats.nodeValue = (float) updateStrategy.calcNodeValue(stats.sumGradStats);
    }

    private void updateTreeNodeStat(int nid) {
        TreeNodeStat tstat = tree.getNodeStat(nid);
        TreeMakerNodeStats mstat = nodeStats.get(nid);
        tstat.setLossChg(mstat.best.getLossChg());
        tstat.setHessSum((float) (mstat.sumGradStats.getSumHess()));
        tstat.setNodeSampleCnt(mstat.sampleCnt);
    }

    private void initAssistData() throws Mp4jException {
        timeStats.clear();
        histPool.clear();
        rightStats.clear();
        expandQueue.clear();
        // all node init to root
        position.clear();

        int[] positionAndIndex = position.getPosAndSampleIndex();
        // subsampling
        int sampleCnt = 0;
        Random rand = new Random();
        if (params.subsample < 1.0f) {
            for (int i = 0; i < sampleNum; i++) {
                int j = i << 1;
                if (!RandomUtils.sampleBinary(rand, params.subsample)) {
                    positionAndIndex[j] = -1;
                } else {
                    sampleCnt++;
                }
            }
            // some workers may have no samples
            LOG_UTILS.verboseInfo(String.format("do subsampling, subsampleRate=%f, sample out %d of all %d instances", params.subsample, sampleCnt, sampleNum), false);
            // sort position, put useless sample in the left side
            position.resetPositionAfterSample();

        } else {
            LOG_UTILS.verboseInfo("no subsample, use all instances");
        }

        // init feature index
        int[] localFeaIndex = new int[featureDim];
        for (int i = 0; i < featureDim; i++) {
            localFeaIndex[i] = i;
        }

        // do feature sampling
        firstSampledFeaIndexInThread = -1;
        if (params.feature_sample_rate < 1.0f) {
            // get a global seed
            long seed = System.nanoTime();
            seed = comm.allreduce(seed, Operands.LONG_OPERAND(), Operators.Long.MAX);

            // sample out features
            RandomUtils.shuffle(seed, localFeaIndex);
            int nFeaSam = Math.round(params.feature_sample_rate * featureDim);
            if (nFeaSam < 1) {
                LOG_UTILS.importantInfo(String.format("[GBDT]: feature sample_rate(%f) is too small and no feature can be included, set sample out feature cnt to 1", params.feature_sample_rate));
            }
            nFeaSam = Math.max(1, nFeaSam);
            feaIndex = new int[nFeaSam];
            for (int i = 0; i < feaIndex.length; i++) {
                feaIndex[i] = localFeaIndex[i];
            }
            Arrays.sort(feaIndex);

            if (trainData.xColRange[0] < trainData.xColRange[1]) {
                for (int i = 0; i < feaIndex.length; i++) {
                    if (feaIndex[i] >= trainData.xColRange[0] && feaIndex[i] < trainData.xColRange[1]) {
                        firstSampledFeaIndexInThread = feaIndex[i];
                        break;
                    }
                }
            }
            LOG_UTILS.verboseInfo("do feature sample complete! feature_sample_rate=" + params.feature_sample_rate + ", sample out " + nFeaSam + " features");
//            LOG_UTILS.verboseInfo("do feature sample complete! feature details:" + StringUtils.join(feaIndex, ","));

        } else {
            feaIndex = localFeaIndex;
            // no feature assigned to this thread
            if (trainData.xColRange[0] < trainData.xColRange[1]) {
                firstSampledFeaIndexInThread = trainData.xColRange[0];
            }
            LOG_UTILS.verboseInfo("no featureSample, use all features");
        }
    }

    private double[] getSubstractGradSum(int parent, double[] parentGradSum) throws Mp4jException {
        histBuilder.substract(parentGradSum, smallLeafThreadFeatGradSum, largeLeafGradSum);
        histPool.releaseHist(parent);
        return histPool.setHist(largeLeafNid, largeLeafGradSum);
    }

    private double[] syncGlobalGradSum(int nid, double[] localGradSum) throws Mp4jException {
        // in fact, no need to set back
        localGradSum = histBuilder.buildHist(position.getNodeSampleInterval().get(nid),
                position.getPosAndSampleIndex(), trainData, feaIndex, groupId, localGradSum, timeStats.buildDetail);
        return histPool.setHist(nid, localGradSum);
    }

    private void buildHist(int nid) throws Mp4jException {
        smallLeafThreadFeatGradSum = syncGlobalGradSum(nid, smallLeafGradSum);
    }

    private void buildHist(int parent, int leftChild, int rightChild) throws Mp4jException {
        double[] parentGradSum = histPool.getHist(parent);

        // left child has less samples
        if (nodeStats.get(leftChild).sampleCnt < nodeStats.get(rightChild).sampleCnt) {
            smallLeafNid = leftChild;
            largeLeafNid = rightChild;
        } else {
            smallLeafNid = rightChild;
            largeLeafNid = leftChild;
        }

        smallLeafThreadFeatGradSum = syncGlobalGradSum(smallLeafNid, smallLeafGradSum);
        if (parentGradSum != null) {
            largeLeafThreadFeatGradSum = getSubstractGradSum(parent, parentGradSum);
        } else {
            largeLeafThreadFeatGradSum = syncGlobalGradSum(largeLeafNid, largeLeafGradSum);
            histMissCnt++;
        }
    }

    private void initTreeMakerNodeStats(int nid) throws Mp4jException {
        CheckUtils.check(nodeStats.size() >= nid, "[GBDT] inner error! nodeStats size(%d) >= %d false", nodeStats.size(), nid);
        if (nodeStats.size() == nid) {
            nodeStats.add(new TreeMakerNodeStats());
        } else {
            nodeStats.get(nid).clear();
        }
        long sampleCnt = position.getSampleCntInNode(nid);
        sampleCnt = comm.allreduce(sampleCnt, Operands.LONG_OPERAND(), Operators.Long.SUM);
        nodeStats.get(nid).sampleCnt = sampleCnt;
    }

    private void initTreeMakerNodeStats(int leftChildNid, int rightChildNid) throws Mp4jException {
        CheckUtils.check(leftChildNid + 1 == rightChildNid, "[GBDT] inner error! leftChildId(%d) + 1 != rightChildId(%d)", leftChildNid, rightChildNid);
        CheckUtils.check(nodeStats.size() >= leftChildNid, "[GBDT] inner error! nodeStats size(%d) >= %d false", nodeStats.size(), leftChildNid);
        if (nodeStats.size() == leftChildNid) {
            nodeStats.add(new TreeMakerNodeStats());
        } else {
            nodeStats.get(leftChildNid).clear();
        }
        if (nodeStats.size() == rightChildNid) {
            nodeStats.add(new TreeMakerNodeStats());
        } else {
            nodeStats.get(rightChildNid).clear();
        }

        leafSampleCnt[0] = position.getSampleCntInNode(leftChildNid);
        leafSampleCnt[1] = position.getSampleCntInNode(rightChildNid);
        leafSampleCnt = comm.allreduceArray(leafSampleCnt, Operands.LONG_OPERAND(), Operators.Long.SUM, 0, leafSampleCnt.length);
        nodeStats.get(leftChildNid).sampleCnt = leafSampleCnt[0];
        nodeStats.get(rightChildNid).sampleCnt = leafSampleCnt[1];
    }

    private void updateTreeMakerNodeStats(int nid, double[] globalGradSum, boolean needComm) throws Mp4jException {
        TreeMakerNodeStats stats = nodeStats.get(nid);
        if (!needComm) {
            stats.sumGradStats.set(globalGradSum[0], globalGradSum[1]);

        } else {
            double[] gradStats = new double[]{Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY};

            if (firstSampledFeaIndexInThread != -1) {
                for (int i = featureStartIndex[firstSampledFeaIndexInThread]; i < featureStartIndex[firstSampledFeaIndexInThread + 1]; i++) {
                    int startGradSumoffset = i - featureStartIndex[trainData.xColRange[0]];
                    stats.sumGradStats.add(globalGradSum, startGradSumoffset);
                }
                gradStats[0] = stats.sumGradStats.getSumGrad();
                gradStats[1] = stats.sumGradStats.getSumHess();
            }
//            gradStats = comm.allreduceArray(gradStats, Operands.DOUBLE_OPERAND(), Operators.Double.MAX, 0, gradStats.length);
            gradStats = comm.allreduceArrayRpc(gradStats, Operands.DOUBLE_OPERAND(), Operators.Double.MAX);
            stats.sumGradStats.set(gradStats[0], gradStats[1]);

              // may cause inconsistent
//            int fid = trainData.xColRange[0];
//            for (int i = 0; i < featureStartIndex[fid + 1] - featureStartIndex[fid]; i++) {
//                stats.sumGradStats.add(globalGradSum, i);
//            }
        }

        stats.rootGain = (float) updateStrategy.calcGain(stats.sumGradStats);
        stats.nodeValue = (float) updateStrategy.calcNodeValue(stats.sumGradStats);

    }

    private boolean findBestSplit(int nid, double[] globalGradSum) throws Mp4jException {
        TreeMakerNodeStats mstats = nodeStats.get(nid);
        if (updateStrategy.canSplit(mstats.sumGradStats.getSumHess(), mstats.sampleCnt)) {
            for (int fid : feaIndex) {
                if (fid >= trainData.xColRange[0] && fid < trainData.xColRange[1]) {
                    enumerateSplit(nid, fid, featureStartIndex, globalGradSum);
                }
            }
            // sync best split with other workers
            long start = System.currentTimeMillis();
            syncBestSplit(nid);
            timeStats.syncBestSplit += System.currentTimeMillis() - start;
            SplitInfo spBest = nodeStats.get(nid).best;
            if (spBest.getLossChg() > params.min_split_loss) {
                int[] interval = spBest.getSplitSlotInterval();
                tree.getNode(nid).setSplit(spBest.getSplitIndex(), interval[0], interval[1]);
                return true;
            }
        }
        return false;
    }

    // enumerate split values of specific feature
    private void enumerateSplit(int nid, int fid, int[] feaStartIndex, double[] globalGradSum) throws Mp4jException {
        TreeMakerNodeStats stats = nodeStats.get(nid);
        stats.leftBranchGradStats.clear();
        rightStats.clear();  //tmp var, for computing lossChg

        float lossChg = 0.0f;
        int feaValue = 0;
        // traverse grad appro vec to find the best split
        int feaSlotNum = feaStartIndex[fid + 1] - feaStartIndex[fid];
        int startGradSumoffset = featureStartIndex[fid] - featureStartIndex[trainData.xColRange[0]];
        int globalGradSumOffet = 0;
        for (int j = 0; j < feaSlotNum; j++) {
            feaValue = j;
            globalGradSumOffet = j + startGradSumoffset;

            // has no sample in this feature slot
            if (!hasSample(globalGradSum, globalGradSumOffet)) {
                continue;
            }

            // first hit, init info
            if (stats.leftBranchGradStats.empty()) {
                stats.leftBranchGradStats.add(globalGradSum, globalGradSumOffet);
                stats.lastFeaValue = j;

            } else {
                if (stats.leftBranchGradStats.getSumHess() >= params.min_child_hessian_sum) {
                    rightStats.setSubstract(stats.sumGradStats, stats.leftBranchGradStats);

                    if (rightStats.getSumHess() >= params.min_child_hessian_sum) {
                        lossChg = (float) (updateStrategy.calcGain(stats.leftBranchGradStats) + updateStrategy.calcGain(rightStats) - stats.rootGain);
                        stats.best.update(lossChg, fid, stats.lastFeaValue, feaValue);
                    }
                }
                // update the left branch sum
                stats.leftBranchGradStats.add(globalGradSum, globalGradSumOffet);
                stats.lastFeaValue = feaValue;
            }
        }
    }

    // sync best split result with other workers
    private void syncBestSplit(int nid) throws Mp4jException {
//         SplitInfo splitInfo = comm.allreduce(nodeStats.get(nid).best, Operands.OBJECT_OPERAND(new SplitInfo.SplitInfoSerializer(), SplitInfo.class), new IObjectOperator<SplitInfo>() {
         SplitInfo splitInfo = comm.allreduceRpc(nodeStats.get(nid).best, Operands.OBJECT_OPERAND(new SplitInfo.SplitInfoSerializer(), SplitInfo.class), new IObjectOperator<SplitInfo>() {
            @Override
            public SplitInfo apply(SplitInfo t1, SplitInfo t2) {
                if (t1.needReplace(t2.getLossChg(), t2.getSplitIndex())) {
                    return t2;
                } else {
                    return t1;
                }
            }
        });
        nodeStats.get(nid).best = splitInfo.deepClone();
    }

    // judge whether the feature value slot has samples
    // second order gradient will not be zero in ObjFunction implementation
    private boolean hasSample(double[] globalGradSum, int idx) {
        idx <<= 1;
        if (globalGradSum[idx] == .0f && globalGradSum[idx + 1] == .0f) {
            return false;
        }
        return true;
    }
}
