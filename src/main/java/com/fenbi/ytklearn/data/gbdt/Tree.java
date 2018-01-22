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

package com.fenbi.ytklearn.data.gbdt;

import com.fenbi.ytklearn.data.Constants;
import com.fenbi.ytklearn.data.Tuple;
import com.fenbi.ytklearn.feature.gbdt.FeatureSplitType;
import com.fenbi.ytklearn.utils.NumConvertUtils;
import com.fenbi.ytklearn.utils.CheckUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * regression tree data structure
 * @author wufan
 */

public class Tree {
    //todo: \\]
    public static String innerNodePatternStr = "(\\S+):\\[f_(\\S+)<=(\\S+)\\] yes=(\\S+),no=(\\S+),missing=(\\S+),gain=(\\S+),hess_sum=(\\S+),sample_cnt=(\\S+)";
    public static String leafPatternStr = "(\\S+):leaf=(\\S+),hess_sum=(\\S+),sample_cnt=(\\S+)";

    private List<TreeNode> nodes;
    private List<TreeNodeStat> stats;

    public Tree() {
        nodes = new ArrayList<>();
        stats = new ArrayList<>();
    }

    protected int AllocTreeNode() {
        int nid = nodes.size();
        CheckUtils.check(nid < Integer.MAX_VALUE, "number of nodes in the tree exceed Integer.MAX_VALUE");
        nodes.add(new TreeNode());
        stats.add(new TreeNodeStat());
        return nid;
    }

    public void addChilds(int nid) {
        int left = AllocTreeNode();
        int right = AllocTreeNode();
        nodes.get(nid).setLeftChild(left);
        nodes.get(nid).setRightChild(right);
        nodes.get(left).setParent(nid);
        nodes.get(right).setParent(nid);
    }

    public int getDepth(int nid) {
        int depth = 0;
        while (!nodes.get(nid).isRoot()) {
            depth++;
            nid = nodes.get(nid).getParent();
        }
        return depth;
    }

    private int maxDepthToLeaf(int nid) {
        if (nodes.get(nid).isLeaf()) {
            return 0;
        }
        return Math.max(maxDepthToLeaf(nodes.get(nid).getLeftChild()),
                maxDepthToLeaf(nodes.get(nid).getRightChild())) + 1;
    }

    public int getMaxDepth() {
        int maxd = maxDepthToLeaf(0);
        return maxd;
    }

    private void getLeafCnt(int nid, int[] leafCnt) {
        TreeNode node = nodes.get(nid);
        if (node.isLeaf()) {
            leafCnt[0]++;
        } else {
            getLeafCnt(node.getLeftChild(), leafCnt);
            getLeafCnt(node.getRightChild(), leafCnt);
        }
    }

    public int getLeafCnt() {
        int[] leafCnt = new int[]{0};
        getLeafCnt(0, leafCnt);
        return leafCnt[0];
    }


    public double predict(Map<String, Float> features) {
        // for online predict, k-v features
        int pid = getLeafIndex(features);
        return nodes.get(pid).getLeafValue();

    }

    public int getLeafIndex(Map<String, Float> features) {
        int pid = 0;
        String splitFeaName = "";
        while (!nodes.get(pid).isLeaf()) {
            splitFeaName = nodes.get(pid).getSplitFeatureName();
            Float feaVal = features.get(splitFeaName);
            if (feaVal == null) {
                pid = getNext(pid, 0, true);
            } else {
                pid = getNext(pid, feaVal, false);
            }
        }
        return pid;
    }

    public float predict(int[] feaMatrix, int instOffset, boolean isOriginTree) {
        // used in train phase, featureMatrix flatten
        int pid = getLeafIndex(feaMatrix, instOffset, isOriginTree);
        return nodes.get(pid).getLeafValue();
    }

    public int getLeafIndex(int[] feaMatrix, int instOffset, boolean isOriginTree) {
        int pid = 0;
        int splitIndex = 0;
        while (!nodes.get(pid).isLeaf()) {
            splitIndex = nodes.get(pid).getSplitFeatureIndex();
            if (isOriginTree) {
                pid = getNext(pid, NumConvertUtils.int2float(feaMatrix[instOffset + splitIndex]), feaMatrix[instOffset + splitIndex] == Constants.INT_MISSING_VALUE);
            } else { //train phase: missing value is filled
                pid = getNext(pid, feaMatrix[instOffset + splitIndex], false);
            }
        }
        return pid;
    }

    private int getNext(int pid, float feaVal, boolean isMissing) {
        TreeNode node = nodes.get(pid);
        if (isMissing) {
            return node.getDefualtChild();
        } else {
            float splitVal = node.getSplitCond();
            if (feaVal <= splitVal) {
                return node.getLeftChild();
            } else {
                return node.getRightChild();
            }
        }
    }

    public int getNodeNum() {
        return nodes.size();
    }

    public TreeNode getNode(int index) {
        return nodes.get(index);
    }

    public TreeNodeStat getNodeStat(int index) {
        return stats.get(index);
    }

    // init a tree, add root
    public void init() {
        TreeNode node = new TreeNode();
        node.setParent(-1);
        node.setLeaf(0.0f);
        nodes.add(node);
        stats.add(new TreeNodeStat());

    }

    public void loadModel(BufferedReader reader, Pattern innerNodePattern, Pattern leafPattern) throws IOException, ClassNotFoundException {
        CheckUtils.check(nodes.size() == 0, "[GBDT] RegTree load model error! node num is not 0, num_nodes=" + nodes.size());
        int nodeNum = Integer.parseInt(reader.readLine().split(",")[1].split("=")[1]);
        CheckUtils.check(nodeNum > 0,
                String.format("[GBDT] load invalid model, nodeNum=%d should > 0", nodeNum));

        nodes = new ArrayList<>(nodeNum);
        stats = new ArrayList<>(nodeNum);
        for (int i = 0; i < nodeNum; i++) {
            nodes.add(new TreeNode());
            stats.add(new TreeNodeStat());
        }

        for (int i = 0; i < nodeNum; i++) {
            String line = reader.readLine();
            if (line.indexOf("leaf") >= 0) {
                parseLeaf(leafPattern, line);
            } else {
                parseInnerNode(innerNodePattern, line);
            }

        }
    }

    private void parseLeaf(Pattern leafP, String nodeStr) {
        Matcher m = leafP.matcher(nodeStr);
        CheckUtils.check(m.find() == true, "[GBDT] parse model error, leaf line:" + nodeStr);
        int nid = Integer.parseInt(m.group(1));
        float leafVal = Float.parseFloat(m.group(2));
        float hessSum = Float.parseFloat(m.group(3));
        long sampleCnt = Long.parseLong(m.group(4));

        nodes.get(nid).setLeaf(leafVal);
        stats.get(nid).setHessSum(hessSum);
        stats.get(nid).setNodeSampleCnt(sampleCnt);
    }

    private void parseInnerNode(Pattern innerNodeP, String nodeStr) {
        Matcher m = innerNodeP.matcher(nodeStr);
        CheckUtils.check(m.find() == true, "[GBDT] parse model error, non-leaf line:" + nodeStr);
        int nid = Integer.parseInt(m.group(1));
        String splitFeatName =  m.group(2);
        float splitFeatVal = Float.parseFloat(m.group(3));

        int leftChild =  Integer.parseInt(m.group(4));
        int rightChild =  Integer.parseInt(m.group(5));
        boolean isDefaultLeft = Integer.parseInt(m.group(6)) == leftChild;
        float gain = Float.parseFloat(m.group(7));
        float hessSum = Float.parseFloat(m.group(8));
        long sampleCnt = Long.parseLong(m.group(9));

        TreeNode node = nodes.get(nid);
        node.setLeftChild(leftChild);
        node.setRightChild(rightChild);
        node.setSplitFeatureName(splitFeatName);
        node.setSplit(-1, splitFeatVal);
        node.setDefaultDirection(isDefaultLeft);
        nodes.get(leftChild).setParent(nid);
        nodes.get(rightChild).setParent(nid);

        TreeNodeStat stat = stats.get(nid);
        stat.setLossChg(gain);
        stat.setHessSum(hessSum);
        stat.setNodeSampleCnt(sampleCnt);
    }

    public String dumpModel(int iter, boolean withStats) {
        int nodeNum = nodes.size();
        CheckUtils.check(nodeNum > 0,
                String.format("[GBDT] save invalid model, nodeNum=%d should > 0", nodeNum));
        StringBuffer sb = new StringBuffer("");
        sb.append(String.format("booster[%d] depth=%d,node_num=%d,leaf_cnt=%d\n", iter + 1, getMaxDepth(), nodeNum, getLeafCnt()));
        dump(0, sb, 0, withStats);
        return sb.toString();
    }

    private void dump(int nid, StringBuffer sb, int depth, boolean withStats) {
        for (int i = 0; i < depth; i++) {
            sb.append("\t");
        }
        TreeNode node = nodes.get(nid);
        if (node.isLeaf()) {
            sb.append(String.format("%d:leaf=%s", nid, Float.toString(node.getLeafValue())));
            if (withStats) {
                sb.append(getNodeStat(nid).print(true));
            }
            sb.append("\n");
        } else {
            String splitFeaName = node.getSplitFeatureName();
            float cond = node.getSplitCond();
            sb.append(String.format("%d:[f_%s<=%s] yes=%d,no=%d,missing=%d",
                    nid, splitFeaName, Float.toString(cond), node.getLeftChild(), node.getRightChild(), node.getDefualtChild()));
            if (withStats) {
                sb.append(getNodeStat(nid).print(false));
            }
            sb.append("\n");
            dump(node.getLeftChild(), sb, depth + 1, withStats);
            dump(node.getRightChild(), sb, depth + 1, withStats);
        }
    }

    public void convertFeatureSplitValueInModel(Map<Integer, float[]> globalFeaSplitValsSorted, FeatureSplitType fsType) {
        CheckUtils.check(globalFeaSplitValsSorted != null && globalFeaSplitValsSorted.size() > 0,
                "[GBDT] global sorted feature split values map is empty, convert model error!");
        TreeNode node;
        for (int i = 0; i < nodes.size(); i++) {
            node = nodes.get(i);
            if (node.isLeaf()) {
                continue;
            }
            int splitFeaIndex = node.getSplitFeatureIndex();
            int[] interval = node.getSplitFeaApproSlotInterval();  // size = 2
            float[] feaSplitValSorted = globalFeaSplitValsSorted.get(splitFeaIndex);

            float splitFeaVal = fsType.getFeatureSplit(feaSplitValSorted, interval);
            node.setSplit(splitFeaIndex, splitFeaVal);
        }
    }

    public void addFeatureNameInModel(Map<Integer, String> fIndex2Name) {
        // convert feature split index to name, called before saveModel or dumpModel
        TreeNode node;
        int feaIndex;
        String feaName;
        for (int i = 0; i < nodes.size(); i++) {
            node = nodes.get(i);
            if (node.isLeaf()) {
                continue;
            }
            feaIndex = node.getSplitFeatureIndex();
            feaName = fIndex2Name.get(node.getSplitFeatureIndex());
            CheckUtils.check(feaName != null, "[GBDT] inner error! can't find feature name for feature index(" + feaIndex + ")");
            node.setSplitFeatureName(feaName);
        }
    }

    public void updateFeatureIndexInModel(Map<String, Integer> fName2Index) {
        // convert feature split name to index, called after loadModel
        TreeNode node;
        Integer feaIndex;
        String feaName;
        for (int i = 0; i < nodes.size(); i++) {
            node = nodes.get(i);
            if (node.isLeaf()) {
                continue;
            }
            feaName = node.getSplitFeatureName();
            feaIndex = fName2Index.get(feaName);
            CheckUtils.check(feaIndex != null, "[GBDT] inner error! can't find feature index for feature name(" + feaName + ")");
            node.setSplitFeatureIndex(feaIndex);
        }
    }

    public List<Integer> getLeafNodes() {
        List<Integer> leafNodes = new ArrayList<>();
        TreeNode node;
        for (int i = 0; i < nodes.size(); i++) {
            node = nodes.get(i);
            if (node.isLeaf()) {
                leafNodes.add(i);
            }
        }
        return leafNodes;
    }

    public void addDefaultDirection(float[] missValueArr) {
        if (missValueArr == null || missValueArr.length == 0) {
            return;
        }
        TreeNode node;
        for (int i = 0; i < nodes.size(); i++) {
            node = nodes.get(i);
            if (node.isLeaf()) {
                continue;
            }
            int splitFeaIndex = node.getSplitFeatureIndex();
            float splitFeaVal = node.getSplitCond();
            if (missValueArr[splitFeaIndex] < splitFeaVal) {
                node.setDefaultDirection(true);
            } else {
                node.setDefaultDirection(false);
            }
        }
    }

    public void genFeatureDict(Map<String, Integer> feaDictMap) {
        TreeNode node;
        String splitFeaName;
        for (int i = 0; i < nodes.size(); i++) {
            node = nodes.get(i);
            if (node.isLeaf()) {
                continue;
            }
            splitFeaName = node.getSplitFeatureName();
            Integer index = feaDictMap.get(splitFeaName);
            if (index == null) {
                feaDictMap.put(splitFeaName, feaDictMap.size());
            }
        }
    }

    public void featureImportance(Map<String, Tuple<Integer, Double>> feaImpMap) {
        TreeNode node;
        String splitFeaName;
        for (int i = 0; i < nodes.size(); i++) {
            node = nodes.get(i);
            if (node.isLeaf()) {
                continue;
            }
            splitFeaName = node.getSplitFeatureName();
            Tuple<Integer, Double> imp = feaImpMap.get(splitFeaName);
            if (imp == null) {
                feaImpMap.put(splitFeaName, new Tuple<>(1, (double)stats.get(i).getLossChg()));
            } else {
                imp.v1++;
                imp.v2 += stats.get(i).getLossChg();
            }
        }
    }

    public static void main(String[] args){
        String nodeStr = "   1:[f_3<=27.0] yes=3,no=4,missing=3,gain=186403.17,hess_sum=143.0,sample_cnt=143";
        Pattern innerNodePattern = Pattern.compile(Tree.innerNodePatternStr);
        Pattern leafPattern = Pattern.compile(Tree.leafPatternStr);
        Matcher m = innerNodePattern.matcher(nodeStr);
        CheckUtils.check(m.find() == true, "[GBDT] parse model error, non-leaf line:" + nodeStr);
        int nid = Integer.parseInt(m.group(1));
        String splitFeatName =  m.group(2);
        float splitFeatVal = Float.parseFloat(m.group(3));

        int leftChild =  Integer.parseInt(m.group(4));
        int rightChild =  Integer.parseInt(m.group(5));
        boolean isDefaultLeft = Integer.parseInt(m.group(6)) == leftChild;
        float gain = Float.parseFloat(m.group(7));
        float hessSum = Float.parseFloat(m.group(8));
        long sampleCnt = Long.parseLong(m.group(9));
        System.out.println("nid=" + nid + ", splitFeat=" + splitFeatName);


    }

}
