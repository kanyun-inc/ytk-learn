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
import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.utils.CheckUtils;

import java.util.*;

/**
 * used for sorting samples by node index
 * @author wufan
 * @author xialong
 */

public class SamplePositionData {

    // tree node index and sample index for each sample, can hold 2^32 samples in each thread
    private int[] posAndSampleIndex;
    private int sampleNum;

    // sample interval for tree node, front closing and back opening
    private List<Tuple<Integer, Integer>> nodeSampleInterval;

    public SamplePositionData(int sampleNum) {
        this.sampleNum = sampleNum;
        posAndSampleIndex = new int[sampleNum << 1];
        nodeSampleInterval = new ArrayList<>(Constants.RESERVE_NUM);
        clear();
    }

    public void clear() {
        for (int i = 0; i < sampleNum; i++) {
            posAndSampleIndex[i << 1] = 0;
            posAndSampleIndex[(i << 1) + 1] = i;
        }
        nodeSampleInterval = new ArrayList<>(Constants.RESERVE_NUM);
        nodeSampleInterval.add(new Tuple<>(0, sampleNum));
    }

    public int[] getPosAndSampleIndex() {
        return posAndSampleIndex;
    }

    public long getSampleNum() {
        return sampleNum;
    }

    public List<Tuple<Integer, Integer>> getNodeSampleInterval() {
        return nodeSampleInterval;
    }

    public int getSampleCntInNode(int nid) {
//        if (nid < 0 || nid >= nodeSampleInterval.size()) {
//            return 0;
//        }
        return nodeSampleInterval.get(nid).v2 - nodeSampleInterval.get(nid).v1;
    }

    // sort samples, put not-sampled instances in the left and sampled instances in the right and
    public void resetPositionAfterSample() {
        int leftCnt = sortByNodeIndex(nodeSampleInterval.get(0), -1, 0, false);
        nodeSampleInterval.get(0).v1 = leftCnt;
    }

    // change the relationship between node indexes and samples after making splits
    public void resetPosition(List<Integer> expandNodes, Tree tree) {
//        LOG.debug(Thread.currentThread().getName() + " expand node num:" + expandNodes.size() + ", origin interval:" + nodeSampleInterval.size());
        for (int nid : expandNodes) {
            TreeNode node = tree.getNode(nid);
            if (node.isLeaf())
                continue;
            Tuple<Integer, Integer> interval = nodeSampleInterval.get(nid);
            sortByNodeIndex(interval, node.getLeftChild(), node.getRightChild(), true);
        }
        //for debug
//        for (int i: expandNodes) {
//            System.out.println(Thread.currentThread().getName() + " SamplePositionData[77] interval:" + i + ", v1=" + nodeSampleInterval.get(i).v1 + ", v2=" + nodeSampleInterval.get(i).v2);
//            StringBuffer sb = new StringBuffer("expandNid:" + i + ": cnt:" + (nodeSampleInterval.get(i).v2 - nodeSampleInterval.get(i).v1) + ":");
//            for (int j = nodeSampleInterval.get(i).v1; j < nodeSampleInterval.get(i).v2; j++) {
//                sb.append(posAndSampleIndex[2*j + 1]);
//                sb.append(",");
//            }
//            sb.append("\n");
//            LOG.info(sb.toString());
//            System.out.println(sb.toString());
//        }

    }

    // change the relationship between node indexes and samples after making splits
    public void resetPosition(GBDTCoreData trainData, int nid, Tree tree) {
//        LOG.debug(Thread.currentThread().getName() + " expand node num:" + expandNodes.size() + ", origin interval:" + nodeSampleInterval.size());
        TreeNode node = tree.getNode(nid);
        if (node.isLeaf()) {
            return;
        }
        // set sample position after split
        Tuple<Integer, Integer> interval = nodeSampleInterval.get(nid);
        for (int offset = interval.v1 * 2; offset < interval.v2 * 2; offset += 2) {
            int feaIndex = node.getSplitFeatureIndex();
            int sampleIndex = posAndSampleIndex[offset + 1];
            if (trainData.getFeatureVal(sampleIndex, feaIndex) < node.getSplitCond()) {
                posAndSampleIndex[offset] = node.getLeftChild();
            } else {
                posAndSampleIndex[offset] = node.getRightChild();
            }
        }
        sortByNodeIndex(interval, node.getLeftChild(), node.getRightChild(), true);
    }

    // sort samples by tree node id, put the samples belong to left node before samples belong to right node
    private int sortByNodeIndex(Tuple<Integer, Integer> posInterval, int leftChid, int rightChild, boolean addInterval) {
        CheckUtils.check(leftChid + 1 == rightChild, "left child index(%d) > right child index(%d), invalid!",
                leftChid, rightChild);
        int posStart = posInterval.v1;
        int posEnd = posInterval.v2 - 1;
//       LOG.debug("start:" + posStart + "end:" + posEnd);
        int leftCnt = 0;
        while (posStart <= posEnd) {
            while (posStart <= posEnd && posAndSampleIndex[posStart << 1] == leftChid) {
                posStart++;
                leftCnt++;
            }
            while (posStart <= posEnd && posAndSampleIndex[posEnd << 1] == rightChild) {
                posEnd--;
            }
            if (posStart < posEnd && posAndSampleIndex[posStart << 1] > posAndSampleIndex[posEnd << 1]) {
                swap(posStart, posEnd);
                posStart++;
                posEnd--;
                leftCnt++;
            }
        }
        if (addInterval) {
            // interval is front closing and back opening
            nodeSampleInterval.add(leftChid, new Tuple<>(posInterval.v1, posInterval.v1 + leftCnt));
            nodeSampleInterval.add(rightChild, new Tuple<>(posInterval.v1 + leftCnt, posInterval.v2));
        }
//        LOG.debug("SamplePositionData sortByNodeIndex:" + leftCnt);
        return leftCnt;
    }

    // swap sample and node index
    private void swap(int samIdx1, int samIdx2) {
        samIdx1 <<= 1;
        samIdx2 <<= 1;
        int tmp = posAndSampleIndex[samIdx1];
        posAndSampleIndex[samIdx1] = posAndSampleIndex[samIdx2];
        posAndSampleIndex[samIdx2] = tmp;

        tmp = posAndSampleIndex[samIdx1 + 1];
        posAndSampleIndex[samIdx1 + 1] = posAndSampleIndex[samIdx2 + 1];
        posAndSampleIndex[samIdx2 + 1] = tmp;
    }

}
