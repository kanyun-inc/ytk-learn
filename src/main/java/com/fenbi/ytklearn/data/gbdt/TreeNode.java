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

import java.io.Serializable;

/**
 * @author wufan
 */

public class TreeNode implements Serializable {
    private int parent;
    private int leftChild;
    private int rightChild;
    private boolean isDefaultLeft;

    // split feature index, used for train phase
    private int splitFeatureIndex;
    // split feature name
    private String splitFeatureName;

    // for leaf node it's prediction
    // for non-leaf node, it's feature split value
    private float nodeInfo;

    // used for train, split feature slot index, split feature point is between start and end
    private transient int[] splitFeaApproSlotInterval;

    public TreeNode() {
        splitFeatureIndex = 0;
        splitFeatureName = "";
        parent = -1;   //is root
        leftChild = -1;  //is leaf
        rightChild = -1;
        isDefaultLeft = true; // default go left path
        nodeInfo = 0.0f;
        splitFeaApproSlotInterval = new int[]{0, 0};
    }

    public int getLeftChild() {
        return leftChild;
    }

    public int getRightChild() {
        return rightChild;
    }

    public int getSplitFeatureIndex() {
        return splitFeatureIndex;
    }

    public String getSplitFeatureName() {
            return splitFeatureName;
    }

    // next node id when feature value is missing
    public int getDefualtChild() {
        return isDefaultLeft ? leftChild : rightChild;
    }

    // get split feature value
    public float getSplitCond() {
        return nodeInfo;
    }

    // get prediction result for current leaf node
    public float getLeafValue() {
        return nodeInfo;
    }

    // used in train phase
    public int[] getSplitFeaApproSlotInterval() {
        return splitFeaApproSlotInterval;
    }

    public void setParent(int pid) {
        parent = pid;
    }

    public int getParent() {
        return parent;
    }

    public boolean isLeaf() {
        return leftChild == -1;
    }

    public boolean isRoot() {
        return parent == -1;
    }

    public void setLeftChild(int nodeId) {
        leftChild = nodeId;
    }

    public void setRightChild(int nodeId) {
        rightChild = nodeId;
    }

    public void setLeaf(float value) {
            nodeInfo = value;
            leftChild = -1;
            rightChild = -1;
    }

    public void setSplitFeatureIndex(int featureIndex) {
        this.splitFeatureIndex = featureIndex;
    }

    public void setSplitFeatureName(String featureName) {
        this.splitFeatureName = featureName;
    }

    public void setSplit(int _splitFeatureIndex, float _splitFeaValue) {
        splitFeatureIndex = _splitFeatureIndex;
        nodeInfo = _splitFeaValue;
    }

   public void setDefaultDirection(boolean _isDefaultLeft) {
       isDefaultLeft = _isDefaultLeft;
   }

    public void setSplit(int _splitFeatureIndex, int feaApproStart, int feaApproEnd) {
        splitFeatureIndex = _splitFeatureIndex;
        splitFeaApproSlotInterval[0] = feaApproStart;
        splitFeaApproSlotInterval[1] = feaApproEnd;
        nodeInfo = 0.5f * (feaApproStart + feaApproEnd);
    }

    @Override
    public String toString() {
        return "parent:" + getParent()
                + ", left_child:" + leftChild
                + ", right_child:" + rightChild
                + ", split_feature_index:" + splitFeatureIndex
                + ", node_info:" + nodeInfo;
    }

}
