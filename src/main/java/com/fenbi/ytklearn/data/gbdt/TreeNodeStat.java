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
 * stats info of tree node
 * @author wufan
 */

public class TreeNodeStat implements Serializable {

    // loss chg in current split
    private float lossChg;
    // sum of hessian
    private float hessSum;
    private long nodeSampleCnt;

    public TreeNodeStat() {
        lossChg = 0.0f;
        hessSum = 0.0f;
        nodeSampleCnt = 0;
    }

    public float getLossChg() {
        return lossChg;
    }

    public void setLossChg(float lossChg) {
        this.lossChg = lossChg;
    }

    public float getHessSum() {
        return hessSum;
    }

    public void setHessSum(float hessSum) {
        this.hessSum = hessSum;
    }

    public long getNodeSampleCnt() {
        return nodeSampleCnt;
    }

    public void setNodeSampleCnt(long nodeSampleCnt) {
        this.nodeSampleCnt = nodeSampleCnt;
    }

    public String print(boolean isLeaf) {
        String info = "";
        if (!isLeaf) {
            info = String.format(",gain=%s,hess_sum=%s,sample_cnt=%d", Float.toString(lossChg), Float.toString(hessSum), nodeSampleCnt);
        } else {
            info = String.format(",hess_sum=%s,sample_cnt=%d", Float.toString(hessSum), nodeSampleCnt);
        }
        return info;
    }

}
