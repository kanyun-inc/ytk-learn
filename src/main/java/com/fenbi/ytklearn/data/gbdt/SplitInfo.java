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

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

/**
 * statistics used to represent a split for the tree
 * @author wufan
 */

public class SplitInfo {
    private float lossChg;
    private int splitIndex;
    // local version
    private float splitValue;
    // distributed version
    private int[] splitSlotInterval;

    public static class SplitInfoSerializer extends Serializer<SplitInfo> {

        @Override
        public void write(Kryo kryo, Output output, SplitInfo object) {
            output.writeFloat(object.lossChg);
            output.writeInt(object.splitIndex);
            output.writeFloat(object.splitValue);
            output.writeInt(object.splitSlotInterval.length);
            for (int i = 0; i < object.splitSlotInterval.length; i++) {
                output.writeInt(object.splitSlotInterval[i]);
            }
        }

        @Override
        public SplitInfo read(Kryo kryo, Input input, Class<SplitInfo> type) {
            SplitInfo splitInfo = new SplitInfo();
            splitInfo.lossChg = input.readFloat();
            splitInfo.splitIndex = input.readInt();
            splitInfo.splitValue = input.readFloat();
            int len = input.readInt();
            splitInfo.splitSlotInterval = new int[len];
            for (int i = 0; i < len; i++) {
                splitInfo.splitSlotInterval[i] = input.readInt();
            }
            return splitInfo;
        }
    }

    public SplitInfo() {
        lossChg = Float.NEGATIVE_INFINITY;
        splitIndex = -1;
        splitValue = 0.0f;
        splitSlotInterval = new int[] {-1, -1};
    }

    private SplitInfo(float lossChg, int splitIndex, float splitValue, int[] splitSlotInterval) {
        this.lossChg = lossChg;
        this.splitIndex = splitIndex;
        this.splitValue = splitValue;
        this.splitSlotInterval = new int[] {splitSlotInterval[0], splitSlotInterval[1]};
    }

    public SplitInfo deepClone() {
        return new SplitInfo(lossChg, splitIndex, splitValue, splitSlotInterval);
    }

    public void clear() {
        lossChg = Float.NEGATIVE_INFINITY;;
        splitIndex = -1;
        // used in local
        splitValue = 0.0f;
        // used in distributed
        splitSlotInterval[0] = splitSlotInterval[1] = -1;
    }

    public boolean needReplace(float newLossChg, int newSplitIndex) {
        if (splitIndex <= newSplitIndex) {
            return newLossChg > lossChg;
        }
        return newLossChg >= lossChg;
    }

    public boolean update(SplitInfo entry) {
        if (needReplace(entry.lossChg, entry.splitIndex)) {
            lossChg = entry.lossChg;
            splitIndex = entry.splitIndex;
            splitValue = entry.splitValue;

            splitSlotInterval = entry.splitSlotInterval;
            return true;
        } else {
            return false;
        }
    }

    public boolean update(float newLossChg, int newSplitIndex, float newSplitValue) {
        if (needReplace(newLossChg, newSplitIndex)) {
            lossChg = newLossChg;
            splitIndex = newSplitIndex;
            splitValue = newSplitValue;
            return true;
        } else {
            return false;
        }
    }

    public boolean update(float newLossChg, int newSplitIndex, int startFeaApprSlot, int endFeaApprSlot) {
        if (needReplace(newLossChg, newSplitIndex)) {
            lossChg = newLossChg;
            splitIndex = newSplitIndex;
            splitSlotInterval[0] = startFeaApprSlot;
            splitSlotInterval[1] = endFeaApprSlot;
            return true;
        } else {
            return false;
        }
    }


    public int getSplitIndex() {
        return splitIndex;
    }

    public float getLossChg() {
        return lossChg;
    }

    public float getSplitValue() {
        return splitValue;
    }

    public int[] getSplitSlotInterval() {
        return splitSlotInterval;
    }

    @Override
    public String toString() {
        return "loss change: " + lossChg
                + ", split feature index:" + splitIndex
                + ", split feature val:" + splitValue
                + ", split feature interval: [" + splitSlotInterval[0] + "," + splitSlotInterval[1] + "]";
    }


}
