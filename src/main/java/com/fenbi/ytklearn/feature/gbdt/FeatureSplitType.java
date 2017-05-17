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

package com.fenbi.ytklearn.feature.gbdt;

/**
 * feature split type, it decides how to choose the split feature value when makes a split
 * @author wufan
 * @author xialong
 */

public enum FeatureSplitType {

    MEAN {
        @Override
        public float getFeatureSplit(float[] feaSplitValSorted, int[] interval) {
            return 0.5f * (feaSplitValSorted[interval[0]] + feaSplitValSorted[interval[1]]);
        }

        @Override
        public String getName() {
            return "mean";
        }
    },

    MEDIAN {
        @Override
        public float getFeatureSplit(float[] feaSplitValSorted, int[] interval) {
            int sum = interval[0] + interval[1];
            if (sum % 2 == 0) {
                return feaSplitValSorted[sum / 2];
            } else {
                int left = (sum - 1) / 2;
                int right = (sum + 1) / 2;
                return  0.5f * (feaSplitValSorted[left] + feaSplitValSorted[right]);
            }
        }

        @Override
        public String getName() {
            return "median";
        }
    };

    public abstract float getFeatureSplit(float[] feaSplitValSorted, int[] interval);

    public abstract String getName();

    public static FeatureSplitType valueOfFeatureSplitType(String feaSplitType) {
        for (FeatureSplitType fsType: values()) {
            if (fsType.getName().equals(feaSplitType)) {
                return fsType;
            }
        }
        return null;
    }

}
