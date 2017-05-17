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

package com.fenbi.ytklearn.predictor;

import java.util.Map;

/**
 * @author xialong
 */

public interface ITreePredictor {

    /**
     * If the tree model is GBDT, leaf indexes are returned, length = tree number
     * else if the tree model is GBST, gating values(soft leaf indexes) are returned, length = tree number * mixture number
     * @param features
     * @return leaf indexes or gating values, these values usually be used as features for other models.
     */
    double[] predictLeaf(Map<String, Float> features);

    default String leafFeatures(Map<String, Float> features, String featuresDelim, String featureNameValueDelim) {
        double []leaf = predictLeaf(features);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < leaf.length; i++) {
            sb.append("tree_leaf_").append(i).append(featureNameValueDelim).append(leaf[i]);
            if (i < leaf.length - 1) {
                sb.append(featuresDelim);
            }
        }
        return sb.toString();
    }

}
