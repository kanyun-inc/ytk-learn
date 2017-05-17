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

package com.fenbi.ytklearn.dataflow;

import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.feature.FeatureHash;
import org.python.core.PyFunction;

import java.util.Map;

/**
 * @author xialong
 */

public class ContinuousCoreData extends CoreData {
    private FeatureHash featureHash;
    public ContinuousCoreData(ThreadCommSlave comm,
                              DataFlow.CoreParams coreParams,
                              IFeatureMap featureMap,
                              PyFunction pyTransformFunc,
                              boolean needPyTransform,
                              FeatureHash featureHash
                            ) {
        super(comm, coreParams, featureMap, pyTransformFunc, needPyTransform);
        this.featureHash = featureHash;
    }


    @Override
    protected Map<String, Float> line2FeatureMap(String line, String []info) {
        return featureHash.line2Map(info[2], coreParams.need_feature_hash);
    }
}
