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

package com.fenbi.ytklearn.eval;

import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.dataflow.CoreData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashSet;
import java.util.Set;

/**
 * @author wufan
 * @author xialong
 */

public class EvaluatorFactory {

    public static final Logger LOG = LoggerFactory.getLogger(EvaluatorFactory.class);

    public static final Set<String> EvalNameSet = new HashSet() {
        {
            add("auc");
            add("rmse");
            add("mae");
            add("confusion_matrix");
        }
    };

    public static IEvaluator createEvaluator(String name, ThreadCommSlave comm, CoreData coreData) throws Exception {

        if (name.startsWith("auc")) {
            return new AucEvaluator(name, comm, coreData);
        } else if (name.equals("rmse") || name.equals("mae")) {
            return new PointWiseEvaluator(name, comm, coreData);
        } else if (name.startsWith("confusion_matrix")) {
            return new ConfusionMatrixEvaluator(name, comm, coreData);
        } else {
            throw new Exception("unknown evaluation metric type:" + name);
        }

    }
}
