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

import java.util.ArrayList;
import java.util.List;

/**
 * @author wufan
 */

public class EvalSet {

    public static final Logger LOG = LoggerFactory.getLogger(EvalSet.class);
    private List<IEvaluator> evals;
    private ThreadCommSlave comm;
    private CoreData coreData;

    // for distributed mode
    public EvalSet(ThreadCommSlave comm, CoreData coreData) {
        this.comm = comm;
        evals = new ArrayList<>();
        this.coreData = coreData;
    }

    public void addEvals(List<String> evalNames) throws Exception {
        for (String name: evalNames) {
            evals.add(EvaluatorFactory.createEvaluator(name.trim(), comm, coreData));
        }
    }

    public String eval(Object info, String prefix, boolean weightAndReal) throws Exception {
        StringBuilder sb = new StringBuilder();
        for (IEvaluator eval: evals) {
            String res = eval.eval(info, prefix, weightAndReal);
            sb.append(String.format("%s", res)).append("\n");
        }
        return sb.toString();
    }
}
