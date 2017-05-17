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

package com.fenbi.ytklearn.optimizer;

import com.fenbi.ytklearn.dataflow.DataFlow;
import com.fenbi.ytklearn.dataflow.GBDTDataFlow;
import com.fenbi.ytklearn.dataflow.ContinuousDataFlow;

/**
 * @author xialong
 */

public class OptimizerFactory {
    public static IOptimizer createOptimizer(String name, DataFlow dataFlow, int threadIdx) throws Exception {
        if (name.equalsIgnoreCase("linear")) {
            return new LinearHoagOptimizer(name, (ContinuousDataFlow)dataFlow, threadIdx);
        } else if (name.equalsIgnoreCase("fm")) {
            return new FMHoagOptimizer(name, (ContinuousDataFlow)dataFlow, threadIdx);
        } else if (name.equalsIgnoreCase("ffm")) {
            return new FFMHoagOptimizer(name, (ContinuousDataFlow)dataFlow, threadIdx);
        } else if (name.equalsIgnoreCase("gbmlr")) {
            return new GBMLRHoagOptimizer(name, (ContinuousDataFlow)dataFlow, threadIdx);
        } else if (name.equalsIgnoreCase("gbsdt")) {
            return new GBSDTHoagOptimizer(name, (ContinuousDataFlow)dataFlow, threadIdx);
        } else if (name.equalsIgnoreCase("multiclass_linear")) {
            return new MulticlassLinearHoagOptimizer(name, (ContinuousDataFlow)dataFlow, threadIdx);
        } else if (name.equalsIgnoreCase("gbdt")) {
            return new GBDTOptimizer((GBDTDataFlow) dataFlow, threadIdx);
        } else if (name.equalsIgnoreCase("gbhmlr")) {
            return new GBHMLRHoagOptimizer(name, (ContinuousDataFlow)dataFlow, threadIdx);
        } else if (name.equalsIgnoreCase("gbhsdt")) {
            return new GBHSDTHoagOptimizer(name, (ContinuousDataFlow)dataFlow, threadIdx);
        } else {
            throw new Exception("unknown optimizer name");
        }
    }
}
