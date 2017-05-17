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

package com.fenbi.ytklearn.operation;

import java.util.HashSet;
import java.util.Set;

/**
 * @author xialong
 */

public class TrainOperationFactory {
    public static final Set<String> LBFGS_MODEL_NAME_SET = new HashSet<>();
    static {
        LBFGS_MODEL_NAME_SET.add("linear");
        LBFGS_MODEL_NAME_SET.add("fm");
        LBFGS_MODEL_NAME_SET.add("ffm");
        LBFGS_MODEL_NAME_SET.add("mlr");
        LBFGS_MODEL_NAME_SET.add("multiclass_linear");
    }
    public static ITrainOperation createTrainOperation(String modelName) throws Exception {
        if (LBFGS_MODEL_NAME_SET.contains(modelName)) {
            return new HoagOperation();
        } else if (modelName.equalsIgnoreCase("gbdt")) {
            return new GBDTOperation();
        } else if (modelName.equalsIgnoreCase("gbmlr") || modelName.equalsIgnoreCase("gbsdt")
                || modelName.equalsIgnoreCase("gbhmlr") || modelName.equalsIgnoreCase("gbhsdt")) {
            return new GBMLROperation();
        } else {
            throw new Exception("unknown model name!");
        }

    }
}
