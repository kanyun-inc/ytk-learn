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
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.ytklearn.dataflow.CoreData;

/**
 * @author xialong
 */

public class PointWiseEvaluator extends AbstractEvaluator  {

    private final String name;
    private EvalPointWiseType evalType;
    public PointWiseEvaluator(String name, ThreadCommSlave comm, CoreData coreData) {
        super(comm, coreData);
        this.name = name;
        evalType = EvalPointWiseType.valueOf(name.toUpperCase());
    }

    @Override
    public String getEvalName() {
        return name;
    }

    @Override
    public String eval(Object info, String prefix, boolean weightAndReal) throws Exception {

        double sum = 0;
        double weightSum = 0;

        double sumPure = 0;
        double weightSumPure = 0;
        for (int k = 0; k < cursor2d; k++) {
            int lsNumInt = realNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                sum += evalType.evalRow(y[k][i], predict[k][i]) * weight[k][i];
                weightSum += weight[k][i];

                sumPure += evalType.evalRow(y[k][i], predict[k][i]);
                weightSumPure += 1;
            }
        }
        double[] data = new double[2];
        data[0] = sum;
        data[1] = weightSum;
        if (comm != null) {
            comm.allreduceArray(data, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, data.length);
        }

        double[] dataPure = new double[2];
        dataPure[0] = sumPure;
        dataPure[1] = weightSumPure;
        if (comm != null) {
            comm.allreduceArray(dataPure, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, dataPure.length);
        }

        if (weightAndReal) {
            return prefix + " " + getEvalName() + "(weighted) = " + String.valueOf(evalType.getFinal(data[0], data[1])) +
                    "\n" + prefix + " " + getEvalName() + "(real) = " + String.valueOf(evalType.getFinal(dataPure[0], dataPure[1]));
        } else {
            return prefix + " " + getEvalName() + " = " + String.valueOf(evalType.getFinal(data[0], data[1]));
        }

    }

}
