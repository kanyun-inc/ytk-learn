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

import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.ytklearn.data.Constants;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.ytklearn.dataflow.CoreData;
import com.fenbi.ytklearn.utils.CheckUtils;

/**
 * @author wufan
 */

public class AucEvaluator extends AbstractEvaluator {

    private final String name;
    private final int AUC_APPROXIMATE_SLOT_NUM;

    public AucEvaluator(String name, ThreadCommSlave comm, CoreData coreData) throws Mp4jException {
        super(comm, coreData);
        this.name = name;
        String[] fields = name.split(Constants.CONFIG_NAME_PARAM_SPLIT);
        CheckUtils.check(fields.length == 1 || fields.length == 2, "[GBDT] invalid auc(%s) conf", name);
        if (fields.length == 2) {
            AUC_APPROXIMATE_SLOT_NUM = Integer.parseInt(fields[1]);
        } else {
            AUC_APPROXIMATE_SLOT_NUM = Constants.AUC_APPROXIMATE_SLOT_NUM;
        }
    }

    @Override
    public String getEvalName() {
        return name;
    }

    @Override
    public String eval(Object info, String prefix, boolean weightAndReal) throws Exception {
        // init pos and neg cnt to 0
        double[] posNegCnt = new double[AUC_APPROXIMATE_SLOT_NUM << 1];
        double[] posNegCntPure = new double[AUC_APPROXIMATE_SLOT_NUM << 1];
        for (int i = 0; i < posNegCnt.length; i++) {
            posNegCnt[i] = 0;
            posNegCntPure[i] = 0;
        }

        int index = 0;
        for (int k = 0; k < cursor2d; k++) {
            int lsNumInt = realNum[k];
            for (int i = 0; i < lsNumInt; i++) {
                index = (int) (predict[k][i] * AUC_APPROXIMATE_SLOT_NUM);
                if (index < 0.0) {
                    index = 0;
                } else if (index >= AUC_APPROXIMATE_SLOT_NUM) {
                    index = AUC_APPROXIMATE_SLOT_NUM - 1;
                }
                if (y[k][i] == 1.0f) {
                    posNegCnt[index << 1] += weight[k][i];
                    posNegCntPure[index << 1] += 1.0;
                } else {
                    posNegCnt[(index << 1) + 1] += weight[k][i];
                    posNegCntPure[(index << 1) + 1] += 1.0;
                }
            }
        }

        if (comm != null) {
            comm.allreduceArray(posNegCnt, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, posNegCnt.length);
            comm.allreduceArray(posNegCntPure, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, posNegCntPure.length);
        }

        double posSum = 0;
        double negSum = 0;
        double posPairSum = 0;

        double posSumPure = 0;
        double negSumPure = 0;
        double posPairSumPure = 0;
        for (int i = AUC_APPROXIMATE_SLOT_NUM - 1; i >= 0; i--) {
            int j = (i << 1);
            posPairSum += posNegCnt[j + 1] * (posSum + posNegCnt[j] * 0.5);
            posSum += posNegCnt[j];
            negSum += posNegCnt[j + 1];

            posPairSumPure += posNegCntPure[j + 1] * (posSumPure + posNegCntPure[j] * 0.5);
            posSumPure += posNegCntPure[j];
            negSumPure += posNegCntPure[j + 1];
        }

        if (weightAndReal) {
            return prefix + " " + getEvalName() + "(weighted) = " + String.valueOf(posPairSum / (posSum * negSum)) +
                    "\n" + prefix + " " + getEvalName() + "(real) = " + String.valueOf(posPairSumPure / (posSumPure * negSumPure));
        } else {
            return prefix + " " + getEvalName() + " = " + String.valueOf(posPairSum / (posSum * negSum));
        }

    }
}
