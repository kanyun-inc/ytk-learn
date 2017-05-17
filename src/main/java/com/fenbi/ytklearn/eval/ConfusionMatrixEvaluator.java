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
import lombok.Data;

/**
 * @author xialong
 */

public class ConfusionMatrixEvaluator extends AbstractEvaluator {

    private final String name;
    private final float thred;
    public ConfusionMatrixEvaluator(String name, ThreadCommSlave comm, CoreData coreData) {
        super(comm, coreData);
        this.name = name;
        String []splits =  name.trim().split(NAME_DELIM);
        if (splits.length <= 1) {
            thred = 0.5f;
        } else {
            thred = Float.parseFloat(splits[1]);
        }
    }

    @Data
    public static class ConfusionMatrixInfo {
        private final int K;
        private final boolean isSoftmax;
        public ConfusionMatrixInfo(int K, boolean isSoftmax) {
            this.K = K;
            this.isSoftmax = isSoftmax;
        }
    }

    @Override
    public String getEvalName() {
        return name;
    }


    public String drawLine(int K, int width) {
        StringBuilder sb = new StringBuilder("");
        sb.append("+");
        for (int i = 0; i < K + 1; i++) {
            for (int j = 0; j < width - 1; j++) {
                sb.append("-");
            }
            sb.append("+");
        }
        return sb.toString();
    }

    @Override
    public String eval(Object info, String prefix, boolean weightAndReal) throws Exception {
        ConfusionMatrixInfo cmInfo = (ConfusionMatrixInfo) info;
        int K = cmInfo.getK();
        boolean isSoftmax = cmInfo.isSoftmax();
        double []matrix = new double[K * K];
        double []matrixPure = new double[K * K];
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = 0;
            matrixPure[i] = 0;
        }

        if (isSoftmax) {
            for (int k = 0; k < cursor2d; k++) {
                int lsNumInt = realNum[k];
                for (int i = 0; i < lsNumInt; i++) {
                    int target = -1000000;
                    int pred = 0;
                    float maxPred = -Float.MAX_VALUE;
                    int yidx = i * K;
                    for (int p = 0; p < K; p++) {
                        if (y[k][yidx + p] == 1.0f) {
                            target = p;
                        }
                        if (predict[k][yidx + p] > maxPred) {
                            maxPred = predict[k][yidx + p];
                            pred = p;
                        }
                    }

                    int idx = target * K + pred;
                    if (target < 0 && comm != null) {
                        comm.info("softmax must contain 1.0 label!");
                    }
                    matrix[idx] += weight[k][i];
                    matrixPure[idx] += 1;
                }
            }
        } else {
            for (int k = 0; k < cursor2d; k++) {
                int lsNumInt = realNum[k];
                for (int i = 0; i < lsNumInt; i++) {
                    int target = (int)y[k][i];
                    int pred = predict[k][i] >= thred ? 1 : 0;
                    int idx = target * K + pred;
                    matrix[idx] += weight[k][i];
                    matrixPure[idx] += 1;
                }
            }
        }

        if (comm != null) {
            comm.allreduceArray(matrix, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, matrix.length);
            comm.allreduceArray(matrixPure, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, matrixPure.length);
        }

        double rowAccu[] = new double[K];
        double columuAccu[] = new double[K];
        double acc = 0;
        double all = 0;

        double rowAccuPure[] = new double[K];
        double columuAccuPure[] = new double[K];
        double accPure = 0;
        double allPure = 0;
        for (int i = 0; i < K; i++) {
            rowAccu[i] = 0;
            columuAccu[i] = 0;
            acc += matrix[i * K + i];

            rowAccuPure[i] = 0;
            columuAccuPure[i] = 0;
            accPure += matrixPure[i * K + i];
            for (int j = 0; j < K; j++) {
                rowAccu[i] += matrix[i * K + j];
                all += matrix[i * K + j];
                columuAccu[i] += matrix[j * K + i];

                rowAccuPure[i] += matrixPure[i * K + j];
                allPure += matrixPure[i * K + j];
                columuAccuPure[i] += matrixPure[j * K + i];
            }
        }

        StringBuilder sb = new StringBuilder("");
        StringBuilder sbPure = new StringBuilder("");
        sb.append(drawLine(K, 16)).append("\n");
        sbPure.append(drawLine(K, 16)).append("\n");

        sb.append("|").append(String.format("%16s", "|"));
        sbPure.append("|").append(String.format("%16s", "|"));
        for (int i = 0; i < K; i++) {
            sb.append(String.format("%16s", "pred c" + i + " |"));
            sbPure.append(String.format("%16s", "pred c" + i + " |"));
        }
        sb.append("\n");
        sbPure.append("\n");

        sb.append(drawLine(K, 16)).append("\n");
        sbPure.append(drawLine(K, 16)).append("\n");

        for (int i = 0; i < K; i++) {
            sb.append("|").append(String.format("%16s", "actual c" + i + " |"));
            sbPure.append("|").append(String.format("%16s", "actual c" + i + " |"));
            for (int j = 0; j < K; j++) {
                sb.append(String.format("%14.1f |", matrix[i * K + j]));
                sbPure.append(String.format("%14.1f |", matrixPure[i * K + j]));
            }
            sb.append("\n");
            sbPure.append("\n");

            sb.append(drawLine(K, 16)).append("\n");
            sbPure.append(drawLine(K, 16)).append("\n");
        }

        for (int i = 0; i < K; i++) {
            sb.append(prefix + " class = ").append(i)
                    .append(", precision = ").append(matrix[i * K + i] * 1.0 / columuAccu[i])
                    .append(", recall = ").append(matrix[i * K + i] * 1.0 / rowAccu[i]).append("\n");
            sbPure.append(prefix + " class = ").append(i)
                    .append(", precision = ").append(matrixPure[i * K + i] * 1.0 / columuAccuPure[i])
                    .append(", recall = ").append(matrixPure[i * K + i] * 1.0 / rowAccuPure[i]).append("\n");
        }
        sb.append(prefix + " accuracy = " + (acc * 1.0 / all));
        sbPure.append(prefix + " accuracy = " + (accPure * 1.0 / allPure));

        if (weightAndReal) {
            return prefix + " " + getEvalName() + "(weighted) = \n" + sb.toString() +
                    "\n" + prefix + " " + getEvalName() + "(real) = \n" + sbPure.toString();
        } else {
            return prefix + " " + getEvalName() + " = \n" + sb.toString();
        }

    }
}
