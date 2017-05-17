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

import java.util.Random;

/**
 * @author wufan
 * @author xialong
 */

public class PredictCoreData extends CoreData {

    private int SAMPLE_MAX_1D_LEN;
    private int max_1d_len;
    private int K;
    private int yindex;

    public PredictCoreData(ThreadCommSlave comm, int K) {
        super(comm);
        this.y = new float[MAX_2D_LEN][];
        this.weight = new float[MAX_2D_LEN][];
        this.realNum = new int[MAX_2D_LEN];
        this.predict = new float[MAX_2D_LEN][];

        this.K = K;
        this.SAMPLE_MAX_1D_LEN = MAX_1D_LEN / K;
        this.max_1d_len = SAMPLE_MAX_1D_LEN * K;

        this.cursor2d = -1;
        this.count = SAMPLE_MAX_1D_LEN;
        this.yindex = 0;
        this.gWeightNum = 0;
        this.gRealNum = 0;

    }


    protected boolean exceed1DRange() {
        return count >= SAMPLE_MAX_1D_LEN;
    }

    protected void exceed1DHandle() {
        cursor2d++;
        if (exceed2DRange()) {
            exceed2DHandle();
        }
        y[cursor2d] = new float[max_1d_len];
        predict[cursor2d] = new float[max_1d_len];
        weight[cursor2d] = new float[SAMPLE_MAX_1D_LEN];

        yindex = 0;
        count = 0;
//        System.out.println("cursor2d=" + cursor2d);

    }

    protected boolean exceed2DRange() {
        return cursor2d >= y.length;
    }

    protected void exceed2DHandle() {
        float[][] new_y = new float[y.length * 2][];
        float[][] new_weight = new float[y.length * 2][];
        int[] new_realNum = new int[y.length * 2];
        float[][] new_predict = new float[y.length * 2][];

        for (int i = 0; i < y.length; i++) {
            new_y[i] = y[i];
            new_weight[i] = weight[i];
            new_realNum[i] = realNum[i];
            new_predict[i] = predict[i];
        }

        y = new_y;
        weight = new_weight;
        realNum = new_realNum;
        predict = new_predict;
    }

    private void addData(double[] curPredict, double[] label, float curWeight) {
        for (int i = 0; i < K; i++) {
            predict[cursor2d][yindex] = (float) curPredict[i];
            y[cursor2d][yindex++] = (float) label[i];
        }
        weight[cursor2d][count++] = curWeight;
        realNum[cursor2d]++;

        gRealNum++;
        gWeightNum += curWeight;
    }

    private void addData(double curPredict, double label, float curWeight) {

        predict[cursor2d][yindex] = (float) curPredict;
        y[cursor2d][yindex++] = (float) label;

        weight[cursor2d][count++] = curWeight;
        realNum[cursor2d]++;

        gRealNum++;
        gWeightNum += curWeight;
    }

    public void addPredict(double predict, double label, float weight) {
        if (exceed1DRange()) {
            exceed1DHandle();
        }

        addData(predict, label, weight);
    }

    public void addPredict(double[] predict, double[] label, float weight) {
        if (exceed1DRange()) {
            exceed1DHandle();
        }

        addData(predict, label, weight);
    }

    public void endAddData() {
        cursor2d++;
    }

    public boolean needWeightAndReal() {
        return Math.abs(gWeightNum - gRealNum) > 1e-6;
    }


    public int getK() {
        return K;
    }

    public static void main(String[] args) {
        int K = 3;
        double[] predict = new double[K];
        double[] label = new double[K];
        PredictCoreData data = new PredictCoreData(null, K);
        Random r = new Random();
        for (int i = 0; i < 10000000; i++) {

            for (int j = 0; j < K; j++) {
                predict[j] = r.nextDouble();
                label[j] = 1.5;
            }
            data.addPredict(predict, label, 1.f);
        }
        data.endAddData();

        long num = 0;
        double labelSum = 0;
        for (int i = 0; i < data.cursor2d; i++) {
            System.out.println(i + ":" + data.realNum[i]);
            num += data.realNum[i];
            for (int j = 0; j < data.realNum[i]; j++) {
                for (int k = 0; k < K; k++) {
                    labelSum += data.y[i][j * K + k];
                }
//                System.out.println(data.predict[i][j]);
//                if (j >= 5) {
//                    break;
//                }
            }
//            if (i >= 5) {
//                break;
//            }

        }

        System.out.println("need=" + data.needWeightAndReal());
        System.out.println("labelSum=" + labelSum);
        System.out.println("num=" + num + ", gnum=" + data.gRealNum + ", wei=" + data.gWeightNum + ", cursor2d=" + data.cursor2d);

    }

}
