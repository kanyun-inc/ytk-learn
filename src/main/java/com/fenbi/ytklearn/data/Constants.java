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

package com.fenbi.ytklearn.data;

/**
 * @author wufan
 */

public class Constants {

    public final static double MIN_LOSS_CHG = 1E-5f;

    public final static float MIN_FEA_SPLIT_GAP = 1E-16f;

    public final static int INT_MISSING_VALUE = Float.floatToRawIntBits(Float.NaN);

    // eg: eval name split, fusion_matrix@0.7, missing value process, quantile@0.25
    public final static String CONFIG_NAME_PARAM_SPLIT = "@";

    public final static int MAX_DEPTH = 32;

    // init allocate space for list or set or map
    public final static int RESERVE_NUM = 256;

    // slot_num for distributed auc approximate computation
    public final static int AUC_APPROXIMATE_SLOT_NUM = 100000;

    // used in precise quantile computing
    public final static int QUANTILE_SAMPLE_CNT_THRED = 100000;
    public final static int QUANTILE_MIN_SAMPLE_CNT_PER_WORKER = 10;
    public final static float QUANTILE_MISSGING = Float.POSITIVE_INFINITY;

    // used in approximate quantile computing
    public final static long QUNANTILE_PRECISION_MAX_SAMPLE_CNT = 100000;
    public final static double QUNANTILE_APPROXIMATE_EPS = 1E-5;
    public final static int QUNANTILE_APPROXIMATE_BIN_FACTOR = 8;

    public final static int MB = 1024 * 1024;

}
