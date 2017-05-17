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

package com.fenbi.ytklearn.utils;

import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.mp4j.utils.KryoUtils;
import com.fenbi.ytklearn.data.Constants;
import com.fenbi.ytklearn.data.Tuple;
import com.fenbi.ytklearn.exception.YtkLearnException;

import java.io.*;
import java.util.*;

/**
 * @author wufan
 */

public class PreciseQuantile {
    private static final String key = "0";

    private ThreadCommSlave comm;
    private final int rank;
    private final int threadId;
    private final boolean isSmallData;
    private final double quantile;

    public PreciseQuantile(ThreadCommSlave comm, boolean isSmallData) {
        this.comm = comm;
        if (comm != null) {
            this.rank = comm.getRank();
            this.threadId = Integer.parseInt(Thread.currentThread().getName());
        } else {
            this.rank = 0;
            this.threadId = 0;
        }
        this.isSmallData = isSmallData;
        this.quantile = 0.5;
    }

    public PreciseQuantile(ThreadCommSlave comm, boolean isSmallData, float quantile) {
        this.comm = comm;
        if (comm != null) {
            this.rank = comm.getRank();
            this.threadId = Integer.parseInt(Thread.currentThread().getName());
        } else {
            this.rank = 0;
            this.threadId = 0;
        }
        this.isSmallData = isSmallData;
        this.quantile = quantile;
    }

    private void info(String info) throws Mp4jException {
        if (comm != null && rank == 0 && threadId == 0) {
//        if (comm != null && rank == 0) {
            comm.info(String.format("[GBDT] rank: %d, threadId: %d, ",
                    rank, threadId) + info);
        }
    }

    public Float getWeightedQuantile(List<Integer> samplesList,
                                     float[] data,
                                     float[] weights) throws Mp4jException {
        if (isSmallData) {
            return getWeightedQuantileLocal(samplesList, data, weights);
        } else {
            return getWeightedQuantileDist(samplesList, data, weights);
        }
    }

    public Float getWeightedQuantile(Tuple<Integer, Integer> posInterval,
                                     int[] posAndSampleIndex,
                                     float[] data,
                                     float[] weights) throws Mp4jException {
        if (isSmallData) {
            return getWeightedQuantileLocal(posInterval, posAndSampleIndex, data, weights);
        } else {
            return getWeightedQuantileDist(posInterval, posAndSampleIndex, data, weights);
        }
    }

    // small dataset, can hold all data in a node
    private Float getWeightedQuantileLocal(List<Integer> samplesList,
                                           float[] data,
                                           float[] weights) throws Mp4jException {
//        CheckUtils.check(comm != null, "reduce is null when compute weighted median!");
        // data is not too much, collect all data and compute directly
        List<Float> dataWei = new ArrayList<>(samplesList.size());
        for (int i = 0; i < samplesList.size(); i++) {
            int sampleIdx = samplesList.get(i);
            dataWei.add(data[sampleIdx]);
            dataWei.add(weights[sampleIdx]);
        }

        //for debug
//        StringBuffer sb = new StringBuffer("\nbefore sample and weight data:" + dataAndWeights.size() + "\n details:");
//        for (int i = 0; i < 30 && i < dataAndWeights.size() ; i++){
//            sb.append(dataAndWeights.get(i));
//            sb.append(",");
//        }
        //end debug

        //1. collect all data and weights
        Map<String, List<Float>> mapList = new HashMap<>(1);
        mapList.put(key, dataWei);
        if (comm != null) {
            mapList = comm.allreduceMapListConcat(mapList, KryoUtils.getDefaultSerializer(Float.class), Float.class); // only one thread change this list, no need to copy out
        }
        dataWei = mapList.get(key);
        int dataWeiSize = dataWei.size();

        if (dataWeiSize == 0) {
            return Constants.QUANTILE_MISSGING;
        }
        CheckUtils.check(dataWeiSize % 2 == 0, "[GBDT] weighted median error, after merge data and weights size(%d) invalid!", dataWeiSize);

        if (rank == 0 && threadId == 0) {
            TupleQuickSort.quickSortElement(dataWei);
            //linear search in a sorted array, avoid allocating a double array
            double accWeiSum = 0.0;
            for (int i = 1; i < dataWeiSize; i += 2) {
                accWeiSum += dataWei.get(i);
            }

            double target = accWeiSum * quantile;
            accWeiSum = 0;
            for (int i = 0; i < dataWeiSize / 2; i++) {
                int j = i << 1;
                accWeiSum += dataWei.get(j + 1);
                if (accWeiSum >= target) {
                    return dataWei.get(j);
                }
            }
            return dataWei.get(dataWeiSize - 2);
        }
        return Constants.QUANTILE_MISSGING;
    }

    // small dataset, can hold all data in a node, called by TreeRefiner
    private Float getWeightedQuantileLocal(Tuple<Integer, Integer> posInterval,
                                           int[] posAndSampleIndex,
                                           float[] data,
                                           float[] weights) throws Mp4jException {
//        CheckUtils.check(comm != null, "reduce is null when compute weighted median!");
        // data is not too much, collect all data and compute directly
        List<Float> dataWei = new ArrayList<>(posInterval.v2 - posInterval.v1);
        for (int i = posInterval.v1; i < posInterval.v2; i++) {
            int sampleIdx = posAndSampleIndex[(i << 1) + 1];
            dataWei.add(data[sampleIdx]);
            dataWei.add(weights[sampleIdx]);
        }

        //1. collect all data and weights
        Map<String, List<Float>> mapList = new HashMap<>(1);
        mapList.put(key, dataWei);
        if (comm != null) {
            mapList = comm.allreduceMapListConcat(mapList, KryoUtils.getDefaultSerializer(Float.class), Float.class); // only one thread change this list, no need to copy out
        }

        dataWei = mapList.get(key);
        int dataWeiSize = dataWei.size();
        if (dataWeiSize == 0) {
            return Constants.QUANTILE_MISSGING;
        }
        CheckUtils.check(dataWeiSize % 2 == 0, "[GBDT] weighted median error, after merge data and weights size(%d) invalid!", dataWeiSize);

        if (rank == 0 && threadId == 0) {
            TupleQuickSort.quickSortElement(dataWei);
            //linear search in a sorted array, avoid allocating a double array
            double accWeiSum = 0.0;
            for (int i = 1; i < dataWeiSize; i += 2) {
                accWeiSum += dataWei.get(i);
            }

            double target = accWeiSum * quantile;
            accWeiSum = 0;
            for (int i = 0; i < dataWeiSize / 2; i++) {
                int j = i << 1;
                accWeiSum += dataWei.get(j + 1);
                if (accWeiSum >= target) {
                    return dataWei.get(j);
                }
            }
            return dataWei.get(dataWeiSize - 2);
//        List<Float> newDataWeights = mapList.get(key);
//        CheckUtils.check(newDataWeights.size() > 0, "after merge, data and weights size is 0");
//
//        if (rank == 0 && threadId == 0) {
//            double[] arr = new double[newDataWeights.size()];
//            for (int i = 0; i < newDataWeights.size(); i++) {
//                arr[i] = newDataWeights.get(i);
//            }
//
//            // 2. sort data and weight by key(data)
//            TupleQuickSort.quickSortElement(arr);
//            //calc weight sum
//            for (int i = 1; i < arr.length / 2; i++) {
//                int j = i << 1;
//                arr[j + 1] += arr[j - 1];
//            }
//
//            // 3. search mid_wei_sum
//            double target = arr[arr.length - 1] * quantile;
//            int targetIndex = BinarySearch.findFirstEqualOrUpperTuple(arr, target);
//            int dataIndex = targetIndex * 2;
//            return (float) arr[dataIndex];
//        }
        }
        return Constants.QUANTILE_MISSGING;
    }

    // large dataset
    private Float getWeightedQuantileDist(List<Integer> sampleList,
                                          float[] data,
                                          float[] weights) throws Mp4jException {
        // CheckUtils.check(comm != null, "reduce is null when compute weighted median!");
        // get total sample cnt
        long sampleCnt = sampleList.size();
        if (comm != null) {
            sampleCnt = comm.allreduce(sampleCnt, Operands.LONG_OPERAND(), Operators.Long.SUM);
        }
        // total sample cnt is small
        if (sampleCnt <= Constants.QUANTILE_SAMPLE_CNT_THRED ||
                (comm == null) || (comm.getSlaveNum() * comm.getThreadNum() == 1)) {
            return getWeightedQuantileLocal(sampleList, data, weights);
        }

        // 1. generate buckets
        float[] buckets = generateBuckets(sampleList, data, sampleCnt);

        // 2.statistic acc sum weights in each bucket
        double[] weightAccSumInBucket = getAccWeightSumInBucket(sampleList, data, weights, buckets);

        // 3. find bucket that contains target median
        double midWei = weightAccSumInBucket[buckets.length - 1] * quantile;
//        System.out.println("middle weight:" + midWei);

        int targetBuckectIndex = BinarySearch.findFirstEqualOrUpper(weightAccSumInBucket, midWei);
        CheckUtils.check(targetBuckectIndex >= 0, "[GBDT] findFirstEqualOrUpper error, cant't find element that is no smaller than target(%f)", midWei);

        if (targetBuckectIndex > 0 &&
                buckets[targetBuckectIndex] == buckets[targetBuckectIndex - 1]) {
            if (rank == 0 && threadId == 0) {
//                System.out.println("find here");
                return buckets[targetBuckectIndex];
            } else {
                return Constants.QUANTILE_MISSGING;
            }
        }


//        if (Thread.currentThread().getName().equals(Constants.FIRST_THREAD_NAME)) {
//            LOG.debug("buckets split:" + StringUtils.join(buckets, ","));
//            LOG.debug("weight sum in bucket size:" + weightSumInBucket.length + "detail:" + StringUtils.joinDouble(weightSumInBucket, ","));
//            LOG.debug("weight acc sum:" + weightAccSumInBucket.length +  "detail:" + StringUtils.joinDouble(weightAccSumInBucket, ","));
//            LOG.debug("mid wei:" + midWei + ", index:" + targetBuckectIndex);
//            LOG.debug(String.format("target weight sum: %f, target weight: %f, target index: %d, left wei sum: %f, right weight sum: %f",
//                    weightAccSumInBucket[bucketNum - 1], midWei, targetBuckectIndex, leftWeiSum, rightWeiSum));
//        }

        // 4. sync data vals and weights with other nodes
        List<Float> dataWei = getDataInBuckets(sampleList, data, weights, buckets, targetBuckectIndex);
//        info(Thread.currentThread().getName() + ", data weight size:" + dataWei.size());

        Map<String, List<Float>> mapList = new HashMap<>(1);
        mapList.put(key, dataWei);
        if (comm != null) {
            mapList = comm.allreduceMapListConcat(mapList, KryoUtils.getDefaultSerializer(Float.class), Float.class); // only one thread change this list, no need to copy out
        }
        dataWei = mapList.get(key);

        int dataWeiSize = dataWei.size();
        CheckUtils.check(dataWeiSize >= 2 && dataWeiSize % 2 == 0, "[GBDT] weighted median error, target bucket is empty!");
        if (rank == 0 && threadId == 0) {
            TupleQuickSort.quickSortElement(dataWei);

            double leftWeiSum = targetBuckectIndex >= 1 ? weightAccSumInBucket[targetBuckectIndex - 1] : 0;
//        double rightWeiSum = weightAccSumInBucket[bucketNum - 1] - weightAccSumInBucket[targetBuckectIndex];
            // find target value by linear search in a sorted array
            double target = midWei - leftWeiSum;

            // linear search in a sorted array
            double accWeiSum = 0.0;
            for (int i = 0; i < dataWeiSize / 2; i++) {
                int j = i << 1;
                accWeiSum += dataWei.get(j + 1);
                if (accWeiSum >= target) {
                    return dataWei.get(j);
                }
            }
            return dataWei.get(dataWeiSize - 2);
        }
        return Constants.QUANTILE_MISSGING;
    }

    // large dataset
    private Float getWeightedQuantileDist(Tuple<Integer, Integer> posInterval,
                                          int[] posAndSampleIndex,
                                          float[] data,
                                          float[] weights) throws Mp4jException {
        // CheckUtils.check(comm != null, "reduce is null when compute weighted median!");
        // get total sample cnt

        long sampleCnt = posInterval.v2 - posInterval.v1;
        if (comm != null) {
            sampleCnt = comm.allreduce(sampleCnt, Operands.LONG_OPERAND(), Operators.Long.SUM);
        }
        // total sample cnt is small
        if (sampleCnt <= Constants.QUANTILE_SAMPLE_CNT_THRED ||
                (comm == null) || (comm.getSlaveNum() * comm.getThreadNum() == 1)) {
            return getWeightedQuantileLocal(posInterval, posAndSampleIndex, data, weights);
        }

        // 1. generate buckets
        float[] buckets = generateBuckets(posInterval, posAndSampleIndex, data, sampleCnt);

        // 2.statistic acc sum weights in each bucket
        double[] weightAccSumInBucket = getAccWeightSumInBucket(posInterval, posAndSampleIndex, data, weights, buckets);

        // 3. find bucket that contains target median
        double midWei = weightAccSumInBucket[buckets.length - 1] * quantile;
//        System.out.println("middle weight:" + midWei);

        int targetBuckectIndex = BinarySearch.findFirstEqualOrUpper(weightAccSumInBucket, midWei);
        CheckUtils.check(targetBuckectIndex >= 0, "[GBDT] findFirstEqualOrUpper error, cant't find element that is no smaller than target(%f)", midWei);

        if (targetBuckectIndex > 0 &&
                buckets[targetBuckectIndex] == buckets[targetBuckectIndex - 1]) {
            if (rank == 0 && threadId == 0) {
//                System.out.println("find here");
                return buckets[targetBuckectIndex];
            } else {
                return Constants.QUANTILE_MISSGING;
            }
        }
//        if (Thread.currentThread().getName().equals(Constants.FIRST_THREAD_NAME)) {
//            LOG.debug("buckets split:" + StringUtils.join(buckets, ","));
//            LOG.debug("weight sum in bucket size:" + weightSumInBucket.length + "detail:" + StringUtils.joinDouble(weightSumInBucket, ","));
//            LOG.debug("weight acc sum:" + weightAccSumInBucket.length +  "detail:" + StringUtils.joinDouble(weightAccSumInBucket, ","));
//            LOG.debug("mid wei:" + midWei + ", index:" + targetBuckectIndex);
//            LOG.debug(String.format("target weight sum: %f, target weight: %f, target index: %d, left wei sum: %f, right weight sum: %f",
//                    weightAccSumInBucket[bucketNum - 1], midWei, targetBuckectIndex, leftWeiSum, rightWeiSum));
//        }

        // 4. sync data vals and weights with other nodes
        List<Float> dataWei = getDataInBuckets(posInterval, posAndSampleIndex, data, weights, buckets, targetBuckectIndex);
//        info(Thread.currentThread().getName() + ", data weight size:" + dataWei.size());

        Map<String, List<Float>> mapList = new HashMap<>(1);
        mapList.put(key, dataWei);
        if (comm != null) {
            mapList = comm.allreduceMapListConcat(mapList, KryoUtils.getDefaultSerializer(Float.class), Float.class); // only one thread change this list, no need to copy out
        }
        dataWei = mapList.get(key);
        int dataWeiSize = dataWei.size();
        CheckUtils.check(dataWeiSize >= 2 && dataWeiSize % 2 == 0, "[GBDT] weighted median error, target bucket is empty!");

        if (rank == 0 && threadId == 0) {
            // 5. calc final median
            // caution: all threads in this worker use the same dataWei, but we sort this list only in one thread
            TupleQuickSort.quickSortElement(dataWei);
            double leftWeiSum = targetBuckectIndex >= 1 ? weightAccSumInBucket[targetBuckectIndex - 1] : 0;
//        double rightWeiSum = weightAccSumInBucket[bucketNum - 1] - weightAccSumInBucket[targetBuckectIndex];
            // find target value in a linear search
            double target = midWei - leftWeiSum;

            // linear search in a sorted array
            double accWeiSum = 0.0;
            for (int i = 0; i < dataWeiSize / 2; i++) {
                int j = i << 1;
                accWeiSum += dataWei.get(j + 1);
                if (accWeiSum >= target) {
                    return dataWei.get(j);
                }
            }
            return dataWei.get(dataWeiSize - 2);
        }
        return Constants.QUANTILE_MISSGING;
    }


    private List<Float> getDataInBuckets(List<Integer> sampleList,
                                         float[] data,
                                         float[] weights,
                                         float[] buckets,
                                         int targetBuckectIndex) {
        List<Float> dataWei = new ArrayList<>(Constants.RESERVE_NUM);
        float lowerBound = targetBuckectIndex >= 1 ? buckets[targetBuckectIndex - 1] : Float.NEGATIVE_INFINITY;
//        float lowerBound = targetBuckectIndex >= 1 ? buckets[targetBuckectIndex - 1] : -Float.MAX_VALUE;
        float upperBound = buckets[targetBuckectIndex];
        boolean isLeftClosed = false, isRightClosed = true;
        if (lowerBound == upperBound) {
            isLeftClosed = true;
            isRightClosed = true;
        } else if (targetBuckectIndex + 1 < buckets.length) {
            if (buckets[targetBuckectIndex] == buckets[targetBuckectIndex + 1]) {
                isRightClosed = false;
            }
        }

        // find elem between lowerBound and upperBound
        if (isLeftClosed && isRightClosed) {
            for (int sampIdx : sampleList) {
                if (data[sampIdx] == lowerBound) {
                    dataWei.add(data[sampIdx]);
                    dataWei.add(weights[sampIdx]);
                }
            }

        } else if (!isLeftClosed && isRightClosed) {
            for (int sampIdx : sampleList) {
                if (data[sampIdx] > lowerBound && data[sampIdx] <= upperBound) {
                    dataWei.add(data[sampIdx]);
                    dataWei.add(weights[sampIdx]);
                }
            }
        } else if (!isLeftClosed && !isRightClosed) {
            for (int sampIdx : sampleList) {
                if (data[sampIdx] > lowerBound && data[sampIdx] < upperBound) {
                    dataWei.add(data[sampIdx]);
                    dataWei.add(weights[sampIdx]);
                }
            }
        } else {
            throw new YtkLearnException("[GBDT] weighted median error, isLeftClose = true && isRightClose = false is invalid");
        }
        return dataWei;
    }


    private List<Float> getDataInBuckets(Tuple<Integer, Integer> posInterval,
                                         int[] posAndSampleIndex,
                                         float[] data,
                                         float[] weights,
                                         float[] buckets,
                                         int targetBuckectIndex) {
        List<Float> dataWei = new ArrayList<>(Constants.RESERVE_NUM);
        float lowerBound = targetBuckectIndex >= 1 ? buckets[targetBuckectIndex - 1] : Float.NEGATIVE_INFINITY;
        float upperBound = buckets[targetBuckectIndex];
        boolean isLeftClosed = false, isRightClosed = true;
        if (lowerBound == upperBound) {
            isLeftClosed = true;
            isRightClosed = true;
        } else if (targetBuckectIndex + 1 < buckets.length) {
            if (buckets[targetBuckectIndex] == buckets[targetBuckectIndex + 1]) {
                isRightClosed = false;
            }
        }


        // find elem between lowerBound and upperBound
        if (isLeftClosed && isRightClosed) {
            for (int i = posInterval.v1; i < posInterval.v2; i++) {
                int sampIdx = posAndSampleIndex[(i << 1) + 1];
                if (data[sampIdx] == lowerBound) {
                    dataWei.add(data[sampIdx]);
                    dataWei.add(weights[sampIdx]);
                }
            }

        } else if (!isLeftClosed && isRightClosed) {
            for (int i = posInterval.v1; i < posInterval.v2; i++) {
                int sampIdx = posAndSampleIndex[(i << 1) + 1];
                if (data[sampIdx] > lowerBound && data[sampIdx] <= upperBound) {
                    dataWei.add(data[sampIdx]);
                    dataWei.add(weights[sampIdx]);
                }
            }
        } else if (!isLeftClosed && !isRightClosed) {
            for (int i = posInterval.v1; i < posInterval.v2; i++) {
                int sampIdx = posAndSampleIndex[(i << 1) + 1];
                if (data[sampIdx] > lowerBound && data[sampIdx] < upperBound) {
                    dataWei.add(data[sampIdx]);
                    dataWei.add(weights[sampIdx]);
                }
            }
        } else {
            throw new YtkLearnException("[GBDT] weighted median error, isLeftClose = true && isRightClose = false is invalid");
        }
        return dataWei;
    }


    private float[] generateBuckets(List<Integer> sampleList,
                                    float[] data,
                                    long totalSampleCnt) throws Mp4jException {
        // approximate bucketNum = sqrt(totalSampleCnt);
        // sampleRate = bucketNum /totalSampleCnt = 1 / sqrt(totalSampleCnt)
        // sample out cnt =  sampleInCurrentThread * sampleRate
        int sampleOutCnt = (int) (sampleList.size() / Math.sqrt((double) totalSampleCnt));
        sampleOutCnt = Math.max(sampleOutCnt, Constants.QUANTILE_MIN_SAMPLE_CNT_PER_WORKER);
        sampleOutCnt = Math.min(sampleOutCnt, sampleList.size());
        // sampled out data
        List<Float> sampleData;
        // 1. get data distribution and find bucket
        int[] sampleOut = new int[sampleOutCnt];
        RandomUtils.reservoidSample(sampleList.size(), sampleOutCnt, sampleOut);
        sampleData = new ArrayList<>(sampleOut.length);
        for (int i = 0; i < sampleOut.length; i++) {
            int sampleIdx = sampleList.get(sampleOut[i]);
            sampleData.add(data[sampleIdx]);
        }

        // list collection
        Map<String, List<Float>> mapList = new HashMap<>(1);
        mapList.put(key, sampleData);
        if (comm != null) {
            mapList = comm.allreduceMapListConcat(mapList, KryoUtils.getDefaultSerializer(Float.class), Float.class); // copyed out
        }
        sampleData = mapList.get(key);

        // copy sample Data
        float[] sampleArr = new float[sampleData.size()];
        for (int i = 0; i < sampleData.size(); i++) {
            sampleArr[i] = sampleData.get(i);
        }
        Arrays.sort(sampleArr);

        //for debug
//        StringBuffer sb = new StringBuffer("\nsample out data:" + sampleData.size() + "\n details:");
//        for (int i = 0; i < 30 && i <sampleData.size() ; i++){
//            sb.append(sampleArr[i]);
//            sb.append(",");
//        }
//        System.out.println(sb.toString());
//        info(sb.toString());
        //end debug

        // generate buckets, move element in place
        CheckUtils.check(sampleArr.length >= 1, "[GBDT] compute quantile error! sample out arr length is 0");
        float lastValue = sampleArr[0];
        int lastCnt = 1;
        int bucketIndex = 1;
        for (int i = 1; i < sampleArr.length; i++) {
            if (sampleArr[i] != lastValue) {
                sampleArr[bucketIndex] = sampleArr[i];
                bucketIndex++;
                lastValue = sampleArr[i];
                lastCnt = 1;

            } else if (lastCnt == 1) {
                sampleArr[bucketIndex] = sampleArr[i];
                bucketIndex++;
                lastCnt++;

            } else {
                lastCnt++;
            }
        }

        // put element from sampleArr to buckets, put Float.MAX_VALUE in the last place
        int bucketNum = bucketIndex + 1;
        float[] buckets = new float[bucketNum];
        for (int i = 0; i < bucketIndex; i++) {
            buckets[i] = sampleArr[i];
        }
        buckets[bucketIndex] = Float.MAX_VALUE;

        //for debug
//        StringBuffer sb2 = new StringBuffer("\nbucket size:" + bucketNum + "\n details:");
//        for (int i = 0; i < bucketNum ; i++){
//            sb2.append(buckets[i]);
//            sb2.append(",");
//        }
//        System.out.println(sb2.toString());
//        info(sb2.toString());
        //end debug
        return buckets;
    }

    private float[] generateBuckets(Tuple<Integer, Integer> posInterval,
                                    int[] posAndSampleIndex,
                                    float[] data,
                                    long totalSampleCnt) throws Mp4jException {
        // approximate bucketNum = sqrt(sampleCnt);
        // sampleRate = bucketNum /sampleCnt = 1 / sqrt(sampleCnt)
        // sample out cnt =  sampleInCurrentThread * sampleRate
        int localSampleCnt = posInterval.v2 - posInterval.v1;
        int sampleOutCnt = (int) (localSampleCnt / Math.sqrt((double) totalSampleCnt));
        sampleOutCnt = Math.max(sampleOutCnt, Constants.QUANTILE_MIN_SAMPLE_CNT_PER_WORKER);
        sampleOutCnt = Math.min(sampleOutCnt, localSampleCnt);
        // sampled out data
        List<Float> sampleData;
        // 1. get data distribution and find bucket
        int[] sampleOut = new int[sampleOutCnt];
        RandomUtils.reservoidSample(localSampleCnt, sampleOutCnt, sampleOut);
        sampleData = new ArrayList<>(sampleOut.length);
        for (int i = 0; i < sampleOut.length; i++) {
            int sampleIdx = posAndSampleIndex[(posInterval.v1 + sampleOut[i]) * 2 + 1];
            sampleData.add(data[sampleIdx]);
        }

        // list collection
        Map<String, List<Float>> mapList = new HashMap<>(1);
        mapList.put(key, sampleData);
        if (comm != null) {
            mapList = comm.allreduceMapListConcat(mapList, KryoUtils.getDefaultSerializer(Float.class), Float.class); // copyed out
        }
        sampleData = mapList.get(key);

        // copy sample Data
        float[] sampleArr = new float[sampleData.size()];
        for (int i = 0; i < sampleData.size(); i++) {
            sampleArr[i] = sampleData.get(i);
        }
        Arrays.sort(sampleArr);

        //for debug
//        StringBuffer sb = new StringBuffer("\nsample out data:" + sampleData.size() + "\n details:");
//        for (int i = 0; i < 30 && i <sampleData.size() ; i++){
//            sb.append(sampleArr[i]);
//            sb.append(",");
//        }
//        System.out.println(sb.toString());
//        info(sb.toString());
        //end debug

        // generate buckets, move element in place
        CheckUtils.check(sampleArr.length >= 1, "[GBDT] compute quantile error! sample out arr length is 0");
        float lastValue = sampleArr[0];
        int lastCnt = 1;
        int bucketIndex = 1;
        for (int i = 1; i < sampleArr.length; i++) {
            if (sampleArr[i] != lastValue) {
                sampleArr[bucketIndex] = sampleArr[i];
                bucketIndex++;
                lastValue = sampleArr[i];
                lastCnt = 1;

            } else if (lastCnt == 1) {
                sampleArr[bucketIndex] = sampleArr[i];
                bucketIndex++;
                lastCnt++;

            } else {
                lastCnt++;
            }
        }

        // put element from sampleArr to buckets, put Float.MAX_VALUE in the last place
        int bucketNum = bucketIndex + 1;
        float[] buckets = new float[bucketNum];
        for (int i = 0; i < bucketIndex; i++) {
            buckets[i] = sampleArr[i];
        }
        buckets[bucketIndex] = Float.MAX_VALUE;

        //for debug
//        StringBuffer sb2 = new StringBuffer("\nbucket size:" + bucketNum + "\n details:");
//        for (int i = 0; i < bucketNum ; i++){
//            sb2.append(buckets[i]);
//            sb2.append(",");
//        }
//        System.out.println(sb2.toString());
//        info(sb2.toString());
        //end debug
        return buckets;
    }


    private double[] getAccWeightSumInBucket(List<Integer> sampleList,
                                             float[] data,
                                             float[] weights,
                                             float[] buckets) throws Mp4jException {
        double[] weightSumInBucket = new double[buckets.length];
        for (int i = 0; i < weightSumInBucket.length; i++) {
            weightSumInBucket[i] = 0;
        }

        for (int sampIdx : sampleList) {
            int index = BinarySearch.findLastEqualOrUpper(buckets, data[sampIdx]);
            CheckUtils.check(index >= 0, "[GBDT] findLastEqualOrUpper error, cant't find element that is no smaller than target(%f)", data[sampIdx]);

//            CheckUtils.check(buckets[index] >= data[sampIdx], "[GBDT] quantile find last upper error! target=%f, find=%f", data[sampIdx], buckets[index]);
            weightSumInBucket[index] += weights[sampIdx];
        }
        if (comm != null) {
            weightSumInBucket = comm.allreduceArray(weightSumInBucket, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, weightSumInBucket.length);
        }

        for (int i = 1; i < weightSumInBucket.length; i++) {
            weightSumInBucket[i] += weightSumInBucket[i - 1];
        }
        return weightSumInBucket;
    }

    private double[] getAccWeightSumInBucket(Tuple<Integer, Integer> posInterval,
                                             int[] posAndSampleIndex,
                                             float[] data,
                                             float[] weights,
                                             float[] buckets) throws Mp4jException {
        double[] weightSumInBucket = new double[buckets.length];
        for (int i = 0; i < weightSumInBucket.length; i++) {
            weightSumInBucket[i] = 0;
        }

        for (int i = posInterval.v1; i < posInterval.v2; i++) {
            int sampleIdx = posAndSampleIndex[(i << 1) + 1];
            int index = BinarySearch.findLastEqualOrUpper(buckets, data[sampleIdx]);
            CheckUtils.check(index >= 0, "[GBDT] findLastEqualOrUpper error, cant't find element that is no smaller than target(%f)", data[sampleIdx]);
            weightSumInBucket[index] += weights[sampleIdx];
        }
        if (comm != null) {
            weightSumInBucket = comm.allreduceArray(weightSumInBucket, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, weightSumInBucket.length);
        }

        for (int i = 1; i < weightSumInBucket.length; i++) {
            weightSumInBucket[i] = weightSumInBucket[i - 1] + weightSumInBucket[i];
        }
        return weightSumInBucket;
    }
}
