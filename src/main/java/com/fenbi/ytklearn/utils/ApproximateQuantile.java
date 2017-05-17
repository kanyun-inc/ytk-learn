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

import java.io.Serializable;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

/**
 * @author xialong
 */

public class ApproximateQuantile implements Serializable {

    private double eps;
    private int b;

    private long n;

    private Summary summarys[];
    private int L = 0;
    private boolean exact;

    public ApproximateQuantile(long n, double eps, long exactThre) {
        if (eps > 1.0) {
            eps = 1.0;
        }
        this.b = (int)Math.ceil(log2(eps * n) / eps); // eps * n = 1, log2 = 0
        this.exact = false;
        if (eps * n < 2 || n <= exactThre) {
            b = (int)n;
            exact = true;
        }
        this.n = n;
        this.eps = eps;
        System.out.println("n:" + n + ", b:" + b + ", eps:" + eps +
                ", storage use:" + (log2(eps * n) * log2(eps * n) / eps) + ", exact:" + exact);
        summarys = new Summary[500];
        summarys[L++] = new Summary(b, b, eps, exact);
    }

    public ApproximateQuantile(long n, long errorBound, long exactThre) {
        this(n, errorBound * 1.0 / n, exactThre);
    }

    public ApproximateQuantile(long n, int bins, long exactThre) {
        this(n, 0.5 / bins, exactThre);
    }


    public int getL() {
        return L;
    }

    public int getb() {
        return b;
    }

    public long getn() {
        return n;
    }

    public double getEps() {
        return eps;
    }

    public void update(float value) throws Exception {
        // insert && if full
        if (summarys[0].insert(value)) {
            summarys[0].sort();
            Summary sc = summarys[0].compress(b);
            summarys[0] = new Summary(b, b, eps, exact);

            for (int l = 1; l < L + 1; l++) {
                if (summarys[l] == null) {
                    summarys[l] = sc;

                    if (l == L) {
                        L ++;
                    }
                    break;
                } else {
                    Summary sm = summarys[l].merge(sc);
                    sm.fix();
                    sc = sm.compress(b);
                    summarys[l] = null;
                }
            }
        }
    }

    public Summary mergeAll() throws Exception {
        System.out.println("L:" + L);
        summarys[0].sort();
        summarys[0].B = summarys[0].getRealNum();
        Summary sm = summarys[0];
        for (int l = 1; l < L; l++) {
            if (summarys[l] != null) {
                sm = sm.merge(summarys[l]);
                sm.fix();
            }
        }
        sm.cleanPoolMap();
        return sm;
    }

    public Summary mergeAllAndCompress() throws Exception {
        Summary sm = mergeAll();
        return sm.compress(b);
    }


    private static double log2(double val) {
        return Math.log(val) / Math.log(2.0);
    }

    public static int blockSize(double eps, int n) {
        return (int)Math.floor(log2(eps * n) / eps);
    }


    public static class PoolNode {
        public float []value;
        public long []rmin;
        public long []rmax;


        public PoolNode() {}

        public PoolNode(float []value, long[]rmin, long[]rmax) {
            this.value = value;
            this.rmin = rmin;
            this.rmax = rmax;
        }
    }



    public static class Summary implements Serializable{
        // value
        float []value;
        // rmin, rmax, ...
        long []rmin;
        long []rmax;
        int capacity;
        int cursor = 1;
        long B = 0;
        boolean exact;
        double eps;
        transient TreeMap<Double, PoolNode> poolMap;
        Random rand = new Random();

        public Summary(int capacity, long B, double eps, boolean exact) {
            this.poolMap = new TreeMap<>();
            PoolNode poolNode = this.getPoolNode(capacity);

            this.exact = exact;
            value = poolNode.value;
            rmin = poolNode.rmin;
            rmax = poolNode.rmax;
            value[0] = -Float.MAX_VALUE;
            rmin[0] = 0;
            rmax[0] = 0;
            this.capacity = capacity;
            this.B = B;
            this.eps = eps;
        }

        @Override
        public String toString() {
            return "capacity:" + capacity + ", cursor:" + cursor + ", B:" + B + ", exact:" + exact + "\n" +
                    ", value:" + Arrays.toString(value) + "\n" +
                    ", rmin:" + Arrays.toString(rmin) + "\n" +
                    ", rmax:" + Arrays.toString(rmax);
        }

        public boolean full() {
            return cursor > capacity;
        }

        public boolean insert(float value) {
            this.value[cursor++] = value;
            return cursor > capacity;
        }

        public boolean insert(float value, long rmin, long rmax) {
            this.value[cursor] = value;
            this.rmin[cursor] = rmin;
            this.rmax[cursor] = rmax;
            cursor++;
            return cursor > capacity;
        }

        public void sort() {
            Arrays.sort(value, 1, cursor);
            for (int i = 1; i < cursor; i++) {
                rmin[i] = i;
                rmax[i] = i;
            }
        }

        public void setCapacity(int capacity) {
            this.capacity = capacity;
        }

        public int getCapacity() {
            return capacity;
        }

        public void setB(int B) {
            this.B = B;
        }

        public long getB() {
            return B;
        }

        public void reset() {
            cursor = 1;
        }

        public int getRealNum() {
            return cursor - 1;
        }

        public PoolNode getPoolNode(int len) {
            if (poolMap == null) {
                poolMap = new TreeMap<>();
            }
            Map.Entry<Double, PoolNode> poolEntry = poolMap.ceilingEntry((double)len);
            System.out.println("before get pool size:" + poolMap.size() + ", get len:" + len + ", keys:" + poolMap.keySet());
            len ++;
            if (poolEntry == null) {
                PoolNode poolNode = new PoolNode();
                poolNode.value = new float[len];
                poolNode.rmin = new long[len];
                poolNode.rmax = new long[len];
                return poolNode;
            } else {
                poolMap.remove(poolEntry.getKey());
                System.out.println("after get pool size:" + poolMap.size() + ", get len:" + len + ", keys:" + poolMap.keySet());
                return poolEntry.getValue();
            }
        }

        public void cleanPoolMap() {
            poolMap.clear();
        }

        public void insertPoolNode(int len) {
            if (poolMap == null) {
                poolMap = new TreeMap<>();
            }
            System.out.println("before insert pool size:" + poolMap.size() + ", insert len:" + len);
            poolMap.put(len + rand.nextDouble(), new PoolNode(value, rmin, rmax));
            System.out.println("after insert pool size:" + poolMap.size() + ", keys:" + poolMap.keySet());
        }

        public void fix() {
            // fix rank1
            if (cursor <= 1)
                return;

            double epsB = eps * B;
            if (rmin[1] != 1) {
                rmin[1] = 1;
            }

            for (int i = 2; i < cursor; i++) {
                double prevmin = rmin[i - 1] + epsB;
                double curmax = rmax[i] - epsB;

                if (curmax > prevmin + 1) {
                    rmin[i - 1] = (long)Math.ceil(rmax[i] - 2 * epsB - 1);
                }
            }
        }

        public float query(long r) {
            long rnew = (long)Math.floor(r + eps * B);
            int idx = Arrays.binarySearch(rmax, 1, cursor, rnew);

            idx = idx >= 0 ? idx : (-idx) - 2;
            if (idx < 1) {
                idx = 1;
            } else if (idx >= cursor) {
                idx = cursor - 1;
            }

            return value[idx];
        }

        public int queryIdx(long r) {
            long rnew = (long)Math.floor(r + eps * B);
            int idx = Arrays.binarySearch(rmax, 1, cursor, rnew);

            idx = idx >= 0 ? idx : (-idx) - 2;
            if (idx < 1) {
                idx = 1;
            } else if (idx >= cursor) {
                idx = cursor - 1;
            }

            return idx;
        }


        public Summary compress(double b) throws Exception {
            if (exact) {
                return this;
            }

            if (cursor <= 1) {
                return this;
            }

            double avg = 2.0 * B / b;
            long r = 1;
            int handledCnt = 1;
            int pointer = 1;
            int newSize = ((int)Math.ceil(b / 2)) + 1;
            PoolNode poolNode = getPoolNode(newSize);
            poolNode.value[0] = -Float.MAX_VALUE;
            poolNode.rmin[0] = 0;
            poolNode.rmax[0] = 0;
            while(true) {

                if (r != B) {
                    if (handledCnt >= 2) {
                        r = (int)Math.floor(handledCnt * avg);
                    }
                }

                pointer = queryIdx(r);

                poolNode.value[handledCnt] = value[pointer];
                poolNode.rmin[handledCnt] = rmin[pointer];
                poolNode.rmax[handledCnt] = rmax[pointer];

                if (r == B) {
                    poolNode.value[handledCnt] = value[cursor - 1];
                    poolNode.rmin[handledCnt] = rmin[cursor - 1];
                    poolNode.rmax[handledCnt] = rmax[cursor - 1];
                }


                //pointer ++; // TODO: del?
                handledCnt ++;
                if (r >= B) {
                    if (r == B) {
                        break;
                    } else {
                        if (handledCnt < newSize) {
                            r = B;
                        } else {
                            break;
                        }
                    }
                }
            }

            insertPoolNode(value.length - 1);

            value = poolNode.value;
            rmin = poolNode.rmin;
            rmax = poolNode.rmax;
            capacity = handledCnt - 1;
            cursor = handledCnt;

            System.out.println("compressed size:" + (handledCnt - 1) + ", at most size:" + newSize);
            if (newSize < handledCnt - 1) {
               throw new Exception("new size < handled cnt");
            }

            return this;

        }

        public Summary merge(Summary that) throws Exception {
            int thisNum = cursor;
            int thatNum = that.cursor;
            int thisPointer = 1;
            int thatPointer = 1;
            int thatLSPointer = 0;
            int thatSLPointer = 0;

            int thisLSPointer = 0;
            int thisSLPointer = 0;

            boolean thisLSUndefined = true;
            boolean thisSLUndefined = false;

            boolean thatLSUndefined = true;
            boolean thatSLUndefined = false;

            Summary merged = new Summary(getRealNum() + that.getRealNum(), B + that.B, eps, exact);

            while(thisPointer < thisNum && thatPointer < thatNum) {
                while (thisPointer < thisNum && thatPointer < thatNum && this.value[thisPointer] <= that.value[thatPointer]) {
                    while(thatLSPointer < thatNum && that.value[thatLSPointer] < this.value[thisPointer]) {
                        thatLSPointer ++;
                    }
                    thatLSPointer --;

                    if (thatLSUndefined && thatLSPointer >= 1) {
                        thatLSUndefined = false;
                    }

                    if (that.value[thatLSPointer] >= this.value[thisPointer]) {
                        throw new Exception("that ls pointer >= thispointer");
                    }

                    if (!thatSLUndefined) {
                        while(thatSLPointer < thatNum && that.value[thatSLPointer] <= this.value[thisPointer]) {
                            thatSLPointer ++;
                        }
                    }


                    if (!thatSLUndefined && thatSLPointer >= that.cursor) {
                        thatSLUndefined = true;
                    }

                    long rminDelta = thatLSUndefined ? 0 : that.rmin[thatLSPointer];
                    long rmaxDelta = thatSLUndefined ? that.rmax[thatLSPointer] : that.rmax[thatSLPointer] - 1;

                    long rmin = this.rmin[thisPointer] + rminDelta;
                    long rmax = this.rmax[thisPointer] + rmaxDelta;

                    merged.insert(this.value[thisPointer], rmin, rmax);
                    thisPointer ++;

                }

                while (thisPointer < thisNum && thatPointer < thatNum && that.value[thatPointer] <= this.value[thisPointer]) {
                    while(thisLSPointer < thisNum && this.value[thisLSPointer] < that.value[thatPointer]) {
                        thisLSPointer ++;
                    }
                    thisLSPointer --;

                    if (thisLSUndefined && thisLSPointer >= 1) {
                        thisLSUndefined = false;
                    }

                    if (this.value[thisLSPointer] >= that.value[thatPointer]) {
                        throw new Exception("this ls pointer >= thatpointer");
                    }

                    if (!thisSLUndefined) {
                        while(thisSLPointer < thisNum && this.value[thisSLPointer] <= that.value[thatPointer]) {
                            thisSLPointer ++;
                        }
                    }


                    if (!thisSLUndefined && thisSLPointer >= this.cursor) {
                        thisSLUndefined = true;
                    }

                    long rminDelta = thisLSUndefined ? 0 : this.rmin[thisLSPointer];
                    long rmaxDelta = thisSLUndefined ? this.rmax[thisLSPointer] : this.rmax[thisSLPointer] - 1;

                    long rmin = that.rmin[thatPointer] + rminDelta;
                    long rmax = that.rmax[thatPointer] + rmaxDelta;

                    merged.insert(that.value[thatPointer], rmin, rmax);
                    thatPointer ++;
                }
            }

            while (thisPointer < thisNum) {
                while(thatLSPointer < thatNum && that.value[thatLSPointer] < this.value[thisPointer]) {
                    thatLSPointer ++;
                }
                thatLSPointer --;

                if (thatLSUndefined && thatLSPointer >= 1) {
                    thatLSUndefined = false;
                }

                if (that.value[thatLSPointer] >= this.value[thisPointer]) {
                    throw new Exception("that ls pointer >= thispointer");
                }

                if (!thatSLUndefined) {
                    while(thatSLPointer < thatNum && that.value[thatSLPointer] <= this.value[thisPointer]) {
                        thatSLPointer ++;
                    }
                }


                if (!thatSLUndefined && thatSLPointer >= that.cursor) {
                    thatSLUndefined = true;
                }

                long rminDelta = thatLSUndefined ? 0 : that.rmin[thatLSPointer];
                long rmaxDelta = thatSLUndefined ? that.rmax[thatLSPointer] : that.rmax[thatSLPointer] - 1;

                long rmin = this.rmin[thisPointer] + rminDelta;
                long rmax = this.rmax[thisPointer] + rmaxDelta;

                merged.insert(this.value[thisPointer], rmin, rmax);
                thisPointer ++;
            }

            while (thatPointer < thatNum) {
                while(thisLSPointer < thisNum && this.value[thisLSPointer] < that.value[thatPointer]) {
                    thisLSPointer ++;
                }
                thisLSPointer --;

                if (thisLSUndefined && thisLSPointer >= 1) {
                    thisLSUndefined = false;
                }

                if (this.value[thisLSPointer] >= that.value[thatPointer]) {
                    throw new Exception("this ls pointer >= thatpointer");
                }

                if (!thisSLUndefined) {
                    while(thisSLPointer < thisNum && this.value[thisSLPointer] <= that.value[thatPointer]) {
                        thisSLPointer ++;
                    }
                }


                if (!thisSLUndefined && thisSLPointer >= this.cursor) {
                    thisSLUndefined = true;
                }

                long rminDelta = thisLSUndefined ? 0 : this.rmin[thisLSPointer];
                long rmaxDelta = thisSLUndefined ? this.rmax[thisLSPointer] : this.rmax[thisSLPointer] - 1;

                long rmin = that.rmin[thatPointer] + rminDelta;
                long rmax = that.rmax[thatPointer] + rmaxDelta;

                merged.insert(that.value[thatPointer], rmin, rmax);
                thatPointer ++;
            }

            this.insertPoolNode(this.value.length - 1);
            that.insertPoolNode(that.value.length - 1);

            return merged;
        }

        public Summary deepClone() {
            Summary newSummary = new Summary(capacity, B, eps, exact);
            newSummary.value = new float[value.length];
            System.arraycopy(value, 0, newSummary.value, 0, newSummary.value.length);
            newSummary.rmin = new long[rmin.length];
            System.arraycopy(rmin, 0, newSummary.rmin, 0, newSummary.rmin.length);
            newSummary.rmax = new long[rmax.length];
            System.arraycopy(rmax, 0, newSummary.rmax, 0, newSummary.rmax.length);

            newSummary.capacity = capacity;
            newSummary.cursor = cursor;
            newSummary.B = B;
            newSummary.exact = exact;
            return newSummary;
        }
    }

    private static void shuffleArray(float[] array) {
        int index;
        float temp;
        Random random = new Random();
        for (int i = array.length - 1; i > 0; i--)
        {
            index = random.nextInt(i + 1);
            temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }

    public static void main(String []args) throws Exception {

        int n = 10;
        //int errorBound = 100;
        //double eps = errorBound * 1.0 / n;
        //double eps = 0.001;
        //ApproximateQuantile quantile = new ApproximateQuantile(n, errorBound, 10000);
        //ApproximateQuantile quantile = new ApproximateQuantile(n, eps, 10000);
        ApproximateQuantile quantile = new ApproximateQuantile(n, 1L, 100);
        double eps = quantile.getEps();

        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = i + 1;
        }
        shuffleArray(values);

        long start = System.currentTimeMillis();
        Arrays.sort(values);
        System.out.println("sort time:" + (System.currentTimeMillis() - start));


        for (int i = 0; i < n; i++) {
            quantile.update(values[i]);
        }

        Summary summaryAll = quantile.mergeAllAndCompress();
        //summaryAll = summaryAll.compress(quantile.getb(), quantile.getEps());
        System.out.println("build time:" + (System.currentTimeMillis() - start));

        int interval = 1;
        for (int i = 1; i <= n; i += interval) {
            float value = summaryAll.query(i);
            System.out.println("rank:" + i + ", range:[" + (i - eps * n) + "," + (i + eps * n) + "], value:" + value);

            if (!(value >= i - eps * n) && (value <= i + eps * n)) {
                System.out.println("error range:" + i);
            }
        }

        start = System.currentTimeMillis();
        double sum = 0.0;
        for (int i = 1; i <= n; i ++) {
            float value = summaryAll.query(i);
            sum += value;
        }

        System.out.println("avg query time:" + (System.currentTimeMillis() - start) * 1.0 / n + ", result:" + sum);

    }

//    public static void test() {
//        // approximate quantile
//        comm.info("approximate quantile");
//        LOG.info("approximate quantile");
//        int avgNum = 100;
//        int n = workerNum * avgNum;
//        ApproximateQuantile quantile = new ApproximateQuantile(n, 10, 100000);
//        for (int i = rank * avgNum + 1; i < (rank + 1) * avgNum + 1; i++) {
//            quantile.update(i);
//        }
//        ApproximateQuantile.Summary summaryAll = quantile.mergeAllAndCompress();
//        comm.info("before reduce:" + summaryAll);
//        LOG.info("before reduce:" + summaryAll);
//        ApproximateQuantile.Summary mergeSummary = reduce.reduceObject(summaryAll, Operator.CUSTOM, new IObjectCustomOperator<ApproximateQuantile.Summary>() {
//            @Override
//            public ApproximateQuantile.Summary apply(ApproximateQuantile.Summary o1, ApproximateQuantile.Summary o2) {
//                try {
//                    return o1.merge(o2);
//                } catch (Exception e) {
//                    StringWriter sw = new StringWriter();
//                    PrintWriter pw = new PrintWriter(sw);
//                    e.printStackTrace(pw);
//
//                    try {
//                        comm.error("slave exception:" + sw.toString());
//                    } catch (Mp4jException e1) {
//                        e1.printStackTrace();
//                    }
//                    try {
//                        Thread.sleep(5000);
//                    } catch (InterruptedException e1) {
//                        e1.printStackTrace();
//                    }
//                    try {
//                        reduce.close(1);
//                    } catch (Mp4jException e1) {
//                        e1.printStackTrace();
//                    }
//                    return null;
//                }
//            }
//        });
//
//        comm.info("after reduce:" + mergeSummary);
//        LOG.info("after reduce:" + mergeSummary);
//
//        for (int i = 1; i < workerNum * avgNum + 1; i += 1) {
//            comm.info("rank:" + i + ", value:" + mergeSummary.query(i, quantile.getEps()));
//            LOG.info("rank:" + i + ", value:" + mergeSummary.query(i, quantile.getEps()));
//        }
//
//
//    }


}
