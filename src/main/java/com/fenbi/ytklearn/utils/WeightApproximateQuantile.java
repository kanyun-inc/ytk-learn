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

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.fenbi.ytklearn.data.Constants;

import java.io.Serializable;
import java.util.*;

/**
 * @author xialong
 */

public class WeightApproximateQuantile implements Serializable {

    private double eps;
    private int b;

    private long n;

    private Summary summarys[];
    private int L = 0;
    private boolean exact;

    public WeightApproximateQuantile(long n, double eps, long exactThre) {
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
        //system.out.println("n:" + n + ", b:" + b + ", eps:" + eps +
//                ", storage use:" + (log2(eps * n) * log2(eps * n) / eps) + ", exact:" + exact);
        summarys = new Summary[500];
        summarys[L++] = new Summary(b, 0.0, eps, exact);
    }

    public WeightApproximateQuantile(long n, long errorBound, long exactThre) {
        this(n, errorBound * 1.0 / n, exactThre);
    }

    public WeightApproximateQuantile(long n, int bins, long exactThre) {
        this(n, 1. / (Constants.QUNANTILE_APPROXIMATE_BIN_FACTOR * bins), exactThre);
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

    public void update(float value, float w) throws Exception {
        // insert && if full
        if (summarys[0].insert(value, w)) {
            summarys[0].sort();
            Summary sc = summarys[0].compress(b);
            summarys[0] = new Summary(b, 0.0, eps, exact);

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
        //system.out.println("L:" + L);
        summarys[0].sort();
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
        public double []rmin;
        public double []rmax;
        public float []w;


        public PoolNode() {}

        public PoolNode(float[] value, double[] rmin, double[] rmax, float[] w) {
            this.value = value;
            this.rmin = rmin;
            this.rmax = rmax;
            this.w = w;

        }
    }


    public static class SummarySerializer extends Serializer<Summary> {

        @Override
        public void write(Kryo kryo, Output output, Summary object) {
            output.writeInt(object.value.length);
            for (int i = 0; i < object.value.length; i++) {
                output.writeFloat(object.value[i]);
            }

            output.writeInt(object.rmin.length);
            for (int i = 0; i < object.rmin.length; i++) {
                output.writeDouble(object.rmin[i]);
            }

            output.writeInt(object.rmax.length);
            for (int i = 0; i < object.rmax.length; i++) {
                output.writeDouble(object.rmax[i]);
            }

            output.writeInt(object.w.length);
            for (int i = 0; i < object.w.length; i++) {
                output.writeFloat(object.w[i]);
            }

            output.writeInt(object.capacity);
            output.writeInt(object.cursor);
            output.writeDouble(object.B);
            output.writeBoolean(object.exact);
            output.writeDouble(object.eps);
        }

        @Override
        public Summary read(Kryo kryo, Input input, Class<Summary> type) {
            Summary summary = new Summary();
            int len = input.readInt();
            summary.value = new float[len];
            for (int i = 0; i < len; i++) {
                summary.value[i] = input.readFloat();
            }

            len = input.readInt();
            summary.rmin = new double[len];
            for (int i = 0; i < len; i++) {
                summary.rmin[i] = input.readDouble();
            }

            len = input.readInt();
            summary.rmax = new double[len];
            for (int i = 0; i < len; i++) {
                summary.rmax[i] = input.readDouble();
            }

            len = input.readInt();
            summary.w = new float[len];
            for (int i = 0; i < len; i++) {
                summary.w[i] = input.readFloat();
            }

            summary.capacity = input.readInt();
            summary.cursor = input.readInt();
            summary.B = input.readDouble();
            summary.exact = input.readBoolean();
            summary.eps = input.readDouble();

            summary.poolMap = new TreeMap<>();
            PoolNode poolNode = summary.getPoolNode(summary.capacity);

            return summary;
        }
    }

    public static class Summary {
        // value
        float []value;
        // rmin, rmax, ...
        double []rmin;
        double []rmax;
        float []w;
        int capacity;
        int cursor = 1;
        double B = 0;
        boolean exact;
        double eps;
        transient TreeMap<Double, PoolNode> poolMap;
        Random rand = new Random();


        public Summary() {}

        public Summary(int capacity, double B, double eps, boolean exact) {
            this.poolMap = new TreeMap<>();
            PoolNode poolNode = this.getPoolNode(capacity);

            this.exact = exact;
            value = poolNode.value;
            rmin = poolNode.rmin;
            rmax = poolNode.rmax;
            w = poolNode.w;
            value[0] = -Float.MAX_VALUE;
            rmin[0] = 0;
            rmax[0] = 0;
            w[0] = 0.0f;
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

        public boolean insert(float value, float weight) {
            this.value[cursor] = value;
            this.w[cursor] = weight;
            this.B += weight;
            cursor ++;
            return cursor > capacity;
        }

        private boolean mergeInsert(float value, double rmin, double rmax, float w) {
            this.value[cursor] = value;
            this.rmin[cursor] = rmin;
            this.rmax[cursor] = rmax;
            this.w[cursor] = w;
            cursor++;
            return cursor > capacity;
        }

        public void sort() {
            //Arrays.sort(value, 1, cursor);
            TupleDualPivotSort.sortTuple(value, w, 1, cursor);
            for (int i = 1; i < cursor; i++) {
                rmin[i] = rmin[i - 1] + w[i - 1];
                rmax[i] = rmin[i] + w[i];
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

        public Double getB() {
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
            //system.out.println("before get pool size:" + poolMap.size() + ", get len:" + len + ", keys:" + poolMap.keySet());
            len ++;
            if (poolEntry == null) {
                PoolNode poolNode = new PoolNode();
                poolNode.value = new float[len];
                poolNode.rmin = new double[len];
                poolNode.rmax = new double[len];
                poolNode.w = new float[len];
                return poolNode;
            } else {
                poolMap.remove(poolEntry.getKey());
                //system.out.println("after get pool size:" + poolMap.size() + ", get len:" + len + ", keys:" + poolMap.keySet());
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
            //system.out.println("before insert pool size:" + poolMap.size() + ", insert len:" + len);
            poolMap.put(len + rand.nextDouble(), new PoolNode(value, rmin, rmax, w));
            //system.out.println("after insert pool size:" + poolMap.size() + ", keys:" + poolMap.keySet());
        }

        public void fix() {
            // fix rank1
            if (cursor <= 1)
                return;

            if (exact) {
                return;
            }

            double epsB = eps * B;
            if (rmin[1] != 0.0) {
                rmin[1] = 0.0;
            }

            for (int i = 2; i < cursor; i++) {
                double prevmin = rmin[i - 1] + epsB;
                double curmax = rmax[i] - epsB;

                if (curmax > prevmin + 1) {
                    rmin[i - 1] = rmax[i] - 2 * epsB - 1;
                }
            }
        }

        public float query(double r) {
            return value[queryIdx(r)];
        }

        public int queryIdx(double r) {
            double rnew = exact ? r * B : r * B + eps * B;
            int idx = Arrays.binarySearch(rmax, 1, cursor, rnew);
            if (exact) {
                idx++;
            }

            idx = idx >= 0 ? idx : (-idx) - 2;
            if (idx < 1) {
                idx = 1;
            } else if (idx >= cursor) {
                idx = cursor - 1;
            }

            return idx;
        }


        public Summary compress(int b) throws Exception {
            if (exact) {
                return this;
            }

            if (cursor <= 1) {
                return this;
            }

            double avg = 2.0 / b;
            double r = 0.0;
            int handledCnt = 1;
            int pointer = 1;
            int newSize = ((int)Math.ceil(b / 2)) + 2;
            PoolNode poolNode = getPoolNode(newSize);
            poolNode.value[0] = -Float.MAX_VALUE;
            poolNode.rmin[0] = 0;
            poolNode.rmax[0] = 0;
            poolNode.w[0] = 0.0f;
            while(true) {

                pointer = queryIdx(r);

                poolNode.value[handledCnt] = value[pointer];
                poolNode.rmin[handledCnt] = rmin[pointer];
                poolNode.rmax[handledCnt] = rmax[pointer];
                poolNode.w[handledCnt] = w[pointer];

                handledCnt ++;
                r += avg;
                if (r >= 1.0) {
                    if (r == 1.0) {
                        break;
                    } else {
                        if (handledCnt < newSize) {
                            r = 1.0;
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
            w = poolNode.w;
            capacity = handledCnt - 1;
            cursor = handledCnt;

            //system.out.println("compressed size:" + (handledCnt - 1) + ", at most size:" + newSize);
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
                    while(thatLSPointer < thatNum && that.value[thatLSPointer] <= this.value[thisPointer]) {
                        thatLSPointer ++;
                    }
                    thatLSPointer --;

                    if (thatLSUndefined && thatLSPointer >= 1) {
                        thatLSUndefined = false;
                    }

//                    if (that.value[thatLSPointer] >= this.value[thisPointer]) {
//                        throw new Exception("that ls pointer >= thispointer");
//                    }

                    if (!thatSLUndefined) {
                        while(thatSLPointer < thatNum && that.value[thatSLPointer] < this.value[thisPointer]) {
                            thatSLPointer ++;
                        }
                    }


                    if (!thatSLUndefined && thatSLPointer >= that.cursor) {
                        thatSLUndefined = true;
                    }

                    double rminDelta = thatLSUndefined ? 0 : that.rmin[thatLSPointer];
                    double rmaxDelta = thatSLUndefined ? that.rmax[thatLSPointer] : that.rmax[thatSLPointer] - that.w[thatSLPointer];

                    double rmin = this.rmin[thisPointer] + rminDelta;
                    double rmax = this.rmax[thisPointer] + rmaxDelta;
                    float w = this.w[thisPointer];

                    merged.mergeInsert(this.value[thisPointer], rmin, rmax, w);
                    thisPointer ++;

                }

                while (thisPointer < thisNum && thatPointer < thatNum && that.value[thatPointer] <= this.value[thisPointer]) {
                    while(thisLSPointer < thisNum && this.value[thisLSPointer] <= that.value[thatPointer]) {
                        thisLSPointer ++;
                    }
                    thisLSPointer --;

                    if (thisLSUndefined && thisLSPointer >= 1) {
                        thisLSUndefined = false;
                    }

//                    if (this.value[thisLSPointer] >= that.value[thatPointer]) {
//                        throw new Exception("this ls pointer >= thatpointer");
//                    }

                    if (!thisSLUndefined) {
                        while(thisSLPointer < thisNum && this.value[thisSLPointer] < that.value[thatPointer]) {
                            thisSLPointer ++;
                        }
                    }


                    if (!thisSLUndefined && thisSLPointer >= this.cursor) {
                        thisSLUndefined = true;
                    }

                    double rminDelta = thisLSUndefined ? 0 : this.rmin[thisLSPointer];
                    double rmaxDelta = thisSLUndefined ? this.rmax[thisLSPointer] : this.rmax[thisSLPointer] - this.w[thisSLPointer];

                    double rmin = that.rmin[thatPointer] + rminDelta;
                    double rmax = that.rmax[thatPointer] + rmaxDelta;
                    float w = that.w[thatPointer];

                    merged.mergeInsert(that.value[thatPointer], rmin, rmax, w);
                    thatPointer ++;
                }
            }

            while (thisPointer < thisNum) {
                while(thatLSPointer < thatNum && that.value[thatLSPointer] <= this.value[thisPointer]) {
                    thatLSPointer ++;
                }
                thatLSPointer --;

                if (thatLSUndefined && thatLSPointer >= 1) {
                    thatLSUndefined = false;
                }

//                if (that.value[thatLSPointer] >= this.value[thisPointer]) {
//                    throw new Exception("that ls pointer >= thispointer");
//                }

                if (!thatSLUndefined) {
                    while(thatSLPointer < thatNum && that.value[thatSLPointer] < this.value[thisPointer]) {
                        thatSLPointer ++;
                    }
                }


                if (!thatSLUndefined && thatSLPointer >= that.cursor) {
                    thatSLUndefined = true;
                }

                double rminDelta = thatLSUndefined ? 0 : that.rmin[thatLSPointer];
                double rmaxDelta = thatSLUndefined ? that.rmax[thatLSPointer] : that.rmax[thatSLPointer] - that.w[thatSLPointer];

                double rmin = this.rmin[thisPointer] + rminDelta;
                double rmax = this.rmax[thisPointer] + rmaxDelta;
                float w = this.w[thisPointer];

                merged.mergeInsert(this.value[thisPointer], rmin, rmax, w);
                thisPointer ++;
            }

            while (thatPointer < thatNum) {
                while(thisLSPointer < thisNum && this.value[thisLSPointer] <= that.value[thatPointer]) {
                    thisLSPointer ++;
                }
                thisLSPointer --;

                if (thisLSUndefined && thisLSPointer >= 1) {
                    thisLSUndefined = false;
                }

//                if (this.value[thisLSPointer] >= that.value[thatPointer]) {
//                    throw new Exception("this ls pointer >= thatpointer");
//                }

                if (!thisSLUndefined) {
                    while(thisSLPointer < thisNum && this.value[thisSLPointer] < that.value[thatPointer]) {
                        thisSLPointer ++;
                    }
                }


                if (!thisSLUndefined && thisSLPointer >= this.cursor) {
                    thisSLUndefined = true;
                }

                double rminDelta = thisLSUndefined ? 0 : this.rmin[thisLSPointer];
                double rmaxDelta = thisSLUndefined ? this.rmax[thisLSPointer] : this.rmax[thisSLPointer] - this.w[thisSLPointer];

                double rmin = that.rmin[thatPointer] + rminDelta;
                double rmax = that.rmax[thatPointer] + rmaxDelta;
                float w = that.w[thatPointer];

                merged.mergeInsert(that.value[thatPointer], rmin, rmax, w);
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
            newSummary.rmin = new double[rmin.length];
            System.arraycopy(rmin, 0, newSummary.rmin, 0, newSummary.rmin.length);
            newSummary.rmax = new double[rmax.length];
            System.arraycopy(rmax, 0, newSummary.rmax, 0, newSummary.rmax.length);
            newSummary.w = new float[w.length];
            System.arraycopy(w, 0, newSummary.w, 0, newSummary.w.length);

            newSummary.capacity = capacity;
            newSummary.cursor = cursor;
            newSummary.B = B;
            newSummary.exact = exact;
            newSummary.eps = eps;
            newSummary.rand = new Random();
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
//
//        int n = 1000;
//        float weight = 1f;
//        long errorBound = 0L;
//        //int errorBound = 100;
//        Random rand = new Random();
//        //double eps = errorBound * 1.0 / n;
//        //double eps = 0.001;
//        WeightApproximateQuantile quantile = new WeightApproximateQuantile(n, errorBound, 100);
//        double eps = quantile.getEps();
//
//        float[] values = new float[n];
//        for (int i = 0; i < n; i++) {
//            values[i] = i + 1;
//        }
//        shuffleArray(values);
//
//        long start = System.currentTimeMillis();
//        Arrays.sort(values);
//        System.out.println("sort time:" + (System.currentTimeMillis() - start));
//
//
//        Set<Integer> set = new HashSet<>();
////        set.add(1);
////        set.add(5);
////        set.add(10);
//        set.add(1);
//        set.add(500);
//        set.add(n);
//        for (int i = 1; i <= n; i++) {
//            if (!set.contains(i)) {
//                quantile.update(values[i - 1], weight);
//            } else {
//                quantile.update(values[i - 1], 200000f);
//            }
//
//        }
//
//
//        Summary summaryAll = quantile.mergeAllAndCompress();
//        System.out.println("build time:" + (System.currentTimeMillis() - start));
//
//        int num = 1000;
//        int errorCnt = 0;
//        double maxDiff = -100;
//        for (double r = 0; r <= 1; r += (1.0 / num)) {
//            float value = summaryAll.query(r);
//            System.out.println("rank:" + r * n + ", value:" + value);
//            double diff = Math.abs(r * n - value);
//            if (diff > maxDiff) {
//                maxDiff = diff;
//            }
//            if (diff > errorBound) {
//                errorCnt ++;
//
//            }
//        }
//        float value = summaryAll.query(1.0);
//        System.out.println("rank:" + n + ", value:" + value);
//        System.out.println("error cnt:" + errorCnt + ", max diff:" + maxDiff);

//        String fname = "/Users/xialong/data/fea_8";
//        Random rand = new Random();
//        BufferedReader reader = new BufferedReader(new FileReader(new File(fname)));
//        int sampleCnt = 10500000;
//        float[] data = new float[sampleCnt];
//        String line = "";
//        int cnt = 0;
//        while ((line = reader.readLine()) != null) {
//            data[cnt] = Float.parseFloat(line);
//            cnt++;
//        }
        int cnt = 10000;
        float []data = new float[cnt];
        Random rand = new Random();
        for (int i = 0; i < cnt; i++) {
//            float val;
//            if (i < cnt * 0.1) {
//                val = 0f;
//            } else if (i < cnt * 0.3) {
//                val = 1f;
//            } else if (i < cnt * 0.7) {
//                val = 2f;
//            } else {
//                val = 3f;
//            }
            data[i] = i;

        }

        for (int i = 1; i < cnt; i++) {
            int idx = rand.nextInt(i + 1);
            float tmp = data[i];
            data[i] = data[idx];
            data[idx] = tmp;
        }

        WeightApproximateQuantile quantile = new WeightApproximateQuantile(data.length,
                0.0001, 100);
        for (int i = 0; i < data.length; i++) {
            quantile.update(data[i], 1);
        }

        WeightApproximateQuantile.Summary summaryAll = quantile.mergeAllAndCompress();
        int t = 10000;
        for (int i = 0; i < t + 1; i++) {
            System.out.println(i * 1.0 / t + ":" + summaryAll.query(i * 1.0 / t));
        }



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
