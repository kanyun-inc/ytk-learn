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

package com.fenbi.ytklearn.data.gbdt;

import com.fenbi.ytklearn.exception.YtkLearnException;
import com.fenbi.ytklearn.utils.CheckUtils;

import java.util.*;

/**
 * @author wufan
 * @author xialong
 */

public class HistogramPool {

    // size of primitive double
    public final static int UNIT = 64;

    static class Node<E> {
        E item;
        Node<E> next;
        Node<E> prev;

        Node(Node<E> prev, E element, Node<E> next) {
            this.item = element;
            this.next = next;
            this.prev = prev;
        }
    }

    static class SimpleLinkedList<E> {
        int size = 0;
        Node<E> first;
        Node<E> last;

        SimpleLinkedList() {

        }

        E removeFirst() {
            final Node<E> f = first;
            if (f == null)
                throw new NoSuchElementException();
            return unlinkFirst(f);
        }

        void addLast(E e) {
            if (e == null) {
                throw new NoSuchElementException();
            } else {
                linkLast(e);
            }
        }

        E remove(Node<E> x) {
            if (x == null) {
                throw new NoSuchElementException();
            } else {
                return unlink(x);
            }
        }

        int size() {
            return size;
        }

        boolean isEmpty() {
            return size == 0;
        }

        void linkLast(E e) {
            final Node<E> l = last;
            final Node<E> newNode = new Node<>(l, e, null);
            last = newNode;
            if (l == null)
                first = newNode;
            else
                l.next = newNode;
            size++;
        }


        private E unlinkFirst(Node<E> f) {
            // assert f == first && f != null;
            final E element = f.item;
            final Node<E> next = f.next;
            f.item = null;
            f.next = null; // help GC
            first = next;
            if (next == null)
                last = null;
            else
                next.prev = null;
            size--;
            return element;
        }

        private E unlink(Node<E> x) {
            // assert x != null;
            final E element = x.item;
            final Node<E> next = x.next;
            final Node<E> prev = x.prev;

            if (prev == null) {
                first = next;
            } else {
                prev.next = next;
                x.prev = null;
            }

            if (next == null) {
                last = prev;
            } else {
                next.prev = prev;
                x.next = null;
            }

            x.item = null;
            size--;
            return element;
        }

        void clear() {
            for (Node<E> x = first; x != null; ) {
                Node<E> next = x.next;
                x.item = null;
                x.next = null;
                x.prev = null;
                x = next;
            }
            first = last = null;
            size = 0;
        }
    }

    static class Item {
        int nid;
        // local grad sum in each feature bin, get global grad sum after aggregate
        double[] gradSum;

        Item(int nid, int len) {
            this.nid = nid;
            gradSum = new double[len];
            for (int i = 0; i < gradSum.length; i++) {
                gradSum[i] = 0;
            }
        }

        void clear() {
            nid = -1;
            for (int i = 0; i < gradSum.length; i++) {
                gradSum[i] = 0;
            }
        }
    }

    private Map<Integer, Node<Item>> nid2Iter;

    private SimpleLinkedList<Item> used;
    private Queue<Item> avail;

    private int curSize;
    private int capacity;
    private int numGrad;
    private int[] gradIndexRange;


    public HistogramPool(int capacity, int numLocalGrad, int[] gradIndexRange) {
        // assert capacity >= 2
        this.capacity = capacity;
        this.numGrad = numLocalGrad; // =2*actual_bin_cnt
        this.curSize = 0;
        this.nid2Iter = new HashMap<>(capacity);
        this.used = new SimpleLinkedList<>();
        this.avail = new LinkedList<>();
        this.gradIndexRange = gradIndexRange;
        CheckUtils.check(gradIndexRange[1] - gradIndexRange[0] == numGrad,
                "[GBDT] inner error! gradient sum array index range[%d, %d) should equal to 2*numGrad(%d)", gradIndexRange[1], gradIndexRange[0], numGrad);

    }

    public void clear() {
        nid2Iter.clear();
        while (!used.isEmpty()) {
            Item item = used.removeFirst();
            avail.add(item);
        }
    }

    private double[] allocHist(int nid) {
        if (nid2Iter.containsKey(nid)) {
            throw new YtkLearnException("[GBDT] nid(" + nid + ") exist before allocated!");
        }
        Item histItem;
        if (avail.size() > 0) {
            histItem = avail.poll();
//            histItem.clear();
            histItem.nid = nid;

        } else if (curSize < capacity) {
            histItem = new Item(nid, numGrad);
            curSize++;

        } else {
            histItem = used.removeFirst();
            nid2Iter.remove(histItem.nid);
//            histItem.clear();
            histItem.nid = nid;
        }

        used.addLast(histItem);
        nid2Iter.put(nid, used.last);
        return histItem.gradSum;
    }

    public double[] getHist(int nid) {
        Node<Item> iter = nid2Iter.get(nid);
        if (iter != null) {
            return iter.item.gradSum;
        }
        return null;

    }

    public double[] setHist(int nid, double[] histogram) {
        double[] innerHistogram = allocHist(nid);
        int index = 0;
        for (int i = gradIndexRange[0]; i < gradIndexRange[1]; i++) {
            innerHistogram[index++] = histogram[i];
        }
        return innerHistogram;
    }

    public boolean releaseHist(int nid) {
        Node<Item> iter = nid2Iter.get(nid);
        if (iter != null) {
            Item histItem = iter.item;
            nid2Iter.remove(nid);
            used.remove(iter);
            avail.add(histItem);
            return true;
        } else {
            return false;
        }
    }

    public int getCurSize() {
        return curSize;
    }

}
