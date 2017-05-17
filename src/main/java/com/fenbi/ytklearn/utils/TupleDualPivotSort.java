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

import java.util.Random;

/**
 * @author xialong
 */

public final class TupleDualPivotSort {
    public static void sortTuple(float[] a, float[] b) {
        sortTuple(a, b, 0, a.length);
    }
    public static void sortTuple(float[] a, float[] b, int fromIndex, int toIndex) {
        rangeCheck(a.length, fromIndex, toIndex);
        rangeCheck(b.length, fromIndex, toIndex);
        dualPivotQuicksortTuple(a, b, fromIndex, toIndex - 1, 3);
    }
    private static void rangeCheck(int length, int fromIndex, int toIndex) {
        if (fromIndex > toIndex) {
            throw new IllegalArgumentException("fromIndex > toIndex");
        }
        if (fromIndex < 0) {
            throw new ArrayIndexOutOfBoundsException(fromIndex);
        }
        if (toIndex > length) {
            throw new ArrayIndexOutOfBoundsException(toIndex);
        }
    }
    private static void swap(float[] a, int i, int j) {
        float temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
    private static void dualPivotQuicksortTuple(float[] a, float[] b, int left,int right, int div) {
        int len = right - left;
        if (len < 27) {
            for (int i = left + 1; i <= right; i++) {
                for (int j = i; j > left && a[j] < a[j - 1]; j--) {
                    swap(a, j, j - 1);
                    swap(b, j, j - 1);
                }
            }
            return;
        }
        int third = len / div;
        int m1 = left + third;
        int m2 = right - third;
        if (m1 <= left) {
            m1 = left + 1;
        }
        if (m2 >= right) {
            m2 = right - 1;
        }
        if (a[m1] < a[m2]) {
            swap(a, m1, left);
            swap(a, m2, right);

            swap(b, m1, left);
            swap(b, m2, right);
        }
        else {
            swap(a, m1, right);
            swap(a, m2, left);

            swap(b, m1, right);
            swap(b, m2, left);
        }
        float pivot1 = a[left];
        float pivot2 = a[right];
        int less = left + 1;
        int great = right - 1;
        for (int k = less; k <= great; k++) {
            if (a[k] < pivot1) {
                swap(a, k, less);
                swap(b, k, less);
                less ++;
            }
            else if (a[k] > pivot2) {
                while (k < great && a[great] > pivot2) {
                    great--;
                }
                swap(a, k, great);
                swap(b, k, great);
                great --;
                if (a[k] < pivot1) {
                    swap(a, k, less);
                    swap(b, k, less);
                    less ++;
                }
            }
        }
        int dist = great - less;
        if (dist < 13) {
            div++;
        }
        swap(a, less - 1, left);
        swap(a, great + 1, right);

        swap(b, less - 1, left);
        swap(b, great + 1, right);

        dualPivotQuicksortTuple(a, b, left, less - 2, div);
        dualPivotQuicksortTuple(a, b, great + 2, right, div);

        if (dist > len - 13 && pivot1 != pivot2) {
            for (int k = less; k <= great; k++) {
                if (a[k] == pivot1) {
                    swap(a, k, less);
                    swap(b, k, less);
                    less ++;
                }
                else if (a[k] == pivot2) {
                    swap(a, k, great);
                    swap(b, k, great);
                    great --;
                    if (a[k] == pivot1) {
                        swap(a, k, less);
                        swap(b, k, less);
                        less ++;
                    }
                }
            }
        }

        if (pivot1 < pivot2) {
            dualPivotQuicksortTuple(a, b, less, great, div);
        }
    }

    public static void sortTuple(float[] a) {
        sortTuple(a, 0, a.length >> 1);
    }
    public static void sortTuple(float[] a, int fromIndex, int toIndex) {
        rangeCheck(a.length >> 1, fromIndex, toIndex);
        dualPivotQuicksortTuple(a, fromIndex, toIndex - 1, 3);
    }

    private static void swap2(float[] a, int i, int j) {
        int iidx = i << 1;
        int jidx = j << 1;
        float temp = a[iidx];
        float temp1 = a[iidx + 1];
        a[iidx] = a[jidx];
        a[iidx + 1] = a[jidx + 1];
        a[jidx] = temp;
        a[jidx + 1] = temp1;


    }

    private static void dualPivotQuicksortTuple(float[] a, int left, int right, int div) {
        int len = right - left;
        if (len < 27) {
            for (int i = left + 1; i <= right; i++) {
                for (int j = i; j > left && a[j << 1] < a[(j - 1) << 1]; j--) {
                    swap2(a, j, j - 1);
                }
            }
            return;
        }
        int third = len / div;
        int m1 = left + third;
        int m2 = right - third;
        if (m1 <= left) {
            m1 = left + 1;
        }
        if (m2 >= right) {
            m2 = right - 1;
        }
        if (a[m1 << 1] < a[m2 << 1]) {
            swap2(a, m1, left);
            swap2(a, m2, right);
        }
        else {
            swap2(a, m1, right);
            swap2(a, m2, left);
        }

        float pivot1 = a[left << 1];
        float pivot2 = a[right << 1];

        int less = left + 1;
        int great = right - 1;
        for (int k = less; k <= great; k++) {
            if (a[k << 1] < pivot1) {
                swap2(a, k, less++);
            }
            else if (a[k << 1] > pivot2) {
                while (k < great && a[great << 1] > pivot2) {
                    great--;
                }
                swap2(a, k, great--);
                if (a[k << 1] < pivot1) {
                    swap2(a, k, less++);
                }
            }
        }
        int dist = great - less;
        if (dist < 13) {
            div++;
        }
        swap2(a, less - 1, left);
        swap2(a, great + 1, right);

        dualPivotQuicksortTuple(a, left, less - 2, div);
        dualPivotQuicksortTuple(a, great + 2, right, div);

        if (dist > len - 13 && pivot1 != pivot2) {
            for (int k = less; k <= great; k++) {
                if (a[k << 1] == pivot1) {
                    swap2(a, k, less++);
                }
                else if (a[k << 1] == pivot2) {
                    swap2(a, k, great--);
                    if (a[k << 1] == pivot1) {
                        swap2(a, k, less++);
                    }
                }
            }
        }

        if (pivot1 < pivot2) {
            dualPivotQuicksortTuple(a, less, great, div);
        }
    }


    public static void sortTuple(double[] a, double[] b) {
        sortTuple(a, b, 0, a.length);
    }
    public static void sortTuple(double[] a, double[] b, int fromIndex, int toIndex) {
        rangeCheck(a.length, fromIndex, toIndex);
        rangeCheck(b.length, fromIndex, toIndex);
        dualPivotQuicksortTuple(a, b, fromIndex, toIndex - 1, 3);
    }

    private static void swap(double[] a, int i, int j) {
        double temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
    private static void dualPivotQuicksortTuple(double[] a, double[] b, int left,int right, int div) {
        int len = right - left;
        if (len < 27) {
            for (int i = left + 1; i <= right; i++) {
                for (int j = i; j > left && a[j] < a[j - 1]; j--) {
                    swap(a, j, j - 1);
                    swap(b, j, j - 1);
                }
            }
            return;
        }
        int third = len / div;
        int m1 = left + third;
        int m2 = right - third;
        if (m1 <= left) {
            m1 = left + 1;
        }
        if (m2 >= right) {
            m2 = right - 1;
        }
        if (a[m1] < a[m2]) {
            swap(a, m1, left);
            swap(a, m2, right);

            swap(b, m1, left);
            swap(b, m2, right);
        }
        else {
            swap(a, m1, right);
            swap(a, m2, left);

            swap(b, m1, right);
            swap(b, m2, left);
        }
        double pivot1 = a[left];
        double pivot2 = a[right];
        int less = left + 1;
        int great = right - 1;
        for (int k = less; k <= great; k++) {
            if (a[k] < pivot1) {
                swap(a, k, less);
                swap(b, k, less);
                less ++;
            }
            else if (a[k] > pivot2) {
                while (k < great && a[great] > pivot2) {
                    great--;
                }
                swap(a, k, great);
                swap(b, k, great);
                great --;
                if (a[k] < pivot1) {
                    swap(a, k, less);
                    swap(b, k, less);
                    less ++;
                }
            }
        }
        int dist = great - less;
        if (dist < 13) {
            div++;
        }
        swap(a, less - 1, left);
        swap(a, great + 1, right);

        swap(b, less - 1, left);
        swap(b, great + 1, right);

        dualPivotQuicksortTuple(a, b, left, less - 2, div);
        dualPivotQuicksortTuple(a, b, great + 2, right, div);

        if (dist > len - 13 && pivot1 != pivot2) {
            for (int k = less; k <= great; k++) {
                if (a[k] == pivot1) {
                    swap(a, k, less);
                    swap(b, k, less);
                    less ++;
                }
                else if (a[k] == pivot2) {
                    swap(a, k, great);
                    swap(b, k, great);
                    great --;
                    if (a[k] == pivot1) {
                        swap(a, k, less);
                        swap(b, k, less);
                        less ++;
                    }
                }
            }
        }

        if (pivot1 < pivot2) {
            dualPivotQuicksortTuple(a, b, less, great, div);
        }
    }

    public static void sortTuple(double[] a) {
        sortTuple(a, 0, a.length >> 1);
    }
    public static void sortTuple(double[] a, int fromIndex, int toIndex) {
        rangeCheck(a.length >> 1, fromIndex, toIndex);
        dualPivotQuicksortTuple(a, fromIndex, toIndex - 1, 3);
    }

    private static void swap2(double[] a, int i, int j) {
        int iidx = i << 1;
        int jidx = j << 1;
        double temp = a[iidx];
        double temp1 = a[iidx + 1];
        a[iidx] = a[jidx];
        a[iidx + 1] = a[jidx + 1];
        a[jidx] = temp;
        a[jidx + 1] = temp1;


    }

    private static void dualPivotQuicksortTuple(double[] a, int left, int right, int div) {
        int len = right - left;
        if (len < 27) {
            for (int i = left + 1; i <= right; i++) {
                for (int j = i; j > left && a[j << 1] < a[(j - 1) << 1]; j--) {
                    swap2(a, j, j - 1);
                }
            }
            return;
        }
        int third = len / div;
        int m1 = left + third;
        int m2 = right - third;
        if (m1 <= left) {
            m1 = left + 1;
        }
        if (m2 >= right) {
            m2 = right - 1;
        }
        if (a[m1 << 1] < a[m2 << 1]) {
            swap2(a, m1, left);
            swap2(a, m2, right);
        }
        else {
            swap2(a, m1, right);
            swap2(a, m2, left);
        }

        double pivot1 = a[left << 1];
        double pivot2 = a[right << 1];

        int less = left + 1;
        int great = right - 1;
        for (int k = less; k <= great; k++) {
            if (a[k << 1] < pivot1) {
                swap2(a, k, less++);
            }
            else if (a[k << 1] > pivot2) {
                while (k < great && a[great << 1] > pivot2) {
                    great--;
                }
                swap2(a, k, great--);
                if (a[k << 1] < pivot1) {
                    swap2(a, k, less++);
                }
            }
        }
        int dist = great - less;
        if (dist < 13) {
            div++;
        }
        swap2(a, less - 1, left);
        swap2(a, great + 1, right);

        dualPivotQuicksortTuple(a, left, less - 2, div);
        dualPivotQuicksortTuple(a, great + 2, right, div);

        if (dist > len - 13 && pivot1 != pivot2) {
            for (int k = less; k <= great; k++) {
                if (a[k << 1] == pivot1) {
                    swap2(a, k, less++);
                }
                else if (a[k << 1] == pivot2) {
                    swap2(a, k, great--);
                    if (a[k << 1] == pivot1) {
                        swap2(a, k, less++);
                    }
                }
            }
        }

        if (pivot1 < pivot2) {
            dualPivotQuicksortTuple(a, less, great, div);
        }
    }

    public static void sortTuple(int[] a, int[] b) {
        sortTuple(a, b, 0, a.length);
    }
    public static void sortTuple(int[] a, int[] b, int fromIndex, int toIndex) {
        rangeCheck(a.length, fromIndex, toIndex);
        rangeCheck(b.length, fromIndex, toIndex);
        dualPivotQuicksortTuple(a, b, fromIndex, toIndex - 1, 3);
    }

    private static void swap(int[] a, int i, int j) {
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
    private static void dualPivotQuicksortTuple(int[] a, int[] b, int left,int right, int div) {
        int len = right - left;
        if (len < 27) {
            for (int i = left + 1; i <= right; i++) {
                for (int j = i; j > left && a[j] < a[j - 1]; j--) {
                    swap(a, j, j - 1);
                    swap(b, j, j - 1);
                }
            }
            return;
        }
        int third = len / div;
        int m1 = left + third;
        int m2 = right - third;
        if (m1 <= left) {
            m1 = left + 1;
        }
        if (m2 >= right) {
            m2 = right - 1;
        }
        if (a[m1] < a[m2]) {
            swap(a, m1, left);
            swap(a, m2, right);

            swap(b, m1, left);
            swap(b, m2, right);
        }
        else {
            swap(a, m1, right);
            swap(a, m2, left);

            swap(b, m1, right);
            swap(b, m2, left);
        }
        int pivot1 = a[left];
        int pivot2 = a[right];
        int less = left + 1;
        int great = right - 1;
        for (int k = less; k <= great; k++) {
            if (a[k] < pivot1) {
                swap(a, k, less);
                swap(b, k, less);
                less ++;
            }
            else if (a[k] > pivot2) {
                while (k < great && a[great] > pivot2) {
                    great--;
                }
                swap(a, k, great);
                swap(b, k, great);
                great --;
                if (a[k] < pivot1) {
                    swap(a, k, less);
                    swap(b, k, less);
                    less ++;
                }
            }
        }
        int dist = great - less;
        if (dist < 13) {
            div++;
        }
        swap(a, less - 1, left);
        swap(a, great + 1, right);

        swap(b, less - 1, left);
        swap(b, great + 1, right);

        dualPivotQuicksortTuple(a, b, left, less - 2, div);
        dualPivotQuicksortTuple(a, b, great + 2, right, div);

        if (dist > len - 13 && pivot1 != pivot2) {
            for (int k = less; k <= great; k++) {
                if (a[k] == pivot1) {
                    swap(a, k, less);
                    swap(b, k, less);
                    less ++;
                }
                else if (a[k] == pivot2) {
                    swap(a, k, great);
                    swap(b, k, great);
                    great --;
                    if (a[k] == pivot1) {
                        swap(a, k, less);
                        swap(b, k, less);
                        less ++;
                    }
                }
            }
        }

        if (pivot1 < pivot2) {
            dualPivotQuicksortTuple(a, b, less, great, div);
        }
    }

    public static void sortTuple(int[] a) {
        sortTuple(a, 0, a.length >> 1);
    }
    public static void sortTuple(int[] a, int fromIndex, int toIndex) {
        rangeCheck(a.length >> 1, fromIndex, toIndex);
        dualPivotQuicksortTuple(a, fromIndex, toIndex - 1, 3);
    }

    private static void swap2(int[] a, int i, int j) {
        int iidx = i << 1;
        int jidx = j << 1;
        int temp = a[iidx];
        int temp1 = a[iidx + 1];
        a[iidx] = a[jidx];
        a[iidx + 1] = a[jidx + 1];
        a[jidx] = temp;
        a[jidx + 1] = temp1;


    }

    private static void dualPivotQuicksortTuple(int[] a, int left, int right, int div) {
        int len = right - left;
        if (len < 27) {
            for (int i = left + 1; i <= right; i++) {
                for (int j = i; j > left && a[j << 1] < a[(j - 1) << 1]; j--) {
                    swap2(a, j, j - 1);
                }
            }
            return;
        }
        int third = len / div;
        int m1 = left + third;
        int m2 = right - third;
        if (m1 <= left) {
            m1 = left + 1;
        }
        if (m2 >= right) {
            m2 = right - 1;
        }
        if (a[m1 << 1] < a[m2 << 1]) {
            swap2(a, m1, left);
            swap2(a, m2, right);
        }
        else {
            swap2(a, m1, right);
            swap2(a, m2, left);
        }

        int pivot1 = a[left << 1];
        int pivot2 = a[right << 1];

        int less = left + 1;
        int great = right - 1;
        for (int k = less; k <= great; k++) {
            if (a[k << 1] < pivot1) {
                swap2(a, k, less++);
            }
            else if (a[k << 1] > pivot2) {
                while (k < great && a[great << 1] > pivot2) {
                    great--;
                }
                swap2(a, k, great--);
                if (a[k << 1] < pivot1) {
                    swap2(a, k, less++);
                }
            }
        }
        int dist = great - less;
        if (dist < 13) {
            div++;
        }
        swap2(a, less - 1, left);
        swap2(a, great + 1, right);

        dualPivotQuicksortTuple(a, left, less - 2, div);
        dualPivotQuicksortTuple(a, great + 2, right, div);

        if (dist > len - 13 && pivot1 != pivot2) {
            for (int k = less; k <= great; k++) {
                if (a[k << 1] == pivot1) {
                    swap2(a, k, less++);
                }
                else if (a[k << 1] == pivot2) {
                    swap2(a, k, great--);
                    if (a[k << 1] == pivot1) {
                        swap2(a, k, less++);
                    }
                }
            }
        }

        if (pivot1 < pivot2) {
            dualPivotQuicksortTuple(a, less, great, div);
        }
    }

    public static void sortTuple(long[] a, long[] b) {
        sortTuple(a, b, 0, a.length);
    }
    public static void sortTuple(long[] a, long[] b, int fromIndex, int toIndex) {
        rangeCheck(a.length, fromIndex, toIndex);
        rangeCheck(b.length, fromIndex, toIndex);
        dualPivotQuicksortTuple(a, b, fromIndex, toIndex - 1, 3);
    }

    private static void swap(long[] a, int i, int j) {
        long temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
    private static void dualPivotQuicksortTuple(long[] a, long[] b, int left,int right, int div) {
        int len = right - left;
        if (len < 27) {
            for (int i = left + 1; i <= right; i++) {
                for (int j = i; j > left && a[j] < a[j - 1]; j--) {
                    swap(a, j, j - 1);
                    swap(b, j, j - 1);
                }
            }
            return;
        }
        int third = len / div;
        int m1 = left + third;
        int m2 = right - third;
        if (m1 <= left) {
            m1 = left + 1;
        }
        if (m2 >= right) {
            m2 = right - 1;
        }
        if (a[m1] < a[m2]) {
            swap(a, m1, left);
            swap(a, m2, right);

            swap(b, m1, left);
            swap(b, m2, right);
        }
        else {
            swap(a, m1, right);
            swap(a, m2, left);

            swap(b, m1, right);
            swap(b, m2, left);
        }
        long pivot1 = a[left];
        long pivot2 = a[right];
        int less = left + 1;
        int great = right - 1;
        for (int k = less; k <= great; k++) {
            if (a[k] < pivot1) {
                swap(a, k, less);
                swap(b, k, less);
                less ++;
            }
            else if (a[k] > pivot2) {
                while (k < great && a[great] > pivot2) {
                    great--;
                }
                swap(a, k, great);
                swap(b, k, great);
                great --;
                if (a[k] < pivot1) {
                    swap(a, k, less);
                    swap(b, k, less);
                    less ++;
                }
            }
        }
        int dist = great - less;
        if (dist < 13) {
            div++;
        }
        swap(a, less - 1, left);
        swap(a, great + 1, right);

        swap(b, less - 1, left);
        swap(b, great + 1, right);

        dualPivotQuicksortTuple(a, b, left, less - 2, div);
        dualPivotQuicksortTuple(a, b, great + 2, right, div);

        if (dist > len - 13 && pivot1 != pivot2) {
            for (int k = less; k <= great; k++) {
                if (a[k] == pivot1) {
                    swap(a, k, less);
                    swap(b, k, less);
                    less ++;
                }
                else if (a[k] == pivot2) {
                    swap(a, k, great);
                    swap(b, k, great);
                    great --;
                    if (a[k] == pivot1) {
                        swap(a, k, less);
                        swap(b, k, less);
                        less ++;
                    }
                }
            }
        }

        if (pivot1 < pivot2) {
            dualPivotQuicksortTuple(a, b, less, great, div);
        }
    }

    public static void sortTuple(long[] a) {
        sortTuple(a, 0, a.length >> 1);
    }
    public static void sortTuple(long[] a, int fromIndex, int toIndex) {
        rangeCheck(a.length >> 1, fromIndex, toIndex);
        dualPivotQuicksortTuple(a, fromIndex, toIndex - 1, 3);
    }

    private static void swap2(long[] a, int i, int j) {
        int iidx = i << 1;
        int jidx = j << 1;
        long temp = a[iidx];
        long temp1 = a[iidx + 1];
        a[iidx] = a[jidx];
        a[iidx + 1] = a[jidx + 1];
        a[jidx] = temp;
        a[jidx + 1] = temp1;


    }

    private static void dualPivotQuicksortTuple(long[] a, int left, int right, int div) {
        int len = right - left;
        if (len < 27) {
            for (int i = left + 1; i <= right; i++) {
                for (int j = i; j > left && a[j << 1] < a[(j - 1) << 1]; j--) {
                    swap2(a, j, j - 1);
                }
            }
            return;
        }
        int third = len / div;
        int m1 = left + third;
        int m2 = right - third;
        if (m1 <= left) {
            m1 = left + 1;
        }
        if (m2 >= right) {
            m2 = right - 1;
        }
        if (a[m1 << 1] < a[m2 << 1]) {
            swap2(a, m1, left);
            swap2(a, m2, right);
        }
        else {
            swap2(a, m1, right);
            swap2(a, m2, left);
        }

        long pivot1 = a[left << 1];
        long pivot2 = a[right << 1];

        int less = left + 1;
        int great = right - 1;
        for (int k = less; k <= great; k++) {
            if (a[k << 1] < pivot1) {
                swap2(a, k, less++);
            }
            else if (a[k << 1] > pivot2) {
                while (k < great && a[great << 1] > pivot2) {
                    great--;
                }
                swap2(a, k, great--);
                if (a[k << 1] < pivot1) {
                    swap2(a, k, less++);
                }
            }
        }
        int dist = great - less;
        if (dist < 13) {
            div++;
        }
        swap2(a, less - 1, left);
        swap2(a, great + 1, right);

        dualPivotQuicksortTuple(a, left, less - 2, div);
        dualPivotQuicksortTuple(a, great + 2, right, div);

        if (dist > len - 13 && pivot1 != pivot2) {
            for (int k = less; k <= great; k++) {
                if (a[k << 1] == pivot1) {
                    swap2(a, k, less++);
                }
                else if (a[k << 1] == pivot2) {
                    swap2(a, k, great--);
                    if (a[k << 1] == pivot1) {
                        swap2(a, k, less++);
                    }
                }
            }
        }

        if (pivot1 < pivot2) {
            dualPivotQuicksortTuple(a, less, great, div);
        }
    }

    private static void check(float []a) {
        for (int i = 1; i < a.length; i++) {
            if (a[i] < a[i-1]) {
                System.out.println("errort sort");
            }
        }
    }

    private static void check2(float []a) {
        for (int i = 1; i < a.length / 2; i++) {
            if (a[i << 1] < a[(i-1) << 1]) {
                System.out.println("errort sort");
            }

            if (a[(i << 1) + 1] < a[((i-1) << 1) + 1]) {
                System.out.println("errort sort");
            }
        }
    }


    private static void check(double []a) {
        for (int i = 1; i < a.length; i++) {
            if (a[i] < a[i-1]) {
                System.out.println("errort sort");
            }
        }
    }

    private static void check2(double []a) {
        for (int i = 1; i < a.length / 2; i++) {
            if (a[i << 1] < a[(i-1) << 1]) {
                System.out.println("errort sort");
            }

            if (a[(i << 1) + 1] < a[((i-1) << 1) + 1]) {
                System.out.println("errort sort");
            }
        }
    }

    private static void check(int []a) {
        for (int i = 1; i < a.length; i++) {
            if (a[i] < a[i-1]) {
                System.out.println("errort sort");
            }
        }
    }

    private static void check2(int []a) {
        for (int i = 1; i < a.length / 2; i++) {
            if (a[i << 1] < a[(i-1) << 1]) {
                System.out.println("errort sort");
            }

            if (a[(i << 1) + 1] < a[((i-1) << 1) + 1]) {
                System.out.println("errort sort");
            }
        }
    }

    private static void check(long []a) {
        for (int i = 1; i < a.length; i++) {
            if (a[i] < a[i-1]) {
                System.out.println("errort sort");
            }
        }
    }

    private static void check2(long []a) {
        for (int i = 1; i < a.length / 2; i++) {
            if (a[i << 1] < a[(i-1) << 1]) {
                System.out.println("errort sort");
            }

            if (a[(i << 1) + 1] < a[((i-1) << 1) + 1]) {
                System.out.println("errort sort");
            }
        }
    }

    public static void main(String []args) {
        int len = 100000;

        // float
        float []af = new float[len];
        float []bf = new float[len];
        Random rand = new Random();
        for (int i = 0; i < len; i++) {
            float val = rand.nextFloat();
            af[i] = val;
            bf[i] = val;
        }

        long start = System.currentTimeMillis();
        sortTuple(af, bf);
        System.out.println("takes:" + (System.currentTimeMillis() - start));
        check(af);
        check(bf);

        float []a2f = new float[len * 2];
        for (int i = 0; i < len; i++) {
            float val = rand.nextFloat();
            a2f[i << 1] = val;
            a2f[(i << 1) + 1] = val + 2.0f;
        }
        start = System.currentTimeMillis();
        sortTuple(a2f);
        System.out.println("takes:" + (System.currentTimeMillis() - start));
        check2(a2f);


        // double
        double []ad = new double[len];
        double []bd = new double[len];
        for (int i = 0; i < len; i++) {
            double val = rand.nextFloat();
            ad[i] = val;
            bd[i] = val;
        }

        start = System.currentTimeMillis();
        sortTuple(ad, bd);
        System.out.println("takes:" + (System.currentTimeMillis() - start));
        check(ad);
        check(bd);

        double []a2d = new double[len * 2];
        for (int i = 0; i < len; i++) {
            double val = rand.nextFloat();
            a2d[i << 1] = val;
            a2d[(i << 1) + 1] = val + 2.0;
        }
        start = System.currentTimeMillis();
        sortTuple(a2d);
        System.out.println("takes:" + (System.currentTimeMillis() - start));
        check2(a2d);


        // int
        int []ai = new int[len];
        int []bi = new int[len];
        for (int i = 0; i < len; i++) {
            int val = rand.nextInt();
            ai[i] = val;
            bi[i] = val;
        }

        start = System.currentTimeMillis();
        sortTuple(ai, bi);
        System.out.println("takes:" + (System.currentTimeMillis() - start));
        check(ai);
        check(bi);

        int []a2i = new int[len * 2];
        for (int i = 0; i < len; i++) {
            int val = rand.nextInt();
            a2i[i << 1] = val;
            a2i[(i << 1) + 1] = val + 2;
        }
        start = System.currentTimeMillis();
        sortTuple(a2i);
        System.out.println("takes:" + (System.currentTimeMillis() - start));
        check2(a2i);


        // long
        long []al = new long[len];
        long []bl = new long[len];
        for (int i = 0; i < len; i++) {
            long val = rand.nextLong();
            al[i] = val;
            bl[i] = val;
        }

        start = System.currentTimeMillis();
        sortTuple(al, bl);
        System.out.println("takes:" + (System.currentTimeMillis() - start));
        check(al);
        check(bl);

        long []a2l = new long[len * 2];
        for (int i = 0; i < len; i++) {
            long val = rand.nextLong();
            a2l[i << 1] = val;
            a2l[(i << 1) + 1] = val + 2L;
        }
        start = System.currentTimeMillis();
        sortTuple(a2l);
        System.out.println("takes:" + (System.currentTimeMillis() - start));
        check2(a2l);


    }

}
