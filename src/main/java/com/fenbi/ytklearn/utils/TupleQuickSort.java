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

import java.util.List;

/**
 * @author wufan
 */

public class TupleQuickSort {

    // sort tuple by first element
    public static void quickSortElement(double[] arr) {
        quickSortElement(arr, 0, arr.length / 2 - 1);
    }

    private static void quickSortElement(double[] arr, int l, int u) {
        if (l >= u) {
            return;
        }

        int pivot = RandomUtils.randomInt(l, u);
        swapElement(arr, l, pivot);

        int i = l;
        int j = u + 1;
        double val = arr[l << 1];
        while (true) {
            do {
                i++;
            } while (i <= u && arr[i << 1] < val);

            do {
                j--;
            } while (arr[j << 1] > val);

            if (i > j) {
                break;
            }
            swapElement(arr, i, j);
        }
        swapElement(arr, l, j);
        quickSortElement(arr, l, j - 1);
        quickSortElement(arr, j + 1, u);
    }

    // swap sample and its tree node id
    private static void swapElement(double[] arr, int samIdx1, int samIdx2) {
        samIdx1 <<= 1;
        samIdx2 <<= 1;
        double tmp = arr[samIdx1];
        arr[samIdx1] = arr[samIdx2];
        arr[samIdx2] = tmp;

        tmp = arr[samIdx1 + 1];
        arr[samIdx1 + 1] = arr[samIdx2 + 1];
        arr[samIdx2 + 1] = tmp;
    }

    // sort tuple by first element
    public static void quickSortElement(float[] arr) {
        quickSortElement(arr, 0, arr.length / 2 - 1);
    }

    private static void quickSortElement(float[] arr, int l, int u) {
        if (l >= u) {
            return;
        }
        int pivot = RandomUtils.randomInt(l, u);
        swapElement(arr, l, pivot);

        int i = l;
        int j = u + 1;
        float val = arr[l << 1];
        while (true) {
            do {
                i++;
            } while (i <= u && arr[i << 1] < val);

            do {
                j--;
            } while (arr[j << 1] > val);

            if (i > j) {
                break;
            }
            swapElement(arr, i, j);
        }
        swapElement(arr, l, j);
        quickSortElement(arr, l, j - 1);
        quickSortElement(arr, j + 1, u);
    }

    // swap sample and its tree node id
    private static void swapElement(float[] arr, int samIdx1, int samIdx2) {
        samIdx1 <<= 1;
        samIdx2 <<= 1;
        float tmp = arr[samIdx1];
        arr[samIdx1] = arr[samIdx2];
        arr[samIdx2] = tmp;

        tmp = arr[samIdx1 + 1];
        arr[samIdx1 + 1] = arr[samIdx2 + 1];
        arr[samIdx2 + 1] = tmp;
    }

    // sort tuple by first element
    public static void quickSortElement(List<Float> arr) {
        quickSortElement(arr, 0, arr.size() / 2 - 1);
    }

    private static void quickSortElement(List<Float> arr, int l, int u) {
        if (l >= u) {
            return;
        }

        int pivot = RandomUtils.randomInt(l, u);
        swapElement(arr, l, pivot);

        int i = l;
        int j = u + 1;
        float val = arr.get(l << 1);
        while (true) {
            do {
                i++;
            } while (i <= u && arr.get(i << 1) < val);

            do {
                j--;
            } while (arr.get(j << 1) > val);

            if (i > j) {
                break;
            }
            swapElement(arr, i, j);
        }
        swapElement(arr, l, j);
        quickSortElement(arr, l, j - 1);
        quickSortElement(arr, j + 1, u);
    }

    // swap sample and its tree node id
    private static void swapElement(List<Float>  arr, int samIdx1, int samIdx2) {
        samIdx1 <<= 1;
        samIdx2 <<= 1;
        float tmp = arr.get(samIdx1);
        arr.set(samIdx1, arr.get(samIdx2));
        arr.set(samIdx2, tmp);

        tmp = arr.get(samIdx1 + 1);
        arr.set(samIdx1 + 1, arr.get(samIdx2 + 1));
        arr.set(samIdx2 + 1, tmp);
    }

}
