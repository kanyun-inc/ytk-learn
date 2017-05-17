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

/**
 * @author wufan
 */

public class BinarySearch {

    // binary search, return the index of last element that equal or large than target
    public static int findLastEqualOrUpper(float arr[], float target) {
        CheckUtils.check(arr != null && arr.length >= 1, "[GBDT] find last equal or upper error, arr is null or empty");
        if (target > arr[arr.length -1]) {
            return -1;
        }

        int l = 0;
        int u = arr.length - 1;
        while (l <= u) {
            int mid = l + (u-l)/2;
            if (target >= arr[mid]) {
                l = mid + 1;
            } else {
                u = mid - 1;
            }
        }

        u = Math.max(0, u);  // arr[u] <= target, arr[l] > target
        if (arr[u] == target) {
            return u;
        } else {
            return Math.min(arr.length - 1, l);
        }
    }

    // binary search, return the index of first element that equal or large than target
    public static int findFirstEqualOrUpper(double arr[], double target) {
        CheckUtils.check(arr != null && arr.length >= 1, "[GBDT] find first equal or upper error, arr is null or empty");
        if (target > arr[arr.length -1]) {
            return -1;
        }

        int l = 0;
        int u = arr.length - 1;
        while (l <= u) {
            int mid = l + (u-l)/2;
            if (target <= arr[mid]) {
                u = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return Math.min(arr.length - 1, l);  // arr[l] >= target
    }

    // two number in the arr is treated as an element, key is the second number
    // return the index of first element that equal or large than target
    public static int findFirstEqualOrUpperTuple(double arr[], double target) {
        CheckUtils.check(arr != null && arr.length >= 1, "[GBDT] findFirstEqualOrUpperTuple error, arr is null or empty");
//        CheckUtils.check(target <= arr[arr.length - 1], "[GBDT] findFirstEqualOrUpperTuple error, cant't find element that is no smaller than target(%f)", target);
        if (target > arr[arr.length -1]) {
            return -1;
        }

        int l = 0;
        int u = arr.length/2 - 1;
        while (l <= u) {
            int mid = l + (u-l)/2;
            int index = (mid << 1) + 1;
            if (target <= arr[index]) {
                u = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return Math.min(arr.length/2 - 1, l); // arr[l] >= target
    }


    public static void main(String[] args) {
        float[] arr = {0, 0, 1, 1, 1, 4, 5, 6, 6, 6, 32};
        float[] target = {0, 0.5f, 1, 1.5f, 3, 4.5f, 6, 7};
        for (int i = 0; i < target.length; i++) {
            int index = findLastEqualOrUpper(arr, target[i]);
            System.out.println(target[i] + ", index:" + index + ", value:" + arr[index] );
//            index = findFirstEqualOrUpper(arr, target[i]);
            System.out.println(target[i] + ", index:" + index + ", value:" + arr[index] );
            System.out.println("==");
        }
    }

}
