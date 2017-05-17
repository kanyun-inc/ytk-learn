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

public class StringUtils {

    public static String join(int[] arr, String separator) {
        if (arr == null || arr.length == 0) {
            return "";
        } else {
            StringBuffer sb = new StringBuffer("");
            sb.append(arr[0]);
            for (int i = 1; i < arr.length; i++) {
                sb.append(separator);
                sb.append(arr[i]);
            }
            return sb.toString();
        }
    }

    public static String join(float[] arr, String separator) {
        if (arr == null || arr.length == 0) {
            return "";
        } else {
            StringBuffer sb = new StringBuffer("");
            sb.append(arr[0]);
            for (int i = 1; i < arr.length; i++) {
                sb.append(separator);
                sb.append(arr[i]);
            }
            return sb.toString();
        }
    }

    public static String join(double[] arr, String separator) {
        if (arr == null || arr.length == 0) {
            return "";
        } else {
            StringBuffer sb = new StringBuffer("");
            sb.append(arr[0]);
            for (int i = 1; i < arr.length; i++) {
                sb.append(separator);
                sb.append(arr[i]);
            }
            return sb.toString();
        }
    }

    public static void main(String[] args) {
        float[] a = new float[] {1.f, 2.f, 33.f};
        String s = StringUtils.join(a, ",");
        System.out.println("s:" + s);
        a = new float[0];
        System.out.println("s" + StringUtils.join(a, ","));
    }
}
