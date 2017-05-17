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
 * @author xialong
 */

public class VectorUtils {
    
    public static void daxpy(double scalar, double x[], double y[]) {
        if (scalar == 0) return;

        assert x.length == y.length;
        for (int i = 0; i < x.length; i++) {
            y[i] += scalar * x[i];
        }
    }

    public static void daxpy(double scalar, float x[], float y[]) {
        if (scalar == 0) return;

        assert x.length == y.length;
        for (int i = 0; i < x.length; i++) {
            y[i] += scalar * x[i];
        }
    }

    public static void daxpy(double scalar, float x[], float y[], int start) {
        if (scalar == 0) return;

//        if (x.length != y.length) {
//            for (int i = 0; i < y.length; i++) {
//                if (!(i >= start && i < start + x.length)) {
//                    y[i] = 0.0f;
//                }
//            }
//        }

        for (int i = 0; i < x.length; i++) {
            y[start + i] += scalar * x[i];
        }
    }

    public static double dot(double x[], double y[]) {

        double product = 0;
        assert x.length == y.length;
        for (int i = 0; i < x.length; i++) {
            product += x[i] * y[i];
        }
        return product;

    }

    public static double dot(float x[], float y[]) {

        double product = 0;
        assert x.length == y.length;
        for (int i = 0; i < x.length; i++) {
            product += x[i] * y[i];
        }
        return product;

    }

    public static double dot(float x[], float y[], int start) {

        double product = 0;
        for (int i = 0; i < x.length; i++) {
            product += x[i] * y[start + i];
        }
        return product;

    }

    public static double euclideanNorm(double vector[]) {

        int n = vector.length;

        if (n < 1) {
            return 0;
        }

        if (n == 1) {
            return Math.abs(vector[0]);
        }


        double scale = 0;
        double sum = 1;
        for (int i = 0; i < n; i++) {
            if (vector[i] != 0) {
                double abs = Math.abs(vector[i]);
                if (scale < abs) {
                    double t = scale / abs;
                    sum = 1 + sum * (t * t);
                    scale = abs;
                } else {
                    double t = abs / scale;
                    sum += t * t;
                }
            }
        }

        return scale * Math.sqrt(sum);
    }

    public static double euclideanNorm(float vector[]) {

        int n = vector.length;

        if (n < 1) {
            return 0;
        }

        if (n == 1) {
            return Math.abs(vector[0]);
        }

        double scale = 0;
        double sum = 1;
        for (int i = 0; i < n; i++) {
            if (vector[i] != 0) {
                double abs = Math.abs(vector[i]);
                if (scale < abs) {
                    double t = scale / abs;
                    sum = 1 + sum * (t * t);
                    scale = abs;
                } else {
                    double t = abs / scale;
                    sum += t * t;
                }
            }
        }

        return scale * Math.sqrt(sum);
    }

    public static void scale(double scalar, double vector[]) {
        if (scalar == 1.0) return;
        for (int i = 0; i < vector.length; i++) {
            vector[i] *= scalar;
        }

    }

    public static void scale(double scalar, float vector[]) {
        if (scalar == 1.0) return;
        for (int i = 0; i < vector.length; i++) {
            vector[i] *= scalar;
        }

    }

    public static void vecdiff(float []z, float []x, float []y) {
        assert z.length == x.length;
        assert z.length == y.length;
        for (int i = 0; i < z.length; i++) {
            z[i] = x[i] - y[i];
        }
    }

    public static void vecdiff(float []z, float []x, float []y, int start) {
        for (int i = 0; i < z.length; i++) {
            z[i] = x[start + i] - y[start + i];
        }
    }

    public static void vecdiff(double []z, double []x, double []y) {
        assert z.length == x.length;
        assert z.length == y.length;
        for (int i = 0; i < z.length; i++) {
            z[i] = x[i] - y[i];
        }
    }

}

