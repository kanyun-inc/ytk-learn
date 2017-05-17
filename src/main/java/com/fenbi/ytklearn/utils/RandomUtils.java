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

import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

import java.util.Random;

/**
 * @author wufan
 */

public class RandomUtils {

    public static final Logger LOG = LoggerFactory.getLogger(RandomUtils.class);

    //return true with probability p, coin flip
    public static boolean sampleBinary(Random rand, float p) {
        return rand.nextDouble() < p;
    }

    // return num in [l, u], l<=u
    public static int randomInt(int l, int u) {
        Random r = new Random();
        return r.nextInt(u-l+1) + l;
    }

    // shuffle array, use seed to ensure random result is the same in each workers
    public static void shuffle(long seed, int[] data) {
        Random rand = new Random(seed);
        int npos;
        int tmp;
        for (int i = data.length - 1; i > 0; i--) {
            npos = rand.nextInt(i + 1);
            tmp = data[i];
            data[i] = data[npos];
            data[npos] = tmp;
        }
    }

    // sample out k num from n (n represent for {0, 1, 2, ..., n-1})
    public static boolean reservoidSample(int n, int k, int[] out) {
        if (out == null || out.length != k || n < k) {
            return false;
        }
        for (int i = 0; i < k; i++) {
            out[i] = i;
        }
        Random r = new Random();
        for (int j = k; j < n; j++) {
            int pos = r.nextInt(j + 1);
            if (pos < k) {
                out[pos] = j;
            }
        }
        return true;
    }

    public static boolean reservoidSample(int[] arr, int k, int[] out) {
        if (out == null || out.length != k || arr.length < k) {
            return false;
        }
        for (int i = 0; i < k; i++) {
            out[i] = arr[i];
        }
        Random r = new Random();
        for (int j = k; j < arr.length; j++) {
            int pos = r.nextInt(j + 1);
            if (pos < k) {
                out[pos] = arr[j];
            }
        }
        return true;
    }
}
