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

import com.fenbi.ytklearn.exception.YtkLearnException;

/**
 * @author wufan
 */

public class NumConvertUtils {

    public static float int2float(int v) {
        return Float.intBitsToFloat(v);
    }

    public static int float2int(float v) {
        return Float.floatToRawIntBits(v);
    }

    public static float parseFloat(String s) {
        float f = Float.parseFloat(s);
        if (f == Float.NEGATIVE_INFINITY || f == Float.POSITIVE_INFINITY || f == Float.NaN) {
            throw new YtkLearnException("data format error, feature value should not be -Infinity, Infinity or NaN");
        }
        return f;
    }

    public static void main(String[] args) {
        for (int i = -Integer.MAX_VALUE; i < Integer.MAX_VALUE; i++) {
            if (i != float2int(int2float(i))) {
                System.out.println("ERROR!");
            }
        }

        for (float f = -Float.MAX_VALUE; f < Float.MAX_VALUE; f += 99999999.f) {
            if (f != int2float(float2int(f))) {
                System.out.println("ERROR!");
            }
        }
    }
}
