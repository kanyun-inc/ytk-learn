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

/**
 * @author wufan
 */

public class GradStats {

    double sumGrad;
    double sumHess;

    public GradStats() {
        sumGrad = sumHess = 0.0;
    }

    public void clear() {
        sumGrad = 0.0;
        sumHess = 0.0;
    }

    public void add(GradStats stats) {
        sumGrad += stats.sumGrad;
        sumHess += stats.sumHess;
    }

    public void set(double sumGrad, double sumHess) {
        this.sumGrad = sumGrad;
        this.sumHess = sumHess;
    }

    //for local version, add gradients of a sample
    public void add(float[] gradParis, int idx) {
        idx <<= 1;
        sumGrad += gradParis[idx];
        sumHess += gradParis[idx + 1];
    }

    //for distributed version, add gradients of a feature slot
    public void add(double[] gradParis, int idx) {
        idx <<= 1;
        sumGrad += gradParis[idx];
        sumHess += gradParis[idx + 1];
    }

    public void setSubstract(GradStats a, GradStats b) {
        sumGrad = a.sumGrad - b.sumGrad;
        sumHess = a.sumHess - b.sumHess;
    }

    public boolean empty() {
        return sumHess == 0.0;
    }

    public double getSumGrad() {
        return sumGrad;
    }

    public double getSumHess() {
        return sumHess;
    }
}
