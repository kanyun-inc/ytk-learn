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

package com.fenbi.ytklearn.feature.gbdt.approximate.sampler;

/**
 * @author wufan
 * @author xialong
 */

public enum SampleType {

    CNT("sample_by_cnt"),
    PRECISION("sample_by_precision"),
    RATE("sample_by_rate"),
    QUANTILE("sample_by_quantile"),
    NO_SAMPLE("no_sample");

    private String type;
    private SampleType(String type) {
        this.type = type;
    }

    public String getName() {
        return type;
    }

    public static SampleType valueOfType(String type) {
        for (SampleType mode: values()) {
            if (mode.getName().equalsIgnoreCase(type)) {
                return mode;
            }
        }
        return null;
    }

    public static SampleType getDefault() {
        return SampleType.NO_SAMPLE;
    }
}
