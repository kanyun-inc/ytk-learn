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

package com.fenbi.ytklearn.feature.gbdt.missing;

import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.ytklearn.exception.YtkLearnException;

/**
 * factory class for computing missing value from train dataset
 * @author wufan
 * @author xialong
 */

public class ComputeFactory {

    // comm can be null which means it's local mode
    // type example: mean, quantile@0.5, value@0.45
    public static IComputer create(String type, ThreadCommSlave comm) throws Mp4jException {

        if (type.toLowerCase().equals(MissingValueProcessType.MEAN.getName())) {
            return new ComputeMean(comm);

        } else if (type.toLowerCase().startsWith(MissingValueProcessType.QUANTILE.getName())) {
            return new ComputeQuantile(comm, type);

        } else if (type.toLowerCase().startsWith(MissingValueProcessType.VALUE.getName())) {
            return new ComputeValue(comm, type);

        } else {
            throw new YtkLearnException("[GBDT] unknown missing value process type: " + type + "! you can use mean, quantile or value");
        }
    }
}
