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

import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.mp4j.exception.Mp4jException;

/**
 * @author xialong
 */

public class LogUtils {
    private final ThreadCommSlave threadCommSlave;
    private final boolean userDefinedVerbose;
    public LogUtils(ThreadCommSlave threadCommSlave, boolean userDefinedVerbose) {
        this.threadCommSlave = threadCommSlave;
        this.userDefinedVerbose = userDefinedVerbose;
    }

    public void importantInfo(String info, boolean onlyRank0Thread0) throws Mp4jException {
        threadCommSlave.info(info, onlyRank0Thread0);
    }

    public void importantInfo(String info) throws Mp4jException {
        threadCommSlave.info(info, true);
    }

    public void verboseInfo(String info, boolean onlyRank0Thread0) throws Mp4jException {
        if (userDefinedVerbose) {
            threadCommSlave.info(info, onlyRank0Thread0);
        }
    }

    public void verboseInfo(String info) throws Mp4jException {
        verboseInfo(info, true);
    }

    public void error(String error) throws Mp4jException {
        threadCommSlave.error(error);
    }

    public void exception(Exception exception) throws Mp4jException {
        threadCommSlave.exception(exception);
    }

}
