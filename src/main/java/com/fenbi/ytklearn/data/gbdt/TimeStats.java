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

public class TimeStats {
    public long totalTime;
    public long buildHist;
    public long initStats;
    public long findBestSplit;
    public long syncBestSplit;
    // compute, communicate cost
    public long[] buildDetail;

    public TimeStats() {
        totalTime = 0L;
        buildHist = 0L;
        initStats = 0L;
        findBestSplit = 0L;
        syncBestSplit = 0L;
        buildDetail = new long[]{0, 0};
    }

    public void clear() {
        totalTime = 0L;
        buildHist = 0L;
        initStats = 0L;
        findBestSplit = 0L;
        syncBestSplit = 0L;
        buildDetail[0] = 0;
        buildDetail[1] = 0;
    }

    public void add(TimeStats stats) {
        totalTime += stats.totalTime;
        buildHist += stats.buildHist;
        initStats += stats.initStats;
        findBestSplit += stats.findBestSplit;
        syncBestSplit += stats.syncBestSplit;
        for (int i = 0; i < buildDetail.length; i++) {
            buildDetail[i] += stats.buildDetail[i];
        }
    }

    public String getStats() {
       return String.format("\nTreeMaker cost details\n%-16s%9.2fs(%6.2f%%)\n%-16s%9.2fs(%6.2f%%)\n%-16s%9.2fs(%6.2f%%)\n%-16s%9.2fs(%6.2f%%)\n%-16s%9.2fs(%6.2f%%)\n%-16s%9.2fs(%6.2f%%)\n%-16s%9.2fs\n",
                "BuildHist:", buildHist / 1000., buildHist * 100. / totalTime,
                "computeHist:", buildDetail[0] / 1000., buildDetail[0] * 100. / totalTime,
                "communicateHist:", buildDetail[1] / 1000., buildDetail[1] * 100. / totalTime,
                "InitStats:", initStats / 1000., initStats * 100. / totalTime,
                "FindBestSplit:", findBestSplit / 1000., findBestSplit * 100. / totalTime,
                "SyncBestSplit:", syncBestSplit / 1000., syncBestSplit * 100. / totalTime,
                "Total:", totalTime / 1000.);
    }
}
