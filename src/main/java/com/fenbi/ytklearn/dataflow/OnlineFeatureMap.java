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

package com.fenbi.ytklearn.dataflow;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author xialong
 */

public class OnlineFeatureMap implements IFeatureMap {
    public static final int DEFAULT_MAP_SIZE = 100000;
    private final int lockNum;
    private final Map<String, Integer> name2Idxs[];
    private final AtomicInteger count;

    public OnlineFeatureMap(int lockNum) {
        this.lockNum = lockNum;
        this.count = new AtomicInteger(0);

        this.name2Idxs = new HashMap[lockNum];
        for (int t = 0; t < lockNum; t++) {
            this.name2Idxs[t] = new HashMap<>(DEFAULT_MAP_SIZE);
        }

    }

    private int getNameHashCode(String name) {
        int hash = name.hashCode() % lockNum;
        if (hash < 0) {
            hash += lockNum;
        }
        return hash;
    }


    public Integer getIndex(String name) {
        int hash = getNameHashCode(name);

        synchronized (name2Idxs[hash]) {
            Integer idx = name2Idxs[hash].get(name);
            if (idx == null) {
                idx = count.getAndAdd(1);
                name2Idxs[hash].put(name, idx);
            }

            return idx;
        }
    }

    public Set<String> getFeatureSet() {
        Set<String> featureSet = new HashSet<>(count.get() + 1);
        for (int i = 0; i < lockNum; i++) {
            synchronized (name2Idxs[i]) {
                for (Map.Entry<String, Integer> entry : name2Idxs[i].entrySet()) {
                    featureSet.add(entry.getKey());
                }
            }
        }
        return featureSet;
    }

    public Map<Integer, String> getIndex2NameMap(boolean needBias, String biasName) {
        Map<Integer, String> fIndex2NameMap = new HashMap<>(count.get() + 1);
        int delta = needBias ? 1 : 0;
        if (needBias) {
            fIndex2NameMap.put(0, biasName);
        }
        for (int i = 0; i < lockNum; i++) {
            synchronized (name2Idxs[i]) {
                for (Map.Entry<String, Integer> entry : name2Idxs[i].entrySet()) {
                    fIndex2NameMap.put(entry.getValue() + delta, entry.getKey());
                }
            }
        }
        return fIndex2NameMap;
    }

    public Map<String, Integer> getName2IndexMap() {
        Map<String, Integer> fName2IndexMap = new HashMap<>(count.get() + 1);
        for (int i = 0; i < lockNum; i++) {
            synchronized (name2Idxs[i]) {
                for (Map.Entry<String, Integer> entry : name2Idxs[i].entrySet()) {
                    fName2IndexMap.put(entry.getKey(), entry.getValue());
                }
            }
        }
        return fName2IndexMap;
    }

    public void release() {
        for (int i = 0; i < lockNum; i++) {
            synchronized (name2Idxs[i]) {
                name2Idxs[i].clear();
            }
        }
    }
}
