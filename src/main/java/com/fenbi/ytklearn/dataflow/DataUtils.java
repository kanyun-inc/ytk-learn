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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.python.core.PyFunction;
import org.python.util.PythonInterpreter;

import java.io.*;
import java.net.URI;
import java.util.*;
import java.util.function.Function;

/**
 * @author xialong
 */

public class DataUtils {
    public static final Logger LOG = LoggerFactory.getLogger(DataUtils.class);

    public static class BufferedLineIterator implements Iterator<String> {
        BufferedReader reader;
        String line;
        public BufferedLineIterator(BufferedReader reader) {
            this.reader = reader;
        }

        @Override
        public boolean hasNext() {
            try {
                line = reader.readLine();
            } catch (IOException e) {
                LOG.error("read data error!");
                e.printStackTrace();
                close();
                return false;
            }

            if (line == null) {
                close();
            }
            return line != null;
        }

        @Override
        public String next() {
            return line;
        }

        private void close() {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    LOG.error("reader close error!");
                    e.printStackTrace();
                }
            }

        }
    }

    public static class SelectLineIterator implements Iterator<String> {
        BufferedReader reader;
        String line;
        int divisor;
        long readedLines = 0;
        int remainder;
        public SelectLineIterator(BufferedReader reader, int divisor, int remainder) {
            this.reader = reader;
            this.divisor = divisor;
            this.remainder = remainder;
        }

        @Override
        public boolean hasNext() {
            try {
                for (;;) {
                    line = reader.readLine();
                    readedLines ++;
                    if (((readedLines - 1) % divisor) == remainder) {
                        break;
                    }
                }
            } catch (IOException e) {
                LOG.error("read data error!");
                e.printStackTrace();
                close();
                return false;
            }

            if (line == null) {
                close();
            }
            return line != null;
        }

        @Override
        public String next() {
            return line;
        }

        private void close() {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    LOG.error("reader close error!");
                    e.printStackTrace();
                }
            }

        }
    }



    public static PyFunction getTranformFunction(boolean needPyTransform, String pyTransformScript) {
        PythonInterpreter interpreter;
        PyFunction transformFunc = null;
        if (needPyTransform) {
            interpreter = new PythonInterpreter();
            interpreter.execfile(pyTransformScript);
            transformFunc = interpreter.get("transform", PyFunction.class);
        }

        return transformFunc;
    }

    public static boolean hdfsPathPrefixValid(String path) {
        if (!(path.startsWith("hdfs://") || path.startsWith("file:///"))) {
            return false;
        }
        return true;
    }

    public static boolean pathNotNull(String path) {
        return path != null;
    }

    public static FileSystem getHdfsFileSystem(String scheme) throws IOException {
//        int idx = path.indexOf("/", 7);
//        LOG.info("path:" + path + ", idx:" + idx);
//        if (idx < 0) {
//            idx = path.length();
//        }
//        LOG.info("path:" + path + ", idx:" + idx);
//        String fsURI = path.trim().startsWith("hdfs") ?
//                path.substring(0, idx) :
//                "file:///";
//        LOG.info("fsURI:" + fsURI);
        Configuration conf = new Configuration();
        conf.set("fs.hdfs.impl", org.apache.hadoop.hdfs.DistributedFileSystem.class.getName());
        conf.set("fs.file.impl", org.apache.hadoop.fs.LocalFileSystem.class.getName());
        conf.set("fs.s3.impl", org.apache.hadoop.fs.s3.S3FileSystem.class.getName());
        conf.set("fs.s3n.impl", org.apache.hadoop.fs.s3native.NativeS3FileSystem.class.getName());
//        conf.set("fs.hftp.impl", HftpFileSystem.class.getName());
//        conf.set("fs.hsftp.impl", HsftpFileSystem.class.getName());
//        conf.set("fs.webhdfs.impl", org.apache.hadoop.hdfs.web.WebHdfsFileSystem.class.getName());
//        conf.set("fs.ftp.impl", org.apache.hadoop.fs.ftp.FTPFileSystem.class.getName());
//        conf.set("fs.har.impl", org.apache.hadoop.fs.HarFileSystem.class.getName());
        return FileSystem.get(URI.create(scheme), conf);
    }

    public static boolean isHdfsProtocol(String path) {
        return path.trim().startsWith("hdfs");
    }

    public static int[][]avgAssign(int amount, int bins) {
        int assign[][] = new int[bins][2];
        int avg = amount / bins;
        int mod = amount % bins;
        int fidx = 0;
        for (int t = 0; t < bins; t++) {
            int tnum = t < mod ? avg + 1 : avg;
            assign[t][0] = fidx;
            assign[t][1] = fidx + tnum;
            fidx += tnum;
        }
        return assign;
    }

    public static <T, R> void travel(Function<T, R> func, List<Iterator<T>> iterators) {
        for (Iterator<T> it : iterators) {
            while(it.hasNext()) {
                func.apply(it.next());
            }
        }

    }

    public static void randomBitset(BitSet bitSet, int seed, double thre, int to) {
        Random rand = new Random(seed);
        bitSet.clear();
        for (int i = 0; i < to; i++) {
            if (rand.nextDouble() <= thre) {
                bitSet.set(i);
            }
        }
    }

    public static void randomBitset(BitSet bitSet, double thre, int to) {
        Random rand = new Random();
        bitSet.clear();
        for (int i = 0; i < to; i++) {
            if (rand.nextDouble() <= thre) {
                bitSet.set(i);
            }
        }
    }


    public static void main(String []args) throws IOException {

        //String path = "hdfs://f04/research_pub/ytk-learn/datasets/news20.binary";
//        String path = "hdfs://f04";
//        getHdfsFileSystem(path);

//        List<String> list = Arrays.asList("1", "2", "3");
//        Map<String, Integer> map = new HashMap<>();
//        travel(line -> map.put(line, map.size()), Arrays.asList(list.iterator()));
//        System.out.println(map);
    }


}
