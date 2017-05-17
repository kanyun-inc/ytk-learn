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

package com.fenbi.ytklearn.fs;

import java.io.IOException;
import java.net.URI;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * @author xialong
 */

public class FileSystemFactory {
//    public static IFileSystem createFileSystem(String name) throws IOException {
//
//        if (name.equalsIgnoreCase("local")) {
//            return new HdfsLineFileSystem("file:///");
//        } else if (name.equalsIgnoreCase("hdfs")) {
//            return new HdfsLineFileSystem("hdfs://");
//        } else {
//            throw new IOException("unknown file system");
//        }
//    }

    public final static Set<String> HDFS_FILE_SYSTEM_SET = new HashSet<>();
    static {
        HDFS_FILE_SYSTEM_SET.addAll(Arrays.asList("file", "hdfs", "s3", "s3n",
                "kfs", "hftp", "hsftp", "webhdfs", "ftp", "ramfs", "har"));
    };

    public static IFileSystem createFileSystem(URI uri) throws IOException {

        if (HDFS_FILE_SYSTEM_SET.contains(uri.toString().split(":")[0])) {
            return new HdfsFileSystem(uri.toString());
        } else if (uri.toString().equalsIgnoreCase("local")) {
            return new LocalFileSystem();
        } else {
            throw new IOException("unknown file system uri:" + uri);
        }
    }

}
