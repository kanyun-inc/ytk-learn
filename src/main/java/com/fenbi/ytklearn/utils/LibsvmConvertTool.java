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

import com.fenbi.ytklearn.fs.FileSystemFactory;
import com.fenbi.ytklearn.fs.IFileSystem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URI;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * @author xialong
 */

public class LibsvmConvertTool {
    public static final Logger LOG = LoggerFactory.getLogger(LibsvmConvertTool.class);

    private static String createKLabelStr(int k, int label, String y_delim) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < k; i++) {
            String target = i != label ? "0" : "1";
            sb.append(target);
            if (i <  k - 1) {
                sb.append(y_delim);
            }
        }
        return sb.toString();
    }


    public static void main(String []args) {
        // binary_classification@label1,label2, multi_classification@label1,label2,..., regression
        String mode = args[0];

        String x_delim = args[1];
        String y_delim = args[2];
        String features_delim = args[3];
        String feature_name_val_delim = args[4];

        String fs_scheme = args[5];
        String inputPath = args[6];
        String outputPath = args[7];

        BufferedReader reader = null;
        PrintWriter writer = null;
        int cnt = 0;
        String line = "";

        int k = 2;
        Map<String, Integer> kLabel2IndexMap = new HashMap<>();
        if (mode.contains("classification")) {
            String []labelinfo = mode.split("@")[1].trim().split(",");
            k = labelinfo.length;
            for (int i = 0; i < k; i++) {
                kLabel2IndexMap.put(labelinfo[i], i);
            }
        }

        int [] kcnt = new int[k];
        for (int i = 0; i < k; i++) {
            kcnt[i] = 0;
        }


        try {
            IFileSystem fs = FileSystemFactory.createFileSystem(new URI(fs_scheme));
            reader = new BufferedReader(fs.getReader(inputPath));
            writer = new PrintWriter(fs.getWriter(outputPath));

            LOG.info("libsvm format data path:" + inputPath);
            while((line = reader.readLine()) != null) {
                StringBuilder sb = new StringBuilder();
                String []info = line.trim().split("\\s+");
                boolean hasLabel = info[0].split(":").length == 1;

                // weight 1.0
                sb.append("1").append(x_delim);

                // label
                if (hasLabel) {
                    if (mode.startsWith("binary_classification")) {
                        Integer label = kLabel2IndexMap.get(info[0]);
                        if (label == null) {
                            throw new Exception("unknown label:" + info[0]);
                        }
                        if (label < 0 || label >= k) {
                            throw new Exception("error libsvm format for mode:" + mode + " - " + line);
                        }

                        sb.append(label).append(x_delim);

                        kcnt[label] ++;
                    } else if (mode.startsWith("multi_classification")) {
                        Integer label = kLabel2IndexMap.get(info[0]);
                        if (label == null) {
                            throw new Exception("unknown label:" + info[0]);
                        }

                        if (label < 0 || label >= k) {
                            throw new Exception("error libsvm format for mode:" + mode + " - " + line);
                        }
                        sb.append(label).append(x_delim);

                        kcnt[label] ++;
                    } else if (mode.startsWith("regression")) {
                        float label = Float.parseFloat(info[0]);
                        sb.append("" + label).append(x_delim);
                    } else {
                        throw new Exception("unsupport mode:" + mode);
                    }
                } else {
                    sb.append(x_delim);
                }

                // features
                for (int i = 1; i < info.length; i++) {
                    String []kv = info[i].split(":");
                    sb.append(kv[0]).append(feature_name_val_delim).append(kv[1]);
                    if (i < info.length - 1) {
                        sb.append(features_delim);
                    }
                }

                writer.println(sb.toString());

                cnt++;
            }

            LOG.info("convert finished! convert count:" + cnt);

            if (mode.contains("classification")) {
                for (Map.Entry<String, Integer> entry : kLabel2IndexMap.entrySet()) {
                    LOG.info("libsvm classification label:" + entry.getKey() + " ----> ytklearn classification label:" + entry.getValue() + ", count:" + kcnt[entry.getValue()]);
                }
            }

            LOG.info("ytk-learn format data path:" + outputPath);


        } catch (Exception e) {
            LOG.error("error", e);
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                    LOG.error("error libsvm format for mode:" + mode + " - " + line , e1);
                }
            }

            if (writer != null) {
                writer.close();
            }
        }



    }
}
