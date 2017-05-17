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

package com.fenbi.ytklearn.param;

import com.fenbi.ytklearn.utils.CheckUtils;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import lombok.Data;

import java.io.File;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * @author xialong
 */

@Data
public class DataParams implements Serializable {
    public static final String KEY = "data.";

    public Train train;
    @Data
    public static class Train implements Serializable {
        public static final String KEY = "train.";
        public String data_path;
        public int max_error_tol;
        public Train(Config config, String prefix) {
            data_path = config.getString(prefix + KEY + "data_path");
            max_error_tol = config.getInt(prefix + KEY + "max_error_tol");
        }
    }

    public Test test;
    @Data
    public static class Test implements Serializable {
        public static final String KEY = "test.";
        public String data_path;
        //public String data_temp_path;
        public int max_error_tol;
        public Test(Config config, String prefix) {
            data_path = config.getString(prefix + KEY + "data_path");
            //data_temp_path = config.getString(prefix + KEY + "data_temp_path");
            max_error_tol = config.getInt(prefix + KEY + "max_error_tol");
        }
    }

    public Delim delim;
    @Data
    public static class Delim implements Serializable {
        public static final String KEY = "delim.";
        public String x_delim;
        public String y_delim;
        public String features_delim;
        public String feature_name_val_delim;

        public Delim(Config config, String prefix) {
            x_delim = config.getString(prefix + KEY + "x_delim");
            y_delim = config.getString(prefix + KEY + "y_delim");
            features_delim = config.getString(prefix + KEY + "features_delim");
            feature_name_val_delim = config.getString(prefix + KEY + "feature_name_val_delim");

            CheckUtils.check(!x_delim.equals(y_delim),
                    "%sx_delim:%s must be different with %sy_delim:%s!",
                    prefix + KEY, x_delim,
                    prefix + KEY, y_delim);
            CheckUtils.check(!x_delim.equals(features_delim),
                    "%sx_delim:%s must be different with %sfeatures_delim:%s!",
                    prefix + KEY, x_delim,
                    prefix + KEY, features_delim);
            CheckUtils.check(!x_delim.equals(feature_name_val_delim),
                    "%sx_delim:%s must be different with %sfeature_name_val_delim:%s!",
                    prefix + KEY, x_delim,
                    prefix + KEY, feature_name_val_delim);
            CheckUtils.check(!features_delim.equals(feature_name_val_delim),
                    "%sfeatures_delim:%s must be different with %sfeature_name_val_delim:%s!",
                    prefix + KEY, features_delim,
                    prefix + KEY, feature_name_val_delim);
        }

    }

    public List<String> y_sampling;
    public boolean assigned;

    public UnassignedMode unassigned_mode;
    public static enum UnassignedMode {
        FILES_AVG("files_avg"),
        LINES_AVG("lines_avg"),
        UNKNOWN("unknown");

        private String name;
        public String toString() {
            return name;
        }
        private UnassignedMode(String name) {
            this.name = name;
        }
        public static UnassignedMode getMode(String name) {
            for (UnassignedMode mode : values()) {
                if (name.equalsIgnoreCase(mode.toString())) {
                    return mode;
                }
            }
            return UNKNOWN;
        }

        public static String allToString() {
            UnassignedMode []validModels = new UnassignedMode[values().length - 1];
            for (int i = 0; i < validModels.length; i++) {
                validModels[i] = values()[i];
            }
            return Arrays.toString(validModels);
        }
    }

    public DataParams(Config config, String prefix) {
        train = new Train(config, prefix + KEY);
        test = new Test(config, prefix + KEY);
        delim = new Delim(config, prefix + KEY);
        y_sampling = config.getStringList(prefix + KEY + "y_sampling");

        assigned = config.getBoolean(prefix + KEY + "assigned");
        unassigned_mode = UnassignedMode.getMode(config.getString(prefix + KEY + "unassigned_mode"));

        for (String s : y_sampling) {
            CheckUtils.check(s.matches("\\d+@(-?\\d+)(\\.\\d+)?"),
                    "%sy_sampling:%s must be the format of labelindex@rate. e.g 0@#0.1",
                    prefix + KEY, s);
        }

        CheckUtils.check(unassigned_mode != UnassignedMode.UNKNOWN,
                "unknown %sunassigned_mode:%s, only support:%s",
                prefix + KEY, unassigned_mode.toString(), UnassignedMode.allToString());

    }

    public static void main(String []args) {
        Config config = ConfigFactory.parseFile(new File("config/model/linear.conf"));
        DataParams params = new DataParams(config, "");
        System.out.println(params.toString());
    }

}
