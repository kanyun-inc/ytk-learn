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

import com.google.common.collect.Sets;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import lombok.Data;

import java.io.File;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Set;

/**
 * @author xialong
 */

@Data
public class TransformParams implements Serializable {
    public static final String KEY = "transform.";

    public boolean switch_on;
    public TransformParams.Mode mode;
    public enum Mode {
        STANDARDIZATION("standardization"),
        SCALE_RANGE("scale_range"),
        UNKNOWN("unknown");

        private String name;
        public String toString() {
            return name;
        }
        private Mode(String name) {
            this.name = name;
        }
        public static TransformParams.Mode getMode(String name) {
            for (TransformParams.Mode mode : values()) {
                if (name.equalsIgnoreCase(mode.toString())) {
                    return mode;
                }
            }
            return UNKNOWN;
        }

        public static String allToString() {
            TransformParams.Mode[]validModels = new TransformParams.Mode[values().length - 1];
            for (int i = 0; i < validModels.length; i++) {
                validModels[i] = values()[i];
            }
            return Arrays.toString(validModels);
        }
    }

    public ScaleRange scale_range;
    @Data
    public static class ScaleRange {
        public static final String KEY = "scale_range.";

        public double min;
        public double max;
        public ScaleRange(Config config, String prefix) {
            min = config.getDouble(prefix + KEY + "min");
            max = config.getDouble(prefix + KEY + "max");
        }
    }

    public Set<String> include;
    public Set<String> exclude;
    public TransformParams(Config config, String prefix) {
        switch_on = config.getBoolean(prefix + KEY + "switch_on");
        mode = Mode.getMode(config.getString(prefix + KEY + "mode"));
        scale_range = new ScaleRange(config, prefix + KEY);
        include = Sets.newHashSet(config.getStringList(prefix + KEY + "include_features"));
        exclude = Sets.newHashSet(config.getStringList(prefix + KEY + "exclude_features"));
    }

    public static void main(String []args) {
        Config config = ConfigFactory.parseFile(new File("config/model/linear.conf"));
        TransformParams params = new TransformParams(config, "feature.");
        System.out.println(params.toString());
    }

}
