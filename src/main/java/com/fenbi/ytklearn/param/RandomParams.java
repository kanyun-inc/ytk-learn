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

/**
 * @author xialong
 */

@Data
public class RandomParams implements Serializable {
    public static final String KEY = "random.";
    public Mode mode;
    public static enum Mode {
        NORMAL("normal"),
        UNIFORM("uniform"),
        UNKNOWN("unknown");

        private String name;
        public String toString() {
            return name;
        }
        private Mode(String name) {
            this.name = name;
        }
        public static Mode getMode(String name) {
            for (Mode mode : values()) {
                if (name.equalsIgnoreCase(mode.toString())) {
                    return mode;
                }
            }
            return UNKNOWN;
        }

        public static String allToString() {
            Mode []validModels = new Mode[values().length - 1];
            for (int i = 0; i < validModels.length; i++) {
                validModels[i] = values()[i];
            }
            return Arrays.toString(validModels);
        }
    }

    public int seed;

    public Normal normal;
    @Data
    public static class Normal {
        public static final String KEY = "normal.";
        public double mean;
        public double std;
        public Normal(Config config, String prefix) {
            mean = config.getDouble(prefix + KEY + "mean");
            std = config.getDouble(prefix + KEY + "std");
        }
    }

    public Uniform uniform;
    @Data
    public static class Uniform {
        public static final String KEY = "uniform.";
        public double range_start;
        public double range_end;
        public Uniform(Config config, String prefix) {
            range_start = config.getDouble(prefix + KEY + "range_start");
            range_end = config.getDouble(prefix + KEY + "range_end");
        }
    }

    public RandomParams(Config config, String prefix) {
        mode = Mode.getMode(config.getString(prefix + KEY + "mode"));
        seed = config.getInt(prefix + KEY + "seed");
        normal = new Normal(config, prefix + KEY);
        uniform = new Uniform(config, prefix + KEY);

        CheckUtils.check(mode != Mode.UNKNOWN, "unknown %smode:%s, only support:%s",
                prefix + KEY, mode, Mode.allToString());
    }

    public static void main(String []args) {
        Config config = ConfigFactory.parseFile(new File("config/model/fm.conf"));
        RandomParams randomParams = new RandomParams(config, "");
        System.out.println(randomParams);
    }
}
