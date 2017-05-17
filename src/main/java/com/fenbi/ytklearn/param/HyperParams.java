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
public class HyperParams implements Serializable {
    public static final String KEY = "hyper.";
    public boolean switch_on;
    public boolean restart;

    public Mode mode;
    public static enum Mode {
        HOAG("hoag"),
        GRID("grid"),
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

    public Hoag hoag;
    @Data
    public static class Hoag implements Serializable{
        public static final String KEY = "hoag.";
        public double init_step;
        public double step_decr_factor;
        public double test_loss_reduce_limit;
        public int outer_iter;
        public double []l1;
        public double []l2;

        public Hoag(Config config, String prefix) {
            init_step = config.getDouble(prefix + KEY + "init_step");
            step_decr_factor = config.getDouble(prefix + KEY + "step_decr_factor");
            test_loss_reduce_limit = config.getDouble(prefix + KEY + "test_loss_reduce_limit");
            outer_iter = config.getInt(prefix + KEY + "outer_iter");
            List<Double> l1List = config.getDoubleList(prefix + KEY + "l1");
            List<Double> l2List = config.getDoubleList(prefix + KEY + "l2");
            l1 = new double[l1List.size()];
            l2 = new double[l2List.size()];
            for (int i = 0; i < l2.length; i ++) {
                l1[i] = l1List.get(i);
                l2[i] = l2List.get(i);
            }

            CheckUtils.check(step_decr_factor < 1.0, "%sstep_decr_factor:%f must < 1.0",
                    prefix + KEY, step_decr_factor);
            CheckUtils.check(l1.length == l2.length,
                    "%sl1 lenght must be equal to %sl2 lenght",
                    prefix + KEY,
                    prefix + KEY);
        }
    }

    public Grid grid;
    @Data
    public static class Grid implements Serializable {
        public static final String KEY = "grid.";
        public double [][]l1;
        public double [][]l2;

        public Grid(Config config, String prefix) {
            List<Double> l1List = config.getDoubleList(prefix + KEY + "l1");
            List<Double> l2List = config.getDoubleList(prefix + KEY + "l2");

            CheckUtils.check(l1List.size() == l2List.size(),
                    "%sl1 length must be equal to %sl2 length",
                    prefix + KEY,
                    prefix + KEY);

            l1 = new double[l1List.size() / 3][3];
            l2 = new double[l2List.size() / 3][3];

            for (int i = 0; i < l1.length; i++) {
                for (int j = 0; j < 3; j++) {
                    l1[i][j] = l1List.get(i * 3 + j);
                    l2[i][j] = l2List.get(i * 3 + j);
                }
            }


        }
    }

    public HyperParams(Config config, String prefix) {
        switch_on = config.getBoolean(prefix + KEY + "switch_on");
        restart = config.getBoolean(prefix + KEY + "restart");
        mode = Mode.getMode(config.getString(prefix + KEY + "mode"));
        hoag = new Hoag(config, prefix + KEY);
        grid = new Grid(config, prefix + KEY);

        CheckUtils.check(mode != Mode.UNKNOWN, "unknown %smode:%s, only support:%s",
                prefix + KEY, mode, Mode.allToString());
    }

    public static void main(String []args) {
        Config config = ConfigFactory.parseFile(new File("config/model/linear.conf"));
        HyperParams params = new HyperParams(config, "");
        System.out.println(params.toString());
    }
}
