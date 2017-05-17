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
import com.typesafe.config.ConfigValue;
import com.typesafe.config.ConfigValueFactory;
import lombok.Data;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * @author xialong
 */

@Data
public class LineSearchParams implements Serializable {
    public static final String KEY = "line_search.";

    public Mode mode;
    public static enum Mode {
        SUFFICIENT_DECREASE("sufficient_decrease"),
        WOLFE("wolfe"),
        STRONG_WOLFE("strong_wolfe"),
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

    public BackTracking backtracking;
    @Data
    public static class BackTracking {
        public static final String KEY = "backtracking.";
        public double step_decr;
        public double step_incr;
        public int max_iter;
        public double min_step;
        public double max_step;
        public double c1;
        public double c2;

        public BackTracking(Config config, String prefix) {
            step_decr = config.getDouble(prefix + KEY + "step_decr");
            step_incr = config.getDouble(prefix + KEY + "step_incr");
            max_iter = config.getInt(prefix + KEY + "max_iter");
            min_step = config.getDouble(prefix + KEY + "min_step");
            max_step = config.getDouble(prefix + KEY + "max_step");
            c1 = config.getDouble(prefix + KEY + "c1");
            c2 = config.getDouble(prefix + KEY + "c2");

            CheckUtils.check(step_decr < 1.0, "%sstep_decr:%f must < 1.0", prefix + KEY, step_decr);
            CheckUtils.check(step_incr > 1.0, "%sstep_incr:%f must > 1.0", prefix + KEY, step_incr);

            CheckUtils.check(c1 > 0.0 && c1 < 1.0, "%sc1:%f must be in range(0, 1)", prefix + KEY, c1);
            CheckUtils.check(c2 > c1 && c1 < 1.0, "%sc2:%f must be in range(c1, 1)", prefix + KEY, c2);
        }
    }

    public LBFGSParams lbfgsParams;
    @Data
    public static class LBFGSParams implements Serializable {
        public static final String KEY = "lbfgs.";

        public int m;

        public Convergence convergence;
        @Data
        public static class Convergence implements Serializable {
            public static final String KEY = "convergence.";
            public int max_iter;
            public double eps;
            public Convergence(Config config, String prefix) {
                max_iter = config.getInt(prefix + KEY + "max_iter");
                eps = config.getDouble(prefix + KEY + "eps");
            }
        }

        public LBFGSParams(Config config, String prefix) {
            m = config.getInt(prefix + KEY + "m");
            convergence = new Convergence(config, prefix + KEY);
        }
    }

    public LineSearchParams(Config config, String prefix) {
        mode = Mode.getMode(config.getString(prefix + KEY + "mode"));
        backtracking = new BackTracking(config, prefix + KEY);
        lbfgsParams = new LBFGSParams(config, prefix + KEY);

        CheckUtils.check(mode != Mode.UNKNOWN, "unknown %smode:%s, only support:%s",
                prefix + KEY, config.getString(prefix + KEY + "mode"), Mode.allToString());
    }

    public static void main(String []args) throws Exception {
        Config config = ConfigFactory.parseFile(new File("config/model/linear.conf"));
        LineSearchParams params = new LineSearchParams(config, "optimization.");
        System.out.println(params.toString());

        Map<String, Object> map = new HashMap<>();
//        for (Map.Entry<String, ConfigValue> entry : config.entrySet()) {
//            System.out.println(entry.getKey() + " ---- " + entry.getValue() + " ---- " + entry.getValue().getClass());
//            map.put(entry.getKey(), entry.getValue());
//        }
//
        System.out.println("---------------------" + config.entrySet().size());


        map.put("data.train.data_path", "path");
        map.put("loss.regularization.l1", Arrays.asList(1.23, 4.56));
        map.put("hyper.hoag.init_step", 45);
        map.put("optimization.line_search.backtracking.max_step", 0.34);

        System.out.println(map);

        for (Map.Entry<String, Object> entry : map.entrySet()) {
            config = config.withValue(entry.getKey(), ConfigValueFactory.fromAnyRef(entry.getValue()));
        }

        //Config configfrommap = ConfigFactory.parseMap(map);
        for (Map.Entry<String, ConfigValue> entry : config.entrySet()) {
            System.out.println(entry.getKey() + " ---- " + entry.getValue());
            map.put(entry.getKey(), entry.getValue());
        }

        System.out.println("---------------------" + config.entrySet().size());

    }


}
