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

import com.fenbi.ytklearn.eval.EvaluatorFactory;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import lombok.Data;

import java.io.File;
import java.io.Serializable;
import java.util.List;

/**
 * @author xialong
 */

@Data
public class LossParams implements Serializable {
    public static final String KEY = "loss.";
    public String loss_function;
    public List<String> evaluate_metric;
    public boolean just_evaluate;

    public Regularization regularization;
    @Data
    public static class Regularization implements Serializable {
        public static final String KEY = "regularization.";
        public double l1[];
        public double l2[];

        public Regularization(Config config, String prefix) {
            List<Double> l1List = config.getDoubleList(prefix + KEY + "l1");
            l1 = new double[l1List.size()];
            for (int i = 0; i < l1List.size(); i++) {
                l1[i] = l1List.get(i);
            }
            List<Double> l2List = config.getDoubleList(prefix + KEY + "l2");
            l2 = new double[l2List.size()];
            for (int i = 0; i < l2List.size(); i++) {
                l2[i] = l2List.get(i);
            }

            CheckUtils.check(l1.length == l2.length,
                    "%sl1 lenght must be equal to %sl2 lenght",
                    prefix + KEY,
                    prefix + KEY);

        }
    }

    public LossParams(Config config, String prefix) {
        loss_function = config.getString(prefix + KEY + "loss_function");
        regularization = new Regularization(config, prefix + KEY);
        evaluate_metric = config.getStringList(prefix + KEY + "evaluate_metric");
        just_evaluate = config.getBoolean(prefix + KEY + "just_evaluate");

        for (String metric : evaluate_metric) {
            if (!EvaluatorFactory.EvalNameSet.contains(metric)) {
                boolean hit = false;
                for (String eval : EvaluatorFactory.EvalNameSet) {
                    if (metric.startsWith(eval)) {
                        hit = true;
                    }
                }
                CheckUtils.check(hit, "%sevaluate_metric:%s, only support:%s",
                        prefix + KEY, metric, EvaluatorFactory.EvalNameSet);
            }
        }

    }


    public static void main(String []args) {
        Config config = ConfigFactory.parseFile(new File("config/model/linear.conf"));
        LossParams params = new LossParams(config, "");
        System.out.println(params.toString());
    }



}
