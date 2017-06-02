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

package com.fenbi.ytklearn.param.gbdt;

import com.fenbi.ytklearn.data.Constants;
import com.fenbi.ytklearn.data.gbdt.TreeMakerType;
import com.fenbi.ytklearn.data.gbdt.TreeGrowPolicy;
import com.fenbi.ytklearn.dataflow.GBMLRDataFlow;
import com.fenbi.ytklearn.eval.EvaluatorFactory;
import com.fenbi.ytklearn.eval.IEvaluator;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigException;;
import lombok.Data;

import java.io.Serializable;
import java.util.List;

/**
 * @author wufan
 * @author xialong
 */

@Data
public class GBDTOptimizationParams implements Serializable {
    public static final String KEY = "optimization.";

    public GBMLRDataFlow.Type learn_type;
    public TreeMakerType tree_maker_type;
    public int round_num;
    public int max_depth;
    public double min_child_hessian_sum;
    public int max_leaf_cnt;
    public double min_split_loss;
    public int min_split_samples;

    public String objective;
    public double sigmoid_zmax;
    //max abs leaf value before multiplies learning rate
    public double max_abs_leaf_val;

    // whether to use quantile approximate for lad tree refine
    public boolean lad_refine_appr;
    public TreeGrowPolicy tree_grow_policy;
    public double histogram_pool_capacity;

    public Regularization regularization;

    public float uniform_base_prediction;
    public boolean sample_dependent_base_prediction;

    public float subsample;
    public float feature_sample_rate;

    public int class_num;

    public boolean just_evaluate;
    public List<String> eval_metrics;

    public boolean watch_train;
    public boolean watch_test;
//    public boolean first_order_approximate;
    public boolean verbose;

    @Data
    public static class Regularization implements Serializable {
        public static final String KEY = "regularization.";
        public double l1;
        public double l2;
        public float learningRate;

        public Regularization(Config config, String prefix) {
            l1 = config.getDouble(prefix + KEY + "l1");
            l2 = config.getDouble(prefix + KEY + "l2");
            learningRate = (float)config.getDouble(prefix + KEY + "learning_rate");
        }
    }

    public GBDTOptimizationParams(Config config, String prefix) throws Exception {
        round_num = config.getInt(prefix + KEY + "round_num");
        min_child_hessian_sum = config.getDouble(prefix + KEY + "min_child_hessian_sum");
        // max_depth < 0 means no limit, max_depth = 0 means only a root
        max_depth = config.getInt(prefix + KEY + "max_depth");
        // max_leaf_cnt < 0 means no limit, else max_leaf_cnt should be >= 1
        max_leaf_cnt = config.getInt(prefix + KEY + "max_leaf_cnt");
        objective = config.getString(prefix + KEY + "loss_function");
        try {
            min_split_loss = config.getDouble(prefix + KEY + "min_split_loss");
        } catch (ConfigException.Missing e) {
            min_split_loss = Constants.MIN_LOSS_CHG;
        }
        try {
            min_split_samples = config.getInt(prefix + KEY + "min_split_samples");
        } catch (ConfigException.Missing e) {
            min_split_samples = -1;
        }
        try {
            max_abs_leaf_val = config.getDouble(prefix + KEY + "max_abs_leaf_val");
        } catch (ConfigException.Missing e) {
            max_abs_leaf_val = 0;
        }

        regularization = new Regularization(config, prefix + KEY);

        //default value
        try {
            String typeStr = config.getString("type");
            learn_type = GBMLRDataFlow.Type.getType(typeStr);
            CheckUtils.check(learn_type != null, "[GBDT] learn type(%s) invalid", typeStr);
        } catch (ConfigException.Missing e) {
            learn_type = GBMLRDataFlow.Type.GB;
        }
        if (learn_type == GBMLRDataFlow.Type.RF) {
            regularization.learningRate = 1.0f;
        }

        // config for data-parallel
        String treeMakerTypeStr = config.getString(prefix + KEY + "tree_maker");
        tree_maker_type = TreeMakerType.valueOfType(treeMakerTypeStr);
        CheckUtils.check(tree_maker_type != null, "[GBDT] tree maker type(%s) invalid, data or feature", treeMakerTypeStr);

        if (tree_maker_type.equals(TreeMakerType.DATA_PARALLEL)) {
            String policyStr = config.getString(prefix + KEY + "tree_grow_policy");
            tree_grow_policy = TreeGrowPolicy.valueOfType(policyStr);
            CheckUtils.check(tree_grow_policy != null, "[GBDT] tree_grow_policy (%s) invalid, loss or level", policyStr);

            if (max_depth > 0) {
                if (max_leaf_cnt < 0) {
                    max_leaf_cnt = (int)Math.pow(2, max_depth);
                } else {
                    max_leaf_cnt = Math.min(max_leaf_cnt, (int)Math.pow(2, max_depth));
                }
            }
            histogram_pool_capacity = config.getDouble(prefix + KEY + "histogram_pool_capacity");

        } else if (tree_maker_type.equals(TreeMakerType.FEATURE_PARALLEL)) {
            tree_grow_policy = TreeGrowPolicy.LEVEL_WISE;
            if (max_leaf_cnt > 0) {
                if (max_depth < 0) {
                    max_depth = Constants.MAX_DEPTH;
                }
            }
            histogram_pool_capacity = -1;
        }

        uniform_base_prediction = (float) config.getDouble(prefix + KEY + "uniform_base_prediction");
        sample_dependent_base_prediction = config.getBoolean(prefix + KEY + "sample_dependent_base_prediction");

        subsample = (float)config.getDouble(prefix + KEY + "instance_sample_rate");
        feature_sample_rate = (float)config.getDouble(prefix + KEY + "feature_sample_rate");

        class_num = 1;
        if (objective.startsWith("softmax")) {
            class_num = config.getInt(prefix + KEY + "class_num");
        }
        if (objective.equalsIgnoreCase("sigmoid")) {
            try {
                sigmoid_zmax = config.getDouble(prefix + KEY + "sigmoid_zmax");
            } catch (ConfigException.Missing e) {
                sigmoid_zmax = 0.;
            }
        }
        if (objective.equalsIgnoreCase("l1")) {
            try {
                lad_refine_appr = config.getBoolean(prefix + KEY + "lad_refine_appr");
            } catch (ConfigException.Missing e) {
                lad_refine_appr = true;
            }
        }

        watch_train = config.getBoolean(prefix + KEY + "watch_train");
        watch_test = config.getBoolean(prefix + KEY + "watch_test");

        just_evaluate = config.getBoolean(prefix + KEY + "just_evaluate");
        eval_metrics = config.getStringList(prefix + KEY + "eval_metric");

        try {
            verbose = config.getBoolean(prefix + KEY + "verbose");
        } catch (ConfigException.Missing e) {
            verbose = config.getBoolean("verbose");
        }

        checkParams();
    }

    private void checkEvalName(List<String> evalNameList) {
        for (String evalName : evalNameList) {
            String[] cols = evalName.split(IEvaluator.NAME_DELIM);
            CheckUtils.check(EvaluatorFactory.EvalNameSet.contains(cols[0]), "[GBDT] eval name " + cols[0] + " invalid");
        }
    }


    private void checkParams() {
        CheckUtils.check(round_num >=1, "[GBDT] round_num(%d) should >=1", round_num);
        CheckUtils.check(class_num >= 1, "[GBDT] class_num(%d) should >= 1", class_num);
        CheckUtils.check(min_split_loss >= 0, "[GBDT] min_split_loss(%f) should >= 0", min_split_loss);
        CheckUtils.check((min_split_samples >= 2 || min_split_samples < 0), "[GBDT] min_split_samples(%d) should >=2 or < 0", min_split_samples);

        CheckUtils.check(max_leaf_cnt != 0, "[GBDT] max_leaf_cnt(%d) should not be 0", max_leaf_cnt);
        CheckUtils.check(!(max_leaf_cnt < 0 && max_depth < 0),
                "[GBDT] max_leaf_cnt(%d) and max_depth(%d) should not be both negative", max_leaf_cnt, max_depth);

        // only softmax can set class_num >= 1
        CheckUtils.check(!((!(objective.startsWith("softmax")) && class_num > 1)),
                "[GBDT] class_num(%d) and objective(%s) inconsistent", class_num, objective);
        // check uniform_base_prediction
        CheckUtils.check(!(objective.equals("sigmoid") && (uniform_base_prediction <= 0.f || uniform_base_prediction >= 1.f)),
                "[GBDT] uniform_base_prediction(%f) for sigmoid should between (0, 1)", uniform_base_prediction);

        CheckUtils.check(subsample > 0.0f && subsample <= 1.0f, "instance_sample_rate(%f) should belong to (0, 1]", subsample);
        CheckUtils.check(feature_sample_rate > 0.0f && feature_sample_rate <= 1.0f, "feature_sample_rate(%f) should belong to (0, 1]", feature_sample_rate);

        CheckUtils.check(sigmoid_zmax >= 0.0, "sigmoid_zmax(%f) for sigmoid should >=0, recommend [2, 4]", sigmoid_zmax);

        checkEvalName(eval_metrics);
    }

}
