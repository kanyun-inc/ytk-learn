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

import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.feature.FeatureHash;
import com.fenbi.ytklearn.fs.IFileSystem;
import com.fenbi.ytklearn.loss.LossFunctions;
import com.fenbi.ytklearn.param.CommonParams;
import com.fenbi.ytklearn.param.DataParams;
import com.fenbi.ytklearn.param.HyperParams;
import com.fenbi.ytklearn.param.ModelParams;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.typesafe.config.Config;
import lombok.Data;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author xialong
 */

@Data
public abstract class ContinuousDataFlow extends DataFlow {
    public static final Logger LOG = LoggerFactory.getLogger(ContinuousDataFlow.class);

    protected FeatureHash featureHash;
    protected float [][]w;
    protected float [][]precision;
    protected CommonParams commonParams;
    protected DataParams dataParams;
    protected ModelParams modelParams;
    protected HyperParams hyperParams;

    public ContinuousDataFlow(IFileSystem fs,
                              Config config,
                              ThreadCommSlave comm,
                              int threadNum,
                              boolean needPyTransform,
                              String pyTransformScript) throws Exception {
        super(fs, config,
                comm,
                threadNum,
                needPyTransform,
                pyTransformScript);

        this.commonParams = CommonParams.loadParams(config);
        this.dataParams = commonParams.dataParams;
        this.modelParams = commonParams.modelParams;
        this.featureHash = FeatureHash.build(commonParams.featureParams.feature_hash.bucket_size,
                commonParams.featureParams.feature_hash.seed, commonParams.featureParams.feature_hash.feature_prefix)
                .withDelim(commonParams.dataParams.delim.features_delim, commonParams.dataParams.delim.feature_name_val_delim);

        this.hyperParams = new HyperParams(config, "");

        CheckUtils.check(commonParams.lossParams.regularization.l2.length ==
                hyperParams.hoag.l2.length,
                "commonParams.lossParams.regularization.l2 length must equal to hyperParams.hoag.l2.length");

        CheckUtils.check(commonParams.lossParams.regularization.l2.length ==
                        hyperParams.grid.l2.length,
                "3*commonParams.lossParams.regularization.l2 length must equal to hyperParams.grid.l2.length");

        LOG_UTILS.verboseInfo("commonParams:" + commonParams);
        LOG_UTILS.verboseInfo("hyper:" + hyperParams);
    }

    @Override
    public boolean aheadLoadModel() {
        // just for MLR, must loaded train/test coredata before predict
        return false;
    }

    @Override
    public CoreParams createCoreParams() throws Exception {
        CoreParams coreParams = new CoreParams();
        coreParams.x_delim = dataParams.delim.x_delim;
        coreParams.y_delim = dataParams.delim.y_delim;
        coreParams.features_delim = dataParams.delim.features_delim;
        coreParams.feature_name_val_delim = dataParams.delim.feature_name_val_delim;
        coreParams.lossFunction = LossFunctions.createLossFunction(commonParams.lossParams.loss_function);

        coreParams.needYSampling = dataParams.y_sampling.size() > 0 &&
                LossFunctions.pureClassification(commonParams.lossParams.loss_function);

        if (dataParams.y_sampling.size() > 0 &&
                !LossFunctions.pureClassification(commonParams.lossParams.loss_function)) {
            throw new Exception("loss function:" + commonParams.lossParams.loss_function + " not supporting y sampling!");
        }

        if (coreParams.needYSampling) {
            int classNum = getYnum();
            if (classNum == 1) {
                classNum = 2;
            }
            coreParams.ySampling = new float[classNum];
            for (int i = 0; i < classNum; i++) {
                coreParams.ySampling[i] = 1.0f;
            }
            for (int i = 0; i < dataParams.y_sampling.size(); i++) {
                String []splits = dataParams.y_sampling.get(i).split("@");
                coreParams.ySampling[Integer.parseInt(splits[0])] = Float.parseFloat(splits[1]);
            }
        }

        if (hyperParams.switch_on) { // TODO: GBDT caution
            CheckUtils.check((dataParams.test.data_path != null && dataParams.test.data_path.trim().length() > 0),
                    "hyper params opt must contain test data!");
        }

        coreParams.assigned = dataParams.assigned;
        coreParams.unassigned_mode = dataParams.unassigned_mode;
        coreParams.train_data_path = dataParams.train.data_path;
        coreParams.train_max_error_tol = dataParams.train.max_error_tol;
        coreParams.test_data_path = dataParams.test.data_path;
        //coreParams.test_data_temp_path = dataParams.test.data_temp_path;
        coreParams.test_max_error_tol = dataParams.test.max_error_tol;

        coreParams.need_dict = modelParams.need_dict;
        coreParams.dict_path = modelParams.dict_path;

        coreParams.need_bias = modelParams.need_bias;
        coreParams.bias_feature_name = modelParams.bias_feature_name;
        coreParams.justEvaluation = commonParams.lossParams.just_evaluate;
        coreParams.modelPath = modelParams.data_path;
        coreParams.needYStat = LossFunctions.pureClassification(commonParams.lossParams.loss_function);
        coreParams.need_feature_hash = commonParams.featureParams.feature_hash.need_feature_hash;
        coreParams.verbose = commonParams.verbose;
        coreParams.featureParams = commonParams.featureParams;
        coreParams.continue_train = modelParams.continue_train;

        return coreParams;

    }

//    @Override
//    protected Map<String, Float> line2FeatureMap(String line, String []info) {
//        return featureHash.line2Map(info[2], commonParams.fHashParams.need_feature_hash);
//    }

    protected abstract boolean needLaplace();
}
