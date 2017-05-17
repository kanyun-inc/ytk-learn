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

import com.fenbi.ytklearn.data.Tuple;
import com.fenbi.ytklearn.data.gbdt.TreeMakerType;
import com.fenbi.ytklearn.exception.YtkLearnException;
import com.fenbi.ytklearn.loss.LossFunctions;
import com.fenbi.ytklearn.loss.ILossFunction;
import com.fenbi.ytklearn.fs.IFileSystem;
import com.fenbi.ytklearn.data.Constants;
import com.fenbi.ytklearn.data.gbdt.GBDTModel;
import com.fenbi.ytklearn.param.FeatureParams;
import com.fenbi.ytklearn.param.gbdt.GBDTCommonParams;
import com.fenbi.ytklearn.param.gbdt.GBDTDataParams;
import com.fenbi.ytklearn.param.gbdt.GBDTModelParams;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.typesafe.config.Config;
import lombok.Getter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

/**
 * @author wufan
 * @author xialong
 */

public class GBDTDataFlow extends DataFlow {
    public static final Logger LOG = LoggerFactory.getLogger(GBDTDataFlow.class);

    @Getter
    private  GBDTCommonParams commonParams;
    private GBDTModelParams modelParams;
    private GBDTDataParams dataParams;
    private ILossFunction obj;

    private int maxFeatureDim;
    private final int numTreeInGroup;

    private float baseScore;
    private boolean sampleDepdtBasePrediction;

    @Getter
    private GBDTModel[] GBDTModels;


    GBDTDataFlow(IFileSystem fs,
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

        // GBDTFeatureParams.init() is called by handleOtherTrainInfo, we need fName2IndexMap
        commonParams = GBDTCommonParams.loadParams(config);
        modelParams = commonParams.modelParams;
        dataParams = commonParams.dataParams;

        obj = LossFunctions.createLossFunction(commonParams.optimizationParams.objective);

        maxFeatureDim = dataParams.max_feature_dim;
        numTreeInGroup = commonParams.optimizationParams.class_num;

        baseScore = (float) obj.pred2Score(commonParams.optimizationParams.uniform_base_prediction);
        sampleDepdtBasePrediction = commonParams.optimizationParams.sample_dependent_base_prediction;

        CheckParams();
        LOG_UTILS.importantInfo("commonParams:" + commonParams);
    }

    private void CheckParams() {
        int slaveNum = comm.getSlaveNum();
        if (commonParams.optimizationParams.tree_maker_type.equals(TreeMakerType.FEATURE_PARALLEL) && slaveNum > 1) {
            throw new YtkLearnException("[GBDT] feature parallel only support single machine");
        }
    }

    @Override
    public CoreParams createCoreParams() throws Exception {
        CoreParams coreParams = new CoreParams();
        coreParams.x_delim = dataParams.delim.x_delim;
        coreParams.y_delim = dataParams.delim.y_delim;
        coreParams.features_delim = dataParams.delim.features_delim;
        coreParams.feature_name_val_delim = dataParams.delim.feature_name_val_delim;

        String objName = commonParams.optimizationParams.objective;
        coreParams.lossFunction = LossFunctions.createLossFunction(objName);
        coreParams.needYStat = LossFunctions.pureClassification(objName);
        coreParams.needYSampling = dataParams.y_sampling.size() > 0 &&
                LossFunctions.pureClassification(objName);

        if (dataParams.y_sampling.size() > 0 &&
                !LossFunctions.pureClassification(objName)) {
            throw new Exception("loss function:" + objName + " not supporting y sampling!");
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

        coreParams.assigned = dataParams.assigned;
        coreParams.train_data_path = dataParams.train.data_path;
        coreParams.train_max_error_tol = dataParams.train.max_error_tol;
        coreParams.test_data_path = dataParams.test.data_path;
        coreParams.test_max_error_tol = dataParams.test.max_error_tol;

        coreParams.need_dict = modelParams.need_dict;
        coreParams.dict_path = modelParams.dict_path;

        coreParams.need_bias = false;
        coreParams.bias_feature_name = "";
        coreParams.justEvaluation = commonParams.optimizationParams.just_evaluate;
        coreParams.verbose = commonParams.verbose;
        coreParams.featureParams = new FeatureParams(commonParams.featureParams.filter_threshold);
        coreParams.continue_train = modelParams.continue_train;

        return coreParams;
    }


    @Override
    protected CoreData getCoreData() {
        if (commonParams.optimizationParams.just_evaluate) {
            maxFeatureDim = Math.max(fName2IndexMap.size(), 1);
        }
        return new GBDTCoreData(comm, coreParams, featureMap, pyTransformFunc, needPyTransform,
                maxFeatureDim, numTreeInGroup, obj, baseScore, sampleDepdtBasePrediction);
    }

    @Override
    protected int getYnum() {
        return numTreeInGroup;
    }

    @Override
    protected boolean aheadLoadModel() {
        if (coreParams.justEvaluation) {
            return true;
        } else {
            return false;
        }
    }

    @Override
    protected void setDim() throws Mp4jException {
        dim = fName2IndexMap.size();
        //check
        if (!coreParams.justEvaluation) {
            CheckUtils.check(dim <= maxFeatureDim,
                    "[GBDT] user set max_feature_dim(%d) should >= real feature number(%d) of training data",
                    maxFeatureDim, dim);
            CheckUtils.check(dim > 0, "[GBDT] feature number(%d) <= 0 is invalid! may be cased by no data or filter all feature", dim);
            LOG_UTILS.importantInfo(String.format("feature number of training data is %d, user set max_feature_dim is %d", dim, maxFeatureDim));
        } else {
            LOG_UTILS.importantInfo(String.format("feature number used in model is %d, max_feature_dim used in data is %d", dim, maxFeatureDim));
        }
    }

    @Override
    public void handleLocalIdx(int tidx, Map<Integer, String> localFIndex2NameMap) {
        if (replacedIdx) {
            return;
        }

        int localFeatureDim = localFIndex2NameMap.size();

        if (!coreParams.need_dict) {
            CoreData coreData = threadTrainCoreDatas[tidx];
            int x[][] = coreData.x;
            int[] realNum = coreData.realNum;

            int[] tmpX = new int[localFeatureDim];
            int sampleOffset = 0;
            for (int k = 0; k < coreData.cursor2d; k++) {
                int lsNumInt = realNum[k];
                for (int i = 0; i < lsNumInt; i++) {
                    sampleOffset = i * maxFeatureDim;
                    for (int j = 0; j < localFeatureDim; j++) {
                        tmpX[j] = x[k][sampleOffset + j];
                        // init features to missing
                        x[k][sampleOffset + j] = Constants.INT_MISSING_VALUE;
                    }
                    for (int j = 0; j < localFeatureDim; j++) {
                        String fname = localFIndex2NameMap.get(j);
                        CheckUtils.check(fname != null, "[GBDT] local feature index2name map(size=%d) error, can't find index %d!",localFeatureDim, j);
                        Integer fid = fName2IndexMap.get(fname);
                        if (fid != null) {
                            x[k][sampleOffset + fid] = tmpX[j];
                        }
                    }
                }
            }
        }
    }


    @Override
    protected void handleOtherTrainInfo() throws Mp4jException {
        if (loadingTrainData) {
            // for local version, generate reverse feature matrix in train
            GBDTCoreData gbdtCoreData;
            if (commonParams.optimizationParams.tree_maker_type.equals(TreeMakerType.FEATURE_PARALLEL)) {
                GBDTCoreData allData = GBDTCoreData.mergeThreadFeatData(threadTrainCoreDatas);
                for (int t = 0; t < threadNum; t++) {
                    gbdtCoreData = (GBDTCoreData)threadTrainCoreDatas[t];
                    gbdtCoreData.allData = allData;
                }
            }

            int[][] globalFeatureAssignFrom = new int[slaveNum][threadNum];
            int[][] globalFeatureAssignTo = new int[slaveNum][threadNum];
            // assign feature col index for each rank
            int [][] assignXColForthread = DataUtils.avgAssign(dim, slaveNum * threadNum);
            int index = 0;
            for (int i = 0; i < slaveNum; i++) {
                for (int j = 0; j < threadNum; j++) {
                    globalFeatureAssignFrom[i][j] = assignXColForthread[index][0];
                    globalFeatureAssignTo[i][j] = assignXColForthread[index][1];
                    index++;
                }
            }
            // assign feature col index to each thread
            for (int t = 0; t < threadNum; t++) {
                gbdtCoreData = (GBDTCoreData)threadTrainCoreDatas[t];
                gbdtCoreData.xColRange = new int[] {globalFeatureAssignFrom[rank][t], globalFeatureAssignTo[rank][t]};
                LOG_UTILS.verboseInfo("train data feature index from " + gbdtCoreData.xColRange[0] + " to " + gbdtCoreData.xColRange[1], false);
                gbdtCoreData.globalFeatureAssignFrom = globalFeatureAssignFrom;
                gbdtCoreData.globalFeatureAssignTo = globalFeatureAssignTo;
                gbdtCoreData.usefulFeatureDim = dim;
            }
        }

        // train phase, init feature approximate params
        if (!coreParams.justEvaluation) {
            commonParams.featureParams.init(fName2IndexMap, gTrainRealSampleNum);
        }
    }

    @Override
    protected void handleOtherTestInfo() {
        for (int t = 0; t < threadNum; t++) {
            ((GBDTCoreData)threadTestCoreDatas[t]).usefulFeatureDim = dim;
        }
    }

    @Override
    protected void loadDict() throws IOException, Mp4jException {
        // already load dict from model
        if (coreParams.justEvaluation) {
            coreParams.need_dict = true;
            LOG_UTILS.importantInfo("just evaluation, so load this model's dict, path:" + coreParams.modelPath);
            return;
        }

        super.loadDict();

    }

    @Override
    protected void loadModel() throws IOException, Mp4jException, ClassNotFoundException {
        CheckUtils.check(!(modelParams.continue_train && !fs.exists(modelParams.data_path)),
                "GBDT: set continue_train=true, but old model doesn't exist");

        int numTreeInGroup = commonParams.optimizationParams.class_num;
        int targetNumRound = commonParams.optimizationParams.round_num;
        float basePrediction = commonParams.optimizationParams.uniform_base_prediction;
        String objName = commonParams.optimizationParams.objective;

        int curNumRound = 0;

        GBDTModels = new GBDTModel[threadNum];

        for (int t = 0; t < threadNum; t++) {
            GBDTModel curModel = null;

            if (modelParams.continue_train || coreParams.justEvaluation) {
                // load old model
                curModel = new GBDTModel();
                BufferedReader reader = null;
                try {
                    reader = new BufferedReader(fs.getReader(modelParams.data_path));
                    curModel.loadModel(reader);
                } finally {
                    if (reader != null) {
                        reader.close();
                    }
                }
                // check param consistent
                CheckUtils.check(objName.equals(curModel.objName), "GBDT: params inconsistent! objective is %s in old model," +
                        " but %s in gbdt.conf", curModel.objName, objName);

                CheckUtils.check(Math.abs(basePrediction - curModel.basePrediction) < 1E-6, "GBDT: params inconsistent! uniform_base_prediction is %f in old model," +
                        " but %f in gbdt.conf", curModel.basePrediction, basePrediction);

                CheckUtils.check(numTreeInGroup == curModel.numTreeInGroup,
                        "GBDT: params inconsistent! class_num is %d in old model," +
                                " but class_num is %d in gbdt.conf", curModel.numTreeInGroup, numTreeInGroup);

                int numTree = curModel.trees.size();
                CheckUtils.check(numTree % numTreeInGroup == 0,
                        "GBDT: model error! old model num_tree=%d, but class_num is %d", numTree, numTreeInGroup);
                curNumRound = numTree / numTreeInGroup;

            } else {
                curModel = new GBDTModel(basePrediction, numTreeInGroup, objName);
            }

            GBDTModels[t] = curModel;
        }

        if (modelParams.continue_train) {
            CheckUtils.check(curNumRound < targetNumRound,
                    "GBDT: old model round_num(%d) >= target round_num(%d), no need to train, exit!", curNumRound, targetNumRound);
            LOG_UTILS.importantInfo(String.format("load model finished, old model round_num=%d, target round_num=%d, num_tree_in_group=%d",
                    curNumRound, targetNumRound, numTreeInGroup));

        } else if (coreParams.justEvaluation) {
            CheckUtils.check(curNumRound >= targetNumRound,
                    "GBDT: model round_num(%d) < use round_num(%d), exit!", curNumRound, targetNumRound);
            // load feature dict from model
            fName2IndexMap = GBDTModels[0].genFeatureDict();
            LOG_UTILS.importantInfo(String.format("just evaluation(load model and generate feature dict), load model finished, model round_num=%d, use round_num=%d, num_tree_in_group=%d",
                    curNumRound, targetNumRound, numTreeInGroup));

        } else {
            LOG_UTILS.importantInfo("continue_train = false, train new model...");
        }


    }

    @Override
    // save model to file
    public void dumpModel() throws IOException, Mp4jException {
        CheckUtils.check(GBDTModels.length >= 0, "GBDT model is empty");
        String visualModelPath = modelParams.data_path;
        if (visualModelPath == null || visualModelPath.trim().length() == 0) {
            return;
        }
        PrintWriter visualWriter = null;
        try {
            visualWriter = new PrintWriter(fs.getOutputStream(visualModelPath));
            List<String> dumpStr = GBDTModels[0].dumpModel(true);
            for (int i = 0; i < dumpStr.size(); i++ ) {
                visualWriter.print(dumpStr.get(i));
            }
        } finally {
            if (visualWriter != null) {
                visualWriter.close();
            }
        }
        LOG_UTILS.importantInfo("GBDT model is saved in " + visualModelPath);
    }

    public void dumpFeatureImportance() throws IOException, Mp4jException {
        String feaImpPath = modelParams.feature_importance_path;
        if (feaImpPath == null || feaImpPath.trim().length() == 0) {
            return;
        }
        CheckUtils.check(GBDTModels.length >= 0, "GBDT model is empty");

        Map<String, Tuple<Integer, Double>> feaImpMap = GBDTModels[0].getFeatureImportance();
        String line = "";
        PrintWriter writer = null;
        try {
            writer = new PrintWriter(fs.getOutputStream(feaImpPath));
            writer.println("feature_name\tsum_split_count\tsum_gain");
            for (Map.Entry<String, Tuple<Integer, Double>> entry : feaImpMap.entrySet()) {
                line = entry.getKey() + "\t" + entry.getValue().v1 + "\t" + entry.getValue().v2;
                writer.println(line);
            }
        } finally {
            if (writer != null) {
                writer.close();
            }
        }
        LOG_UTILS.importantInfo("GBDT feature importance is saved in " + feaImpPath);
    }

}
