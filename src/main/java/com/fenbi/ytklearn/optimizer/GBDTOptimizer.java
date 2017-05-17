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

package com.fenbi.ytklearn.optimizer;

import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.ytklearn.data.gbdt.FeatureApprData;
import com.fenbi.ytklearn.data.gbdt.Tree;
import com.fenbi.ytklearn.data.gbdt.SamplePositionData;
import com.fenbi.ytklearn.data.gbdt.TreeMakerType;
import com.fenbi.ytklearn.dataflow.GBDTCoreData;
import com.fenbi.ytklearn.dataflow.GBDTDataFlow;
import com.fenbi.ytklearn.dataflow.GBMLRDataFlow;
import com.fenbi.ytklearn.eval.ConfusionMatrixEvaluator;
import com.fenbi.ytklearn.eval.EvalSet;
import com.fenbi.ytklearn.feature.gbdt.missing.FillMissingValue;
import com.fenbi.ytklearn.loss.ILossFunction;
import com.fenbi.ytklearn.loss.LossFunctions;
import com.fenbi.ytklearn.loss.SigmoidFunction;
import com.fenbi.ytklearn.loss.SoftmaxFunction;
import com.fenbi.ytklearn.optimizer.gbdt.DataParallelTreeMaker;
import com.fenbi.ytklearn.optimizer.gbdt.FeatureParallelTreeMakerByLevel;
import com.fenbi.ytklearn.optimizer.gbdt.ITreeMaker;
import com.fenbi.ytklearn.optimizer.gbdt.TreeRefiner;
import com.fenbi.ytklearn.param.gbdt.GBDTCommonParams;
import com.fenbi.ytklearn.param.gbdt.GBDTOptimizationParams;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.fenbi.ytklearn.utils.LogUtils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author wufan
 * @author xialong
 */

public class GBDTOptimizer implements IOptimizer {

    private static final String TRAIN_STR = "train";
    private static final String TEST_STR = "test";
    public LogUtils LOG_UTILS;

    // train phase related
    // assign from construct function
    private final GBDTDataFlow dataFlow;
    private final GBDTCommonParams commonParams;
    private final TreeMakerType treeMakerType;

    private final ThreadCommSlave comm;
    private final int threadIdx;
    private final int rank;
    private final boolean verbose;


    private final Map<String, Integer> fName2IndexMap;
    private Map<Integer, String> fIndex2NameMap;
    private final boolean hasTestData;

    private GBMLRDataFlow.Type learnType;
    // construct from params
    private ILossFunction obj;
    private EvalSet evalSet;
    private EvalSet testEvalSet;
    private Object evalObjInfo;

    // number of trees in a boosting round, for softmax >= 1, else =1
    private int numTreeInGroup;
    // watch evaluation during each iterator
    private boolean watchTrain;
    private boolean watchTest;

    private boolean enableMissingValue;
    // missing value computed by train dataset
    private float[] missingValueFill;

    private List<Tree> trees;
    private ITreeMaker treeMaker;
    private TreeRefiner treeRefiner;

    // === data part: for train phase ===
    private GBDTCoreData trainData;
    private GBDTCoreData testData;

    // tmp data in predictAndCalcLossGrad for numTreeInGroup > 1 (softmax)
    private double[] localPred;
    private double[] localScore;
    private double[] localY;
    private double[] localGrad;
    private double[] localHess;

    // === distributed version ===
    // global feature approximate
    private FeatureApprData feaApprData;
    // sample position
    private SamplePositionData position;

    // == local version ==
    private int[] positionGlobal;

    private boolean trainWeightAndReal;
    private boolean testWeightAndReal;

    private String REPORT_STR_FORMAT;

    public GBDTOptimizer(GBDTDataFlow dataFlow, int threadIdx) throws Exception {
        this.dataFlow = dataFlow;

        this.comm = dataFlow.getComm();
        this.threadIdx = threadIdx;
        this.rank = comm.getRank();

        this.fName2IndexMap = dataFlow.getFName2IndexMap();  //global feature_name -->index
        this.commonParams = dataFlow.getCommonParams();
        this.treeMakerType = commonParams.optimizationParams.tree_maker_type;
        this.verbose = commonParams.optimizationParams.verbose && treeMakerType.equals(TreeMakerType.DATA_PARALLEL);
        this.learnType = commonParams.optimizationParams.learn_type;

        this.LOG_UTILS = new LogUtils(comm, verbose);
        this.REPORT_STR_FORMAT = String.format("[model=gbdt] [loss=%s] [iter=%%d] %%s", commonParams.optimizationParams.objective);

        //for debug
        if (false) {
            StringBuffer fName2IndexStr = new StringBuffer("");
            for (Map.Entry<String, Integer> entry : this.fName2IndexMap.entrySet()) {
                fName2IndexStr.append(entry.getKey());
                fName2IndexStr.append("=");
                fName2IndexStr.append(entry.getValue());
                fName2IndexStr.append(",");
            }
            LOG_UTILS.verboseInfo("===fname2index map:===" + fName2IndexStr.toString());
        }
        // end debug

        fIndex2NameMap = new HashMap<>(this.fName2IndexMap.size());
        for (Map.Entry<String, Integer> entry : this.fName2IndexMap.entrySet()) {
            fIndex2NameMap.put(entry.getValue(), entry.getKey());
        }

        this.hasTestData = dataFlow.isNeedTest();
        this.trainWeightAndReal = Math.abs(dataFlow.getGTrainWeightSampleNum() - dataFlow.getGTrainRealSampleNum()) > 1e-6;
        if (this.hasTestData) {
            this.testWeightAndReal = Math.abs(dataFlow.getGTestWeightSampleNum() - dataFlow.getGTestRealSampleNum()) > 1e-6;
        } else {
            this.testWeightAndReal = false;
        }
    }

    // train or eval
    public void operate() throws Exception {
        initConf();
        boolean isTrain = !commonParams.optimizationParams.just_evaluate;
        if (isTrain) {
            initDataForTrain();
            train();
        } else {
            initDataForEval();
            eval();
        }
    }


    private void initConf() throws Exception {
        // init config
        final GBDTOptimizationParams optParams = commonParams.optimizationParams;
        obj = LossFunctions.createLossFunction(optParams.objective);
        if (obj instanceof SigmoidFunction) {
            obj.setParam(new HashMap<String, String>() {{
                put("sigmoid_zmax", optParams.sigmoid_zmax + "");
            }});
        }

        watchTrain = optParams.watch_train;
        watchTest = optParams.watch_test;

        numTreeInGroup = optParams.class_num;
        enableMissingValue = commonParams.featureParams.enable_missing_value;

        // init model, add feature split index in model
        trees = dataFlow.getGBDTModels()[threadIdx].trees;
        updateFeatureIndexInModel();
    }

    // called by init in train phase, alloc tmp data space for train & test
    // fill missing value and compute approximate feature values(sync global feature info)
    // set numSample & numTrainSample
    private void initDataForTrain() throws Exception {
        final GBDTOptimizationParams optParams = commonParams.optimizationParams;
        // === init train related data ====
        trainData = (GBDTCoreData) (dataFlow.getThreadTrainCoreDatas()[threadIdx]);
        // avoid repeat compute
        trainData.sampleNum = (int) trainData.getTotalRealNum();
        trainData.weightSum = trainData.getTotalWeightNum();
        trainData.fIndex2NameMap = fIndex2NameMap;

        if (treeMakerType.equals(TreeMakerType.FEATURE_PARALLEL)) {
            trainData.createFeatureColData();

            // get sample global sample index
            int slaveNum = comm.getSlaveNum();
            int threadNum = comm.getThreadNum();

            long[] globalSampleCnt = new long[slaveNum * threadNum];
            for (int i = 0; i < globalSampleCnt.length; i++) {
                globalSampleCnt[i] = 0;
            }
            int curId = rank * threadNum + threadIdx;
            globalSampleCnt[curId] = trainData.sampleNum;
            globalSampleCnt = comm.allreduceArray(globalSampleCnt, Operands.LONG_OPERAND(), Operators.Long.SUM, 0, globalSampleCnt.length);

            trainData.globalSampleAssignFrom = new int[slaveNum][threadNum];
            trainData.globalSampleAssignTo = new int[slaveNum][threadNum];
            trainData.globalGradAssignFrom = new int[slaveNum][threadNum];
            trainData.globalGradAssignTo = new int[slaveNum][threadNum];
            int index = 0;
            int accuSampleNum = 0;
            for (int i = 0; i < slaveNum; i++) {
                for (int j = 0; j < threadNum; j++) {
                    trainData.globalSampleAssignFrom[i][j] = accuSampleNum;
                    accuSampleNum += globalSampleCnt[index];
                    trainData.globalSampleAssignTo[i][j] = accuSampleNum;

                    trainData.globalGradAssignFrom[i][j] = trainData.globalSampleAssignFrom[i][j] << 1;
                    trainData.globalGradAssignTo[i][j] = trainData.globalSampleAssignTo[i][j] << 1;
                    index++;
                }
            }

            trainData.xRowRange = new int[]{trainData.globalSampleAssignFrom[rank][threadIdx], trainData.globalSampleAssignTo[rank][threadIdx]};
            LOG_UTILS.verboseInfo("train data sample index from " + trainData.xRowRange[0] + " to " + trainData.xRowRange[1], false);

            // release origin data set, must wait util all the other threads come here
            trainData.allData = null;
        }

        // compute missing value
        FillMissingValue fillMV = null;
        if (enableMissingValue) {
            // fill missing value in both data.x and data.XT
            fillMV = new FillMissingValue(comm, treeMakerType.equals(TreeMakerType.FEATURE_PARALLEL));
            missingValueFill = fillMV.computeMissingValueFill(commonParams.featureParams, trainData);
            StringBuffer sb = new StringBuffer("");
            for (int fid = 0; fid < missingValueFill.length; fid++) {
                if (fid > 0) {
                    sb.append(",");
                }
                sb.append(fIndex2NameMap.get(fid) + "=" + missingValueFill[fid]);
            }
            LOG_UTILS.importantInfo("missing value(type:" + commonParams.featureParams.featureMissingParams + ") fill array (format:feature_name=missing_value):" + sb.toString());
            fillMV.fillMissingValue(missingValueFill, trainData);
        } else {
            missingValueFill = null;
        }

        // init eval set
        evalObjInfo = getEvalObjectInfo();
        evalSet = new EvalSet(comm, trainData);
        evalSet.addEvals(optParams.eval_metrics);

        // init test related data
        if (hasTestData) {
            testData = (GBDTCoreData) (dataFlow.getThreadTestCoreDatas()[threadIdx]);
            testData.sampleNum = (int) testData.getTotalRealNum();
            testData.weightSum = testData.getTotalWeightNum();
            testData.fIndex2NameMap = fIndex2NameMap;

            if (enableMissingValue) {
                fillMV.fillMissingValue(missingValueFill, testData);
            }

            testEvalSet = new EvalSet(comm, testData);
            testEvalSet.addEvals(optParams.eval_metrics);
        }

        // attention: initPred should be called before new GlobalSyncData
        // new GlobalSyncData will convert original train data feature values to feature approximate indexes
        trainData.initGradPairs();
        initPred();

        if (treeMakerType.equals(TreeMakerType.FEATURE_PARALLEL)) {
            // in local version, sample count max 2^32
            positionGlobal = new int[(int) dataFlow.getGTrainRealSampleNum()];

        } else { // distributed
            // convert origin feature to approximate data index, it'll change featureMatrix value in trainData
            feaApprData = new FeatureApprData(comm, commonParams.featureParams, trainData);
            // init position
            position = new SamplePositionData(trainData.sampleNum);
        }

        if (treeMakerType.equals(TreeMakerType.FEATURE_PARALLEL)) {
            treeMaker = new FeatureParallelTreeMakerByLevel(comm, commonParams.optimizationParams,
                    trainData, positionGlobal);
        } else {
            treeMaker = new DataParallelTreeMaker(comm, commonParams.optimizationParams,
                    trainData, position, feaApprData);
        }

        treeRefiner = new TreeRefiner(comm, commonParams);

    }

    // train phase: initialize origin raw predict(score) for train and test dataset
    // called by initDataForTrain
    // initPred should be called before new GlobalSyncData
    private void initPred() throws Mp4jException {
        predictAndCalcLossGrad(true, -1, true, true);
        if (hasTestData) {
            predictAndCalcLossGrad(false, -1, false, true);
        }
    }

    // called by init in just eval phase
    // alloc tmp data space for train & test
    private void initDataForEval() throws Exception {
        final GBDTOptimizationParams optParams = commonParams.optimizationParams;
        // === init train related data ====
        trainData = (GBDTCoreData) (dataFlow.getThreadTrainCoreDatas()[threadIdx]);
        // avoid repeat compute
        trainData.sampleNum = (int) trainData.getTotalRealNum();
        trainData.weightSum = trainData.getTotalWeightNum();
        trainData.fIndex2NameMap = fIndex2NameMap;

        // init eval set
        evalObjInfo = getEvalObjectInfo();
        evalSet = new EvalSet(comm, trainData);
        evalSet.addEvals(optParams.eval_metrics);

        // init test related data
        if (hasTestData) {
            testData = (GBDTCoreData) (dataFlow.getThreadTestCoreDatas()[threadIdx]);
            testData.sampleNum = (int) testData.getTotalRealNum();
            testData.weightSum = testData.getTotalWeightNum();
            testData.fIndex2NameMap = fIndex2NameMap;

            testEvalSet = new EvalSet(comm, testData);
            testEvalSet.addEvals(optParams.eval_metrics);
        }
    }

    private Object getEvalObjectInfo() {
        if (obj instanceof SigmoidFunction) {
            return new ConfusionMatrixEvaluator.ConfusionMatrixInfo(2, false);
        } else if (obj instanceof SoftmaxFunction) {
            return new ConfusionMatrixEvaluator.ConfusionMatrixInfo(numTreeInGroup, true);
        } else {
            return null;
        }
    }

    // for just eval
    public void eval() throws Exception {

        int numRound = commonParams.optimizationParams.round_num;
        CheckUtils.check(trainData != null, "train data should not be null!");
        boolean needAppend = hasTestData && (commonParams.optimizationParams.eval_metrics.size() > 0);

        LOG_UTILS.importantInfo("gbdt start eval! total round_num=" + numRound);
        long start = System.currentTimeMillis();
        double cost;
        StringBuffer res = new StringBuffer("eval results\n");

        res.append(predictAndCalcLossGrad(true, numRound, false, true));
        res.append(evalSet.eval(evalObjInfo, "train", trainWeightAndReal));
        if (hasTestData) {
            if (needAppend) {
                res.append("\n");
            }
            res.append(predictAndCalcLossGrad(false, numRound, false, true));
            res.append(testEvalSet.eval(evalObjInfo, "test", testWeightAndReal));
        }


        if (res.length() > 0) {
            LOG_UTILS.importantInfo(res.toString());
        }
        cost = (System.currentTimeMillis() - start) / 1000.0;
        LOG_UTILS.importantInfo(String.format("eval end, cost %.5f sec in all\n", cost));
    }

    // train procedure
    public void train() throws Exception {
        int numRound = commonParams.optimizationParams.round_num;
        int curRound = trees.size() / numTreeInGroup;
        CheckUtils.check(curRound <= numRound,
                "[GBDT] inner error, curRoundNum(%d) > totalRoundNum(%d)", curRound, numRound);
        CheckUtils.check(trainData != null, "train data should not be null!");

        boolean isFeaParallel = treeMakerType.equals(TreeMakerType.FEATURE_PARALLEL);
        boolean needAppend = hasTestData && watchTest && (commonParams.optimizationParams.eval_metrics.size() > 0);
        LOG_UTILS.importantInfo("gbdt start train! total round_num=" + numRound + ", current round_num=" + curRound);

        long start = System.currentTimeMillis();
        double cost;
        StringBuffer res = new StringBuffer("");
        for (int i = curRound; i < numRound; i++) {
            doBoost();
            res.append(predictAndCalcLossGrad(true, i + 1, true, isFeaParallel));
            if (watchTrain) {
                res.append(evalSet.eval(evalObjInfo, TRAIN_STR, trainWeightAndReal));
            }

            if (isFeaParallel) {
                addFeatureNameInModel(i);
            } else {
                // convertFeatureSplitValueInModel, replace split value with real, add default direction and add featureName
                convertModel(i);
            }
            // dump model, serve as check point
            if (rank == 0 && threadIdx == 0 &&
                    ((i + 1) % commonParams.modelParams.dump_freq == 0)) {
                dataFlow.dumpModel();
            }

            if (hasTestData) {
                if (needAppend) {
                    res.append("\n");
                }
                res.append(predictAndCalcLossGrad(false, i + 1, false, true));
                if (watchTest) {
                    res.append(testEvalSet.eval(evalObjInfo, TEST_STR, testWeightAndReal));
                }
            }

            cost = (System.currentTimeMillis() - start) / 1000.0;
            if (verbose && treeMaker.getTotalTimeStats() != null) {
                LOG_UTILS.importantInfo(String.format(REPORT_STR_FORMAT, i + 1, String.format(" %.5f sec elapse %s\n%s", cost, treeMaker.getTotalTimeStats().getStats(), res.toString())));
            } else {
                LOG_UTILS.importantInfo(String.format(REPORT_STR_FORMAT, i + 1, String.format(" %.5f sec elapse\n%s", cost, res.toString())));
            }
            res.setLength(0);
        }

        //do final eval, iter number start from 0
        res.setLength(0);
        res.append(evalSet.eval(evalObjInfo, TRAIN_STR, trainWeightAndReal));
        if (hasTestData) {
            if (commonParams.optimizationParams.eval_metrics.size() > 1) {
                res.append("\n");
            }
            res.append(testEvalSet.eval(evalObjInfo, TEST_STR, testWeightAndReal));
        }

        cost = (System.currentTimeMillis() - start) / 1000.0;
        if (verbose && treeMaker.getTotalTimeStats() != null) {
            LOG_UTILS.importantInfo(String.format("training end, %.5f sec in all %s\n%s", cost, treeMaker.getTotalTimeStats().getStats(), res.toString()));
        } else {
            LOG_UTILS.importantInfo(String.format("training end, %.5f sec in all\n%s", cost, res.toString()));
        }

        // save model to file
        if (rank == 0 && threadIdx == 0) {
            dataFlow.dumpModel();
            dataFlow.dumpFeatureImportance();
        }
    }

    private void doBoost() throws Exception {
        for (int gid = 0; gid < numTreeInGroup; gid++) {
            Tree newTree = treeMaker.make(gid);
            treeRefiner.refine(trainData, position, positionGlobal, newTree);
            trees.add(newTree);
        }
    }

    // cal loss and assign value to predict
    private String predictAndCalcLossGrad(boolean isTrain, int useRoundNum, boolean calGrad, boolean isOriginTree) throws Mp4jException {
        double[] data = new double[2];

        if (isTrain) {
            data[0] = predictAndCalcLossGrad(trainData, useRoundNum, isOriginTree, calGrad);
            data[1] = trainData.weightSum;
        } else {
            data[0] = predictAndCalcLossGrad(testData, useRoundNum, isOriginTree, calGrad);
            data[1] = testData.weightSum;
        }

        comm.allreduceArray(data, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, data.length);
        double loss = 0;
        if (data[1] != 0) {
            loss = data[0] / data[1];
        }

        String res = String.format("%s loss = %s\n", isTrain ? TRAIN_STR : TEST_STR, Double.toString(loss));
        return res;
    }


    public double predictAndCalcLossGrad(GBDTCoreData data, int useRoundNum, boolean isOriginTree, boolean computeGrad) throws Mp4jException {
        int totalRoundNum = trees.size() / numTreeInGroup;
        if (useRoundNum == -1) {
            useRoundNum = totalRoundNum;
        }

        CheckUtils.check(trees.size() % numTreeInGroup == 0,
                "[GBDT] inner error, tree num is %d, but treeNumInGroup is %d", trees.size(), numTreeInGroup);
        CheckUtils.check(useRoundNum <= totalRoundNum,
                "[GBDT] inner error, useRoundNum(%d) > totalRoundNum(%d)", useRoundNum, totalRoundNum);
        CheckUtils.check(data.lastPredRound >= 0 && data.lastPredRound <= useRoundNum,
                "[GBDT] inner error, lastPredRound < 0 or  lastPredRound(%d) > useRoundNum(%d)", data.lastPredRound, useRoundNum);

        boolean needPred = totalRoundNum > data.lastPredRound;
        double loss = 0.0;
        double[] deri = new double[2];
        int sampleIdx = 0;
        int denominatorRF = 0;
        if (numTreeInGroup == 1) {
            for (int k = 0; k < data.cursor2d; k++) {
                int curNum = data.realNum[k];
                for (int i = 0; i < curNum; i++) {
                    double wei = data.weight[k][i];
                    if (needPred) {
                        score(data, sampleIdx, 0, data.lastPredRound, useRoundNum, isOriginTree);
                    }
                    double combineScore;
                    if (learnType == GBMLRDataFlow.Type.RF) {
                        denominatorRF = useRoundNum == 0? 1: useRoundNum;
                        combineScore = data.score[k][i] / denominatorRF + data.initScore[k][i];
                    } else {
                        combineScore = data.score[k][i] + data.initScore[k][i];
                    }

                    loss += wei * obj.loss(combineScore, data.y[k][i]);
                    data.predict[k][i] = (float) obj.predict(combineScore);
                    if (computeGrad) {
                        obj.getDerivativeFast(data.predict[k][i], data.y[k][i], deri);
                        data.gradPairs[k][i << 1] = (float) (deri[0] * wei);
                        data.gradPairs[k][(i << 1) + 1] = (float) (deri[1] * wei);
                    }
                    sampleIdx++;
                }
            }

        } else { // softmax
            if (localPred == null) {
                localPred = new double[numTreeInGroup];
                localScore = new double[numTreeInGroup];
                localY = new double[numTreeInGroup];
                localGrad = new double[numTreeInGroup];
                localHess = new double[numTreeInGroup];
            }

            int offset = 0;
            int gradIdx = 0;
            for (int k = 0; k < data.cursor2d; k++) {
                int curNum = data.realNum[k];
                for (int i = 0; i < curNum; i++) {
                    offset = i * numTreeInGroup;
                    double wei = data.weight[k][i];
                    for (int gid = 0; gid < numTreeInGroup; gid++) {
                        if (needPred) {
                            score(data, sampleIdx, gid, data.lastPredRound, useRoundNum, isOriginTree);
                        }

                        if (learnType == GBMLRDataFlow.Type.RF) {
                            denominatorRF = useRoundNum == 0? 1: useRoundNum;
                            localScore[gid] = data.score[k][offset + gid] / denominatorRF + data.initScore[k][offset + gid];
                        } else {
                            localScore[gid] = data.score[k][offset + gid] + data.initScore[k][offset + gid];
                        }
                        localY[gid] = data.y[k][offset + gid];
                    }
                    loss += wei * obj.loss(localScore, localY);
                    obj.predict(localScore, localPred);

                    if (computeGrad) {
                        obj.getDerivativeFast(localPred, localY, localGrad, localHess);
                    }

                    gradIdx = offset << 1;
                    for (int j = 0; j < numTreeInGroup; j++) {
                        data.predict[k][offset + j] = (float) localPred[j];
                        if (computeGrad) {
                            data.gradPairs[k][gradIdx] = (float) (localGrad[j] * wei);
                            data.gradPairs[k][gradIdx + 1] = (float) (localHess[j] * wei);
                            gradIdx += 2;
                        }
                    }
                    sampleIdx++;
                }
            }
        }
        data.lastPredRound = totalRoundNum;
        return loss;
    }

    // useRoundNum=-1 means use all trees
    // isOriginTree=true means tree split value is converted to origin and float feature is storaged in int
    // isOriginTree=false means tree split value is feature slot, and feature values is apporoximated by slot_num
    private void predictScore(GBDTCoreData data, int useRoundNum, boolean isOriginTree) {
        int totalRoundNum = trees.size() / numTreeInGroup;
        if (useRoundNum == -1) {
            useRoundNum = totalRoundNum;
        }

        CheckUtils.check(trees.size() % numTreeInGroup == 0,
                "[GBDT] inner error, tree num is %d, but treeNumInGroup is %d", trees.size(), numTreeInGroup);
        CheckUtils.check(useRoundNum <= totalRoundNum,
                "[GBDT] inner error, useRoundNum(%d) > totalRoundNum(%d)", useRoundNum, totalRoundNum);
        CheckUtils.check(data.lastPredRound >= 0 && data.lastPredRound <= useRoundNum,
                "[GBDT] inner error, lastPredRound < 0 or  lastPredRound(%d) > useRoundNum(%d)", data.lastPredRound, useRoundNum);

        if (totalRoundNum == data.lastPredRound) {
            return;
        }
        for (int i = 0; i < data.sampleNum; i++) {
            for (int groupId = 0; groupId < numTreeInGroup; groupId++) {
                score(data, i, groupId, data.lastPredRound, useRoundNum, isOriginTree);
            }
        }
        data.lastPredRound = totalRoundNum;
    }

    // predict for one instance, called by predict, if has been predicted, then just assigned value from predBuffer to pred
    // sid: the index of instance in dataset
    // lastPredRound: last predict round, start from 0, equals 0 means no previous predict
    private float score(GBDTCoreData data, int sid, int groupId, int lastPredRound, int useRoundNum, boolean isOriginTree) {
        int index2D = sid / data.DENSE_MAX_1D_SAMPLE_CNT;
        int index1D = sid % data.DENSE_MAX_1D_SAMPLE_CNT;
        int scoreIdx = index1D * numTreeInGroup + groupId;

        float accPred = data.score[index2D][scoreIdx];
        float curPred = 0;
        int startTree = lastPredRound * numTreeInGroup + groupId;
        int endTree = useRoundNum * numTreeInGroup;
        for (int i = startTree; i < endTree; i += numTreeInGroup) {
            Tree tree = trees.get(i);
            curPred = tree.predict(data.getX()[index2D], index1D * data.maxFeatureDim, isOriginTree);
            accPred += curPred;
        }
        // set result back
        data.score[index2D][scoreIdx] = accPred;
        return curPred;
    }

    // called before dumping model, convert model of specified roundNum(starts from 0), including:
    // 1. convert slot index of feature to its original value
    // 2. add feature name in feature split info, add direction for missing value
    private void convertModel(int roundNum) {
        convertFeatureSplitValueInModel(roundNum);
        addFeatureNameInModel(roundNum);
    }

    // convert slot index of feature to its original value, roundNum starts from 0
    private void convertFeatureSplitValueInModel(int roundNum) {
        // replace split feature val in tree node according to conf
        Map<Integer, float[]> globalFeaSplitValsSorted = feaApprData.getGlobalFeaSplitValsSorted();
        int start = roundNum * numTreeInGroup;
        for (int i = 0; i < numTreeInGroup; i++) {
            Tree tree = trees.get(start + i);
            tree.convertFeatureSplitValueInModel(globalFeaSplitValsSorted, commonParams.featureParams.split_Type);
        }
    }

    // add feature name in feature split info, add direction for missing value
    private void addFeatureNameInModel(int roundNum) {
        int start = roundNum * numTreeInGroup;
        for (int i = 0; i < numTreeInGroup; i++) {
            Tree tree = trees.get(start + i);
            tree.addFeatureNameInModel(fIndex2NameMap);
            // add default direction according to missing value and split cond in tree node
            if (missingValueFill != null && missingValueFill.length > 0) {
                tree.addDefaultDirection(missingValueFill);
            }
        }
    }

    // convert featureName to featureIndex, call after load model
    public void updateFeatureIndexInModel() {
        for (Tree tree : trees) {
            tree.updateFeatureIndexInModel(fName2IndexMap);
        }
    }

}
