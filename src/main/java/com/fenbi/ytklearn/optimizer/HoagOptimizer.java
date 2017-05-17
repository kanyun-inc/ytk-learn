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

import com.fenbi.mp4j.utils.CommUtils;
import com.fenbi.ytklearn.dataflow.CoreData;
import com.fenbi.ytklearn.dataflow.ContinuousDataFlow;
import com.fenbi.ytklearn.loss.*;
import com.fenbi.ytklearn.eval.ConfusionMatrixEvaluator;
import com.fenbi.ytklearn.eval.EvalSet;
import com.fenbi.mp4j.exception.Mp4jException;
import com.fenbi.mp4j.comm.ThreadCommSlave;
import com.fenbi.mp4j.operand.Operands;
import com.fenbi.mp4j.operator.Operators;
import com.fenbi.ytklearn.utils.LogUtils;
import com.fenbi.ytklearn.utils.VectorUtils;
import com.fenbi.ytklearn.param.*;
import org.apache.commons.lang.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author xialong
 */

public abstract class HoagOptimizer implements IOptimizer {

    public static class TwoLoop {
        public double alpha;
        public double ys;
        public double yy;
        public float []y;
        public float []s;
        public int start;
        public int end;

        public TwoLoop(int dim, int start, int end) {

            this.start = start;
            this.end = end;
            int len = end - start;
            y = new float[len];
            s = new float[len];
            yy = 0.0;
            ys = 0.0;
            alpha = 0.0;
        }

    }

    protected final String modelName;
    protected final ContinuousDataFlow dataFlow;
    protected final ILossFunction lossFunction;

    protected final CommonParams params;
    protected final LineSearchParams lsParams;
    protected final LineSearchParams.LBFGSParams lbfgsParams;
    protected final DataParams dataParams;
    protected final ModelParams modelParams;
    protected final LossParams lossParams;
    protected final HyperParams hyperParams;
    protected final LineSearchParams.BackTracking backTrackingParams;
    protected final LineSearchParams.Mode mode;

//    protected final CoreData trainCoreData;
//    protected final CoreData testCoreData;
    protected final CoreData threadTrainCoreData;
    protected final CoreData threadTestCoreData;


    protected final ThreadCommSlave comm;
    protected final int threadIdx;
    protected final int rank;


    public double[] l2;
    public double[] l1;
    protected int []regularStart;
    protected int []regularEnd;

    protected float []w;
    protected float []precision;

    protected final int [][]x;
    protected final int [][]xidx;
    protected final float [][]y;
    protected final float [][]weight;
    protected final float [][]predict;
    protected final int realNum[];
    protected final double gWeightTrainNum;
    protected final double tWeightTrainNum;
    protected final EvalSet trainEvalSet;

    protected final int [][]xtest;
    protected final int [][]xidxtest;
    protected final float [][]ytest;
    protected final float [][]weighttest;
    protected final float [][]predicttest;
    protected final int realNumtest[];
    protected final double gWeightTestNum;
    protected final double tWeightTestNum;
    protected final EvalSet testEvalSet;

    protected final boolean hasTestData;

    protected final int dim;

    protected final boolean owner;

    protected double lossprev;
    protected double purelossprev;

    protected TwoLoop twoLoops [];
    private int [][]twoLoopsFroms;
    private int [][]twoLoopsTos;

    protected float []g;


    protected int hyper;
    private double tLoss;
    private double bestTestLoss;
    private double bestTrainLoss;
    private double bestPureTrainLoss;
    private double tOldLoss = Double.MAX_VALUE;
    private List<Double> hoagTestLossList = new ArrayList<>();
    //private List<List<Double>> testL1GradList = new ArrayList<>();
    private List<List<Double>> hoagTestL2GradList = new ArrayList<>();
    //private List<List<Double>> testL1StepList = new ArrayList<>();
    private List<List<Double>> hoagTestL2StepList = new ArrayList<>();
    private List<Double> hoagTestLossDeltaList = new ArrayList<>();
    private List<List<Double>> hoagHyperL2List = new ArrayList<>();
    //private List<List<Double>> hyperL1List = new ArrayList<>();
    private float gtest[];
    private float initw[];
    //private double []l1steps;
    private double [] hoagL2Steps;
    private boolean needHoagHyperOpt;
    //private boolean l1hyper[];
    private boolean hoagL2NeedHyper[];
    //private double []bestl1;

    private double []bestl2;
    private double []bestl1;
    private int bestIter;
    private float []bestW;

    private boolean needGridHyper;
    private double l1GridSearch[][];
    private double l2GridSearch[][];

    private boolean needHyperSearch;

    protected LogUtils LOG_UTILS;
    protected boolean trainWeightAndReal;
    protected boolean testWeightAndReal;

    public HoagOptimizer(String modelName,
                         ContinuousDataFlow dataFlow,
                         int threadIdx) throws Exception {
        this.modelName = modelName;
        this.dataFlow = dataFlow;
        this.dim = dataFlow.getDim();

        this.params = dataFlow.getCommonParams();
        this.lsParams = params.lsParams;
        this.lbfgsParams = params.lsParams.lbfgsParams;
        this.dataParams = params.dataParams;
        this.modelParams = params.modelParams;
        this.lossParams = params.lossParams;
        this.hyperParams = dataFlow.getHyperParams();
        this.backTrackingParams = params.lsParams.backtracking;
        this.mode = params.lsParams.mode;

        this.lossFunction = LossFunctions.createLossFunction(lossParams.loss_function);
        this.threadTrainCoreData = dataFlow.getThreadTrainCoreDatas()[threadIdx];

        this.comm = dataFlow.getComm();
        this.threadIdx = threadIdx;
        this.rank = comm.getRank();

        this.l1 = new double[lossParams.regularization.l1.length];
        this.l2 = new double[lossParams.regularization.l2.length];
        this.hoagL2NeedHyper = new boolean[l2.length];
        this.needHoagHyperOpt = false;
        for (int i = 0; i < l1.length; i++) {
            hoagL2NeedHyper[i] = (hyperParams.hoag.l2[i] > 0.0);
            needHoagHyperOpt |= hoagL2NeedHyper[i];
        }
        this.needHoagHyperOpt &= (hyperParams.switch_on && hyperParams.mode == HyperParams.Mode.HOAG && !lossParams.just_evaluate);
        this.needGridHyper = (hyperParams.switch_on && hyperParams.mode == HyperParams.Mode.GRID && !lossParams.just_evaluate);
        this.needHyperSearch = (needHoagHyperOpt || needGridHyper);

        for (int i = 0; i < l1.length; i++) {
            if (needHoagHyperOpt) {
                this.l1[i] = hyperParams.hoag.l1[i];
                this.l2[i] = hyperParams.hoag.l2[i];
            }

            if (!hyperParams.switch_on) {
                this.l1[i] = lossParams.regularization.l1[i];
                this.l2[i] = lossParams.regularization.l2[i];
            }
        }


        this.w = dataFlow.getW()[threadIdx];
        this.precision = dataFlow.getPrecision()[threadIdx];

        this.x = threadTrainCoreData.getX();
        this.xidx = threadTrainCoreData.getXidx();
        this.y = threadTrainCoreData.getY();
        this.weight = threadTrainCoreData.getWeight();
        this.predict = threadTrainCoreData.getPredict();
        this.realNum = threadTrainCoreData.getRealNum();
        this.gWeightTrainNum = dataFlow.getGTrainWeightSampleNum();
        this.tWeightTrainNum = threadTrainCoreData.getTotalWeightNum();
        this.trainEvalSet = new EvalSet(comm, threadTrainCoreData);
        this.trainEvalSet.addEvals(lossParams.evaluate_metric);

        this.trainWeightAndReal = Math.abs(dataFlow.getGTrainWeightSampleNum() - dataFlow.getGTrainRealSampleNum()) > 1e-6;
        this.testWeightAndReal = Math.abs(dataFlow.getGTestWeightSampleNum() - dataFlow.getGTestRealSampleNum()) > 1e-6;

        this.hasTestData = dataFlow.isNeedTest();

        this.needHoagHyperOpt &= hasTestData;

        if (hasTestData) {
            this.threadTestCoreData = dataFlow.getThreadTestCoreDatas()[threadIdx];
            this.xtest = threadTestCoreData.getX();
            this.xidxtest = threadTestCoreData.getXidx();
            this.ytest = threadTestCoreData.getY();
            this.weighttest = threadTestCoreData.getWeight();
            this.predicttest = threadTestCoreData.getPredict();
            this.realNumtest = threadTestCoreData.getRealNum();
            this.gWeightTestNum = dataFlow.getGTestWeightSampleNum();
            this.tWeightTestNum = threadTestCoreData.getTotalWeightNum();
            this.testEvalSet = new EvalSet(comm, threadTestCoreData);
            this.testEvalSet.addEvals(lossParams.evaluate_metric);
        } else {
            this.threadTestCoreData = null;
            this.xtest = null;
            this.xidxtest = null;
            this.ytest = null;
            this.weighttest = null;
            this.predicttest = null;
            this.realNumtest = null;
            this.gWeightTestNum = 0.0;
            this.tWeightTestNum = 0.0;
            this.testEvalSet = null;
        }

        this.owner = threadIdx == 0;

        this.LOG_UTILS = new LogUtils(comm, params.verbose);
    }

    public HoagOptimizer init() throws Exception {
        this.regularStart = getRegularStart();
        this.regularEnd = getRegularEnd();
        LOG_UTILS.verboseInfo("regular start:" + Arrays.toString(regularStart) + "," +
                ", end:" + Arrays.toString(regularEnd));
        LOG_UTILS.importantInfo("hoag hyper optimize:" + needHoagHyperOpt);
        LOG_UTILS.importantInfo("grid hyper search:" + needGridHyper);
        LOG_UTILS.importantInfo("need hyper params search:" + needHyperSearch);
        return this;
    }

    public abstract int[] getRegularStart();
    public abstract int[] getRegularEnd();

    public Object getEvalObjectInfo() {
        if (LossFunctions.pureClassification(lossParams.loss_function)) {
            return new ConfusionMatrixEvaluator.ConfusionMatrixInfo(2, false);
        } else {
            return null;
        }
    }

    protected void otherTrainHandle(int iter) throws Mp4jException {};
    protected void otherTestHandle(int iter) throws Mp4jException {};

    public double lbfgs(boolean close) throws Exception {

        long start = System.currentTimeMillis();
        int iter = 1;
        double testloss = -1;
        bestTestLoss = Double.MAX_VALUE;
        LOG_UTILS.importantInfo("################### optimization ###################");

        if (needHyperSearch) {
            gtest = new float[dim];
            bestW = new float[dim];

            if (hyperParams.restart) {
                initw = new float[dim];
                System.arraycopy(w, 0, initw, 0, dim);
            }

            bestl2 = new double[l2.length];
            bestl1 = new double[l1.length];

            if (needHoagHyperOpt) {
                hoagL2Steps = new double[l2.length];
                for (int i = 0; i < l2.length; i++) {
                    hoagL2Steps[i] = hyperParams.hoag.init_step;
                }

                hoagTestL2StepList.add(Arrays.asList(ArrayUtils.toObject(hoagL2Steps)));
                hoagHyperL2List.add(Arrays.asList(ArrayUtils.toObject(l2)));
            }

            if (needGridHyper) {
                int hyperGridCnt = 1;
                double [][]l1Grid = new double[l1.length][];
                for (int i = 0; i < l1.length; i++) {
                    double l1GridStart = hyperParams.grid.l1[i][0];
                    double l1GridEnd = hyperParams.grid.l1[i][1];
                    boolean l1NeedGrid = !(l1GridStart <= 0.0 || l1GridEnd <= 0.0);
                    int l1GridNum = l1NeedGrid ? (int)hyperParams.grid.l1[i][2] + 1 : 1;
                    hyperGridCnt *= l1GridNum;
                    double l1GridStep = (l1GridEnd - l1GridStart) / hyperParams.grid.l1[i][2];
                    l1Grid[i] = new double[l1GridNum];
                    for (int s = 0; s < l1GridNum; s++) {
                        if (l1NeedGrid) {
                            l1Grid[i][s] = l1GridStart + s * l1GridStep;
                        } else {
                            l1Grid[i][s] = 0.0;
                        }

                    }

                    LOG_UTILS.importantInfo("l1[" + i + "] search range: " + Arrays.toString(l1Grid[i]));
                }

                double [][]l2Grid = new double[l2.length][];
                for (int i = 0; i < l2.length; i++) {
                    double l2GridStart = hyperParams.grid.l2[i][0];
                    double l2GridEnd = hyperParams.grid.l2[i][1];
                    boolean l2NeedGrid = !(l2GridStart <= 0.0 || l2GridEnd <= 0.0);
                    int l2GridNum = l2NeedGrid ? (int)hyperParams.grid.l2[i][2] + 1 : 1;
                    hyperGridCnt *= l2GridNum;
                    double l2GridStep = (l2GridEnd - l2GridStart) / hyperParams.grid.l2[i][2];
                    l2Grid[i] = new double[l2GridNum];
                    for (int s = 0; s < l2GridNum; s++) {
                        if (l2NeedGrid) {
                            l2Grid[i][s] = l2GridStart + s * l2GridStep;
                        } else {
                            l2Grid[i][s] = 0.0;
                        }

                    }

                    LOG_UTILS.importantInfo("l2[" + i + "] search range: " + Arrays.toString(l2Grid[i]));
                }

                l1GridSearch = new double[hyperGridCnt][l1.length];
                l2GridSearch = new double[hyperGridCnt][l2.length];

                List<List<Double>> compositeList = new ArrayList<>();
                for (int i = 0; i < l1Grid[0].length; i++) {
                    compositeList.add(Arrays.asList(l1Grid[0][i]));
                }

                for (int i = 1; i < l1Grid.length; i++) {
                    List<List<Double>> newCompositeList = new ArrayList<>();
                    for (int j = 0; j < l1Grid[i].length; j++) {
                        for (int k = 0; k < compositeList.size(); k++) {
                            List<Double> node = new ArrayList<>();
                            node.addAll(compositeList.get(k));
                            node.add(l1Grid[i][j]);
                            newCompositeList.add(node);
                        }
                    }
                    compositeList = newCompositeList;
                }

                for (int i = 0; i < l2Grid.length; i++) {
                    List<List<Double>> newCompositeList = new ArrayList<>();
                    for (int j = 0; j < l2Grid[i].length; j++) {
                        for (int k = 0; k < compositeList.size(); k++) {
                            List<Double> node = new ArrayList<>();
                            node.addAll(compositeList.get(k));
                            node.add(l2Grid[i][j]);
                            newCompositeList.add(node);
                        }
                    }
                    compositeList = newCompositeList;
                }

                LOG_UTILS.importantInfo("grid composite result size:" + compositeList.size() + "\n" + compositeList);

                if (hyperGridCnt != compositeList.size()) {
                    LOG_UTILS.importantInfo("hyperCnt != composite size");
                    throw new Exception("hyperCnt != composite size");
                }

                for (int i = 0; i < compositeList.size(); i++) {
                    List<Double> node = compositeList.get(i);
                    for (int p = 0; p < l1.length; p++) {
                        l1GridSearch[i][p] = node.get(p);
                    }

                    for (int q = 0; q < l2.length; q++) {
                        l2GridSearch[i][q] = node.get(q + l1.length);
                    }
                }

            }

        }


        float []wprev = new float[dim];
        g = new float[dim];
        float []gprev = new float[dim];
        float []p = new float[dim];

        twoLoopsFroms = CommUtils.createThreadArrayFroms(dim, comm.getSlaveNum(), comm.getThreadNum());
        twoLoopsTos = CommUtils.createThreadArrayTos(dim, comm.getSlaveNum(), comm.getThreadNum());
        twoLoops = new TwoLoop[lbfgsParams.m];

        for (int i = 0; i < lbfgsParams.m; i++) {
            LOG_UTILS.verboseInfo("from:" + twoLoopsFroms[rank][threadIdx] + ", to:" + twoLoopsTos[rank][threadIdx]);
            twoLoops[i] = new TwoLoop(dim, twoLoopsFroms[rank][threadIdx], twoLoopsTos[rank][threadIdx]);
        }

        int status = 0;
        double[] normArr = new double[2];
        double wnorm, gnorm;
        hyper = 1;
        StringBuilder sbLog = new StringBuilder();

        while (true) {
            iter = 1;

            if (needHyperSearch) {
                if (needGridHyper) {
                    System.arraycopy(l1GridSearch[hyper - 1], 0, l1, 0, l1.length);
                    System.arraycopy(l2GridSearch[hyper - 1], 0, l2, 0, l2.length);
                }
                importantInfo(iter, "hyper search new l1:" + Arrays.toString(l1) + ", new l2:" + Arrays.toString(l2));
            }


            if (needHyperSearch && hyperParams.restart) {
                System.arraycopy(initw, 0, w, 0, dim);
            }

            // calc loss and gradient
            double loss[] = calcLossAndGrad(iter, w, g);
            lossprev = loss[1];
            purelossprev = loss[0];

            long endt = System.currentTimeMillis();
            sbLog.append((endt - start) / 1000. + " sec elapse").append("\n");
            sbLog.append("train loss = ").append(loss[0] / gWeightTrainNum).append("\n");
            sbLog.append("train regularized loss = ").append(loss[1] / gWeightTrainNum).append("\n");
            otherTrainHandle(0);
            if (lossParams.evaluate_metric.size() > 0) {
                sbLog.append(trainEvalSet.eval(getEvalObjectInfo(), "train", trainWeightAndReal)).append("\n");
            }

            // calc test loss
            if (hasTestData) {
                testloss = calTestPureLossAndGrad(w, null, iter, false);
                testloss = comm.allreduce(testloss, Operands.DOUBLE_OPERAND(), Operators.Double.SUM);
                if (needHyperSearch) {
                    if (testloss < bestTestLoss) {
                        bestTestLoss = testloss;
                        bestTrainLoss = lossprev;
                        bestPureTrainLoss = purelossprev;
                        System.arraycopy(l1, 0, bestl1, 0, l1.length);
                        System.arraycopy(l2, 0, bestl2, 0, l2.length);
                        System.arraycopy(w, 0, bestW, 0, dim);
                        bestIter = iter;
                    }
                }
                sbLog.append("test loss = ").append(testloss / gWeightTestNum).append("\n");
                otherTestHandle(0);
                if (lossParams.evaluate_metric.size() > 0) {
                    sbLog.append(testEvalSet.eval(getEvalObjectInfo(), "test", testWeightAndReal));
                }
            }
            importantInfo(0, sbLog.toString());
            sbLog.setLength(0);

            if (lossParams.just_evaluate) {
                return lossprev;
            }

            // initial hessian matrix H_0 as the identity matrix
            for (int i = 0; i < dim; i++) {
                p[i] = -g[i];
            }

            // w norm, grad norm
            normArr[0] = 0.0;
            normArr[1] = 0.0;
            if (owner && (rank == 0)) {
                normArr[0] = VectorUtils.euclideanNorm(w);
                normArr[1] =  VectorUtils.euclideanNorm(g);
            }
            normArr = comm.allreduceArray(normArr, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, normArr.length);
            wnorm = normArr[0];
            gnorm = normArr[1];

            if (wnorm < 1.0) wnorm = 1.0;
            verboseInfo(0, "gnorm:" + gnorm + ", wnorm:" + wnorm + ", gnorm / wnorm:" + gnorm / wnorm + ", eps:" + lbfgsParams.convergence.eps);

            if ((!needHyperSearch || (needHyperSearch && iter >= 2 * lbfgsParams.m)) && gnorm / wnorm <= lbfgsParams.convergence.eps) {
                importantInfo(0, "gnorm / wnorm <= lbfgsParams.convergence.eps, initial w meets converge condition, you can decrease eps to get more accurate result!" +
                        "gnorm:" + gnorm + ", wnorm:" + wnorm + ", eps:" + lbfgsParams.convergence.eps);
                status = 1;
                verboseInfo(0, "status:" + status);

                endt = System.currentTimeMillis();
                sbLog.append((endt - start) / 1000. + " sec elapse").append("\n");
                sbLog.append("final train loss = ").append(purelossprev / gWeightTrainNum).append("\n");
                sbLog.append("final train regularized loss = ").append(lossprev / gWeightTrainNum).append("\n");

                if (lossParams.evaluate_metric.size() > 0) {
                    sbLog.append(trainEvalSet.eval(getEvalObjectInfo(), "train", trainWeightAndReal)).append("\n");
                }

                if (hasTestData) {
                    sbLog.append("final test loss = ").append(testloss / gWeightTestNum).append("\n");
                    if (lossParams.evaluate_metric.size() > 0) {
                        sbLog.append(testEvalSet.eval(getEvalObjectInfo(), "test", testWeightAndReal));
                    }
                }
                importantInfo(0, sbLog.toString());
                sbLog.setLength(0);
                return lossprev;
            }

            // init step
            double step = 1.0 / gnorm;

            // loop
            int cursor = 0;
            double yy, ys;
            for (;;) {

                // exchange w and wprev
                System.arraycopy(w, 0, wprev, 0, dim);
                System.arraycopy(g, 0, gprev, 0, dim);

                // backtracking line search
                verboseInfo(iter, "begin line search...");
                int cntiter = lineSearch(iter, step, w, wprev, g, gprev, p);
                verboseInfo(iter, "finish line search iter:" + cntiter);
                if (cntiter < 0) {
                    verboseInfo(iter, "line search failed, move to prev point!");
                    System.arraycopy(wprev, 0, w, 0, dim);
                    System.arraycopy(gprev, 0, g, 0, dim);
                    status = 2;
                    break;
                }

                endt = System.currentTimeMillis();
                sbLog.append((endt - start) / 1000. + " sec elapse").append("\n");
                sbLog.append("train loss = ").append(purelossprev / gWeightTrainNum).append("\n");
                sbLog.append("train regularized loss = ").append(lossprev / gWeightTrainNum).append("\n");
                otherTrainHandle(iter);
                if (lossParams.evaluate_metric.size() > 0) {
                    sbLog.append(trainEvalSet.eval(getEvalObjectInfo(), "train", trainWeightAndReal)).append("\n");
                }

                // calc test loss
                if (hasTestData) {
                    testloss = calTestPureLossAndGrad(w, null, iter, false);
                    testloss = comm.allreduce(testloss, Operands.DOUBLE_OPERAND(), Operators.Double.SUM);
                    if (needHyperSearch) {
                        if (testloss < bestTestLoss) {
                            bestTestLoss = testloss;
                            bestTrainLoss = lossprev;
                            bestPureTrainLoss = purelossprev;
                            System.arraycopy(l1, 0, bestl1, 0, l1.length);
                            System.arraycopy(l2, 0, bestl2, 0, l2.length);
                            System.arraycopy(w, 0, bestW, 0, dim);
                            bestIter = iter;
                        }
                    }
                    sbLog.append("test loss = ").append(testloss / gWeightTestNum).append("\n");
                    otherTestHandle(iter);
                    if (lossParams.evaluate_metric.size() > 0) {
                        sbLog.append(testEvalSet.eval(getEvalObjectInfo(), "test", testWeightAndReal));
                    }
                }
                importantInfo(iter, sbLog.toString());
                sbLog.setLength(0);


                // w norm, grad norm
                normArr[0] = 0.0;
                normArr[1] = 0.0;
                if (owner && (rank == 0)) {
                    normArr[0] = VectorUtils.euclideanNorm(w);
                    normArr[1] =  VectorUtils.euclideanNorm(g);
                }
                normArr = comm.allreduceArray(normArr, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, normArr.length);
                wnorm = normArr[0];
                gnorm = normArr[1];

                verboseInfo(iter, "gnorm:" + gnorm + ", wnorm:" + wnorm + ", gnorm / wnorm:" + gnorm / wnorm + ", eps:" + lbfgsParams.convergence.eps);
                if (wnorm < 1.0) wnorm = 1.0;

                if ((!needHyperSearch || (needHyperSearch && iter >= 2 * lbfgsParams.m)) && gnorm / wnorm <= lbfgsParams.convergence.eps) {
                    importantInfo(iter, "gnorm / wnorm <= lbfgsParams.convergence.eps, converged!" +
                            "gnorm:" + gnorm + ", wnorm:" + wnorm + ", eps:" + lbfgsParams.convergence.eps + ",  you can decrease eps to get more accurate result!");
                    status = 3;
                    break;
                }

                // max iter
                if (iter >= lbfgsParams.convergence.max_iter) {
                    importantInfo(iter, "max iter,  you can increase max iter to get more accurate result!");
                    status = 4;
                    break;
                }

                // precision
                if (modelParams.dump_freq > 0 && iter % modelParams.dump_freq == 0) {
                    if (precision != null && precision.length > 0) {
                        calPrecision();
                        precision = comm.allreduceArray(precision, Operands.FLOAT_OPERAND(), Operators.Float.SUM, 0, precision.length);
                    } else {
                        importantInfo(iter, "precision is invalid!");
                    }

                    verboseInfo(iter, "begin dumping model...");
                    if ((threadIdx == 0)) {
                        dataFlow.dumpModel();
                    }
                    verboseInfo(iter, "finish dumping model!");
                }

                verboseInfo(iter, "begin two loop to calc -H*g...");

                TwoLoop twoLoop = twoLoops[cursor];
                VectorUtils.vecdiff(twoLoop.s, w, wprev, twoLoop.start);
                VectorUtils.vecdiff(twoLoop.y, g, gprev, twoLoop.start);
                ys = VectorUtils.dot(twoLoop.y, twoLoop.s);
                yy = VectorUtils.dot(twoLoop.y, twoLoop.y);
                double []yys = new double[2];
                yys[0] = ys;
                yys[1] = yy;
                yys = comm.allreduceArray(yys, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, yys.length);
                ys = yys[0];
                yy = yys[1];

                if (ys < 1.0e-60) {
                    importantInfo(iter, "ys:" + ys + " is too small or is negtive(you may change to wolfe condition!), set to 0.01*yy!");
                    ys = yy * 0.01;
                }

                twoLoop.ys = ys;
                twoLoop.yy = yy;

                int loops = Math.min(lbfgsParams.m, iter);
                //++iter;
                cursor = (cursor + 1) % lbfgsParams.m;

                // two loop -H*g
                for (int i = 0; i < dim; i++) {
                    p[i] = -g[i];
                }

                Hv(iter, p, cursor, loops, twoLoops, yy, ys);


                // Constrain the search direction
                for (int r = 0; r < l1.length; r++) {
                    if (l1[r] > 0.0) {
                        for (int i = regularStart[r]; i < regularEnd[r]; i++) {
                            if (p[i] * g[i] >= 0.0) {
                                p[i] = 0.0f;
                            }
                        }
                    }
                }

                verboseInfo(iter, "finished two loop calc -Hg!");

                step = 1.0;



                iter ++;

            }
            verboseInfo(iter, "status:" + status);


            if (needHyperSearch) {
                importantInfo(iter, "[hyper search] until now, best test loss:" + bestTestLoss +
                        ", best avg test loss:" + (bestTestLoss / gWeightTestNum) +
                        ", best l1:" + ArrayUtils.toString(bestl1) +
                        ", best l2:" + Arrays.toString(bestl2) +
                        ", best iter step:" + bestIter);

                if (precision != null && precision.length > 0) {
                    calPrecision();
                    precision = comm.allreduceArray(precision, Operands.FLOAT_OPERAND(), Operators.Float.SUM, 0, precision.length);
                } else {
                    verboseInfo(iter, "precision is invalid!");
                }

                verboseInfo(iter, "begin dumping model...");
                if ((threadIdx == 0)) {
                    dataFlow.dumpModel();
                }
                verboseInfo(iter, "finish dumping model!");

                if (needHoagHyperOpt) {
                    if (hyperHoagOptimization(cursor, hyper, iter)) {
                        break;
                    }

                    if (hyper >= hyperParams.hoag.outer_iter) {
                        break;
                    }
                } else {
                    if (hyper < l1GridSearch.length) {
                        System.arraycopy(l1GridSearch[hyper], 0, l1, 0, l1.length);
                        System.arraycopy(l2GridSearch[hyper], 0, l2, 0, l2.length);
                    }

                    if (hyper >= l1GridSearch.length) {
                        break;
                    }

                }

                hyper ++;
            } else {
                break;
            }


        }

        if (needHyperSearch) {
            System.arraycopy(bestW, 0, w, 0, dim);
            testloss = calTestPureLossAndGrad(w, null, iter, false);
            testloss = comm.allreduce(testloss, Operands.DOUBLE_OPERAND(), Operators.Double.SUM);

            double loss[] = calcLossAndGrad(iter, w, g);
            lossprev = loss[1];
            purelossprev = loss[0];
        }

        if (precision != null && precision.length > 0) {
            calPrecision();
            precision = comm.allreduceArray(precision, Operands.FLOAT_OPERAND(), Operators.Float.SUM, 0, precision.length);
        } else {
            verboseInfo(iter, "precision is invalid");
        }
        if ((threadIdx == 0)) {
            verboseInfo(iter, "begin finally dumping model...");
            dataFlow.dumpModel();
            verboseInfo(iter, "finish finally dumping model!");
        }

        long endt = System.currentTimeMillis();
        sbLog.append((endt - start) / 1000. + " sec elapse").append("\n");
        sbLog.append("final train loss = ").append(purelossprev / gWeightTrainNum).append("\n");
        sbLog.append("final train regularized loss = ").append(lossprev / gWeightTrainNum).append("\n");
        otherTrainHandle(iter);
        if (lossParams.evaluate_metric.size() > 0) {
            sbLog.append(trainEvalSet.eval(getEvalObjectInfo(), "train", trainWeightAndReal)).append("\n");
        }

        if (hasTestData) {
            sbLog.append("final test loss = ").append(testloss / gWeightTestNum).append("\n");
            otherTestHandle(iter);
            if (lossParams.evaluate_metric.size() > 0) {
                sbLog.append(testEvalSet.eval(getEvalObjectInfo(), "test", testWeightAndReal));
            }
        }

        importantInfo(iter, sbLog.toString());
        sbLog.setLength(0);

        return lossprev;

    }

    protected boolean hyperHoagOptimization(int cursor, int out, int k) throws Exception {
        importantInfo(out, "################### hoaging ###################");


        // test data loss & grad
        tLoss = calTestPureLossAndGrad(w, gtest, out, true);
        tLoss = comm.allreduce(tLoss, Operands.DOUBLE_OPERAND(), Operators.Double.SUM);
        gtest = comm.allreduceArray(gtest, Operands.FLOAT_OPERAND(), Operators.Float.SUM, 0, gtest.length);
        VectorUtils.scale(1.0 / gWeightTestNum, gtest);

        TwoLoop twoLoop = twoLoops[cursor];

        int loops = (lbfgsParams.m <= k) ? lbfgsParams.m : k;
        Hv(out, gtest, cursor, loops, twoLoops, twoLoop.yy, twoLoop.ys);

        // update hyperparams
        double []gradLambdasL2 = new double[l2.length];
        for (int r = 0; r < l2.length; r++) {
            if (l2[r] > 0.0) {
                double temp = 0.0;
                for (int j = regularStart[r]; j < regularEnd[r]; j++) {
                    temp += w[j] * gtest[j];
                }
                gradLambdasL2[r] = -l2[r] * gWeightTrainNum * temp;
            }
        }

        hoagTestLossList.add(tLoss / gWeightTestNum);
        hoagTestL2GradList.add(Arrays.asList(ArrayUtils.toObject(gradLambdasL2)));
        hoagTestLossDeltaList.add((tLoss - tOldLoss) / gWeightTestNum);
        tOldLoss = tLoss;

        if (hoagTestL2GradList.size() >= 2) {
            for (int r = 0; r < l2.length; r++) {
                if (l2[r] > 0.0) {
                    List<Double> lastGrads = hoagTestL2GradList.get(hoagTestL2GradList.size() - 1);
                    List<Double> prevGrads = hoagTestL2GradList.get(hoagTestL2GradList.size() - 2);
                    double lastgrad = lastGrads.get(r);
                    double prevgrad = prevGrads.get(r);
                    if (prevgrad * lastgrad < 0.0) {
                        hoagL2Steps[r] *= hyperParams.hoag.step_decr_factor;
                    }
                }
            }
        }
        hoagTestL2StepList.add(Arrays.asList(ArrayUtils.toObject(hoagL2Steps)));

        if (hoagTestLossDeltaList.size() >= 3) {
            double sumdelta = 0.0;
            for (int idx = hoagTestLossDeltaList.size() - 3; idx < hoagTestLossDeltaList.size(); idx++) {
                sumdelta += Math.abs(hoagTestLossDeltaList.get(idx));
            }
            sumdelta /= 3;
            if (sumdelta < hyperParams.hoag.test_loss_reduce_limit) {
                importantInfo(out, "[hoag] last 3 avg test reduce loss:" + sumdelta +
                        " < " + hyperParams.hoag.test_loss_reduce_limit + ", exit! final l2:" + Arrays.toString(l2));

                verboseInfo(out, "[hoag] test avg loss list:" + hoagTestLossList);
                verboseInfo(out, "[hoag] test l2 grad list:" + hoagTestL2GradList);
                verboseInfo(out, "[hoag] test l2 step list:" + hoagTestL2StepList);
                verboseInfo(out, "[hoag] hyper l2 list:" + hoagHyperL2List);
                verboseInfo(out, "[hoag] test avg loss delta list:" + hoagTestLossDeltaList);
                return true;
            }
        }

        verboseInfo(out, "[hoag] test avg loss list:" + hoagTestLossList);
        verboseInfo(out, "[hoag] test l2 grad list:" + hoagTestL2GradList);
        verboseInfo(out, "[hoag] test l2 step list:" + hoagTestL2StepList);
        verboseInfo(out, "[hoag] hyper l2 list:" + hoagHyperL2List);
        verboseInfo(out, "[hoag] test avg loss delta list:" + hoagTestLossDeltaList);

        for (int r = 0; r < l2.length; r++) {
            if (l2[r] > 0.0) {
                double logl2 = Math.log(l2[r]);
                if (-gradLambdasL2[r] >=0) {
                    logl2 += hoagL2Steps[r];
                } else {
                    logl2 -= hoagL2Steps[r];
                }
                l2[r] = Math.exp(logl2);
            }
        }
        hoagHyperL2List.add(Arrays.asList(ArrayUtils.toObject(l2)));

        importantInfo(out, "[hoag] l1:" + Arrays.toString(l1) + ", new l2:" + Arrays.toString(l2));


        return false;
    }

    protected void Hv(int iter, float []p, int cursor, int loops, TwoLoop twoLoops [], double yy, double ys) throws Mp4jException {

        double beta;
        for (int i = 0; i < loops; ++i) {
            cursor = (cursor + lbfgsParams.m - 1) % lbfgsParams.m;
            TwoLoop tl = twoLoops[cursor];
            tl.alpha = VectorUtils.dot(tl.s, p, tl.start);
            tl.alpha = comm.allreduce(tl.alpha, Operands.DOUBLE_OPERAND(), Operators.Double.SUM);
            tl.alpha /= tl.ys;
            VectorUtils.daxpy(-tl.alpha, tl.y, p, tl.start);
        }

        p = comm.allgatherArray(p, Operands.FLOAT_OPERAND(), twoLoopsFroms, twoLoopsTos);


        VectorUtils.scale(ys / yy, p);
        for (int i = 0; i < loops; ++i) {
            TwoLoop tl = twoLoops[cursor];
            beta = VectorUtils.dot(tl.y, p, tl.start);
            beta = comm.allreduce(beta, Operands.DOUBLE_OPERAND(), Operators.Double.SUM);
            beta /= tl.ys;
            VectorUtils.daxpy(tl.alpha - beta, tl.s, p, tl.start);
            cursor = (cursor + 1) % lbfgsParams.m;
        }
        comm.allgatherArray(p, Operands.FLOAT_OPERAND(), twoLoopsFroms, twoLoopsTos);
    }

    protected String extraInfo() {
        return "";
    }

    protected void importantInfo(int iter, String info, boolean onlyRank0Thread0) throws Mp4jException {
        if (!needHyperSearch) {
            LOG_UTILS.importantInfo(String.format("[model=%s] [loss=%s] [iter=%d] %s%s", modelName, lossFunction.getName(), iter, extraInfo(), info), onlyRank0Thread0);
        } else {
            LOG_UTILS.importantInfo(String.format("[model=%s] [loss=%s] [hyper=%d] [iter=%d] %s%s", modelName, lossFunction.getName(), hyper, iter, extraInfo(), info), onlyRank0Thread0);
        }
    }

    protected void importantInfo(int iter, String info) throws Mp4jException {
        importantInfo(iter, info, true);
    }

    protected void verboseInfo(int iter, String info, boolean onlyRank0Thread0) throws Mp4jException {
        if (!needHyperSearch) {
            LOG_UTILS.verboseInfo(String.format("[model=%s] [loss=%s] [iter=%d] %s%s", modelName, lossFunction.getName(), iter, extraInfo(), info), onlyRank0Thread0);
        } else {
            LOG_UTILS.verboseInfo(String.format("[model=%s] [loss=%s] [hyper=%d] [iter=%d] %s%s", modelName, lossFunction.getName(), hyper, iter, extraInfo(), info), onlyRank0Thread0);
        }
    }

    protected void verboseInfo(int iter, String info) throws Mp4jException {
        verboseInfo(iter, info, true);
    }

    protected void error(int iter, String error) throws Mp4jException {
        LOG_UTILS.error(error);
    }

    protected void exception(Exception e) throws Mp4jException {
        LOG_UTILS.exception(e);
    }

    public String getSomeParams(float []v, String prefix, int num) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < num; i++) {
            sb.append(prefix + "[" + i + "]=" + v[i] + ", ");
        }

        return sb.toString();
    }

    public abstract double calcPureLossAndGrad(float []w, float []g, int iter) throws Exception;

    private double[] calcLossAndGrad(int iter, float []w, float []g) throws Exception {

        // pure loss & grad, exclude regularization
        double loss = calcPureLossAndGrad(w, g, iter);
        double allloss = loss;

        // f = 0.5 * l2 * ||w||^2, excluded bias
        for (int r = 0; r < l2.length; r++) {
            if (l2[r] > 0.0) {
                double l2regularLoss = 0.0;
                for (int i = regularStart[r]; i < regularEnd[r]; i++) {
                    l2regularLoss += w[i] * w[i];
                }

                l2regularLoss *= 0.5 * l2[r] * tWeightTrainNum;
                allloss += l2regularLoss;
            }

            // f = l1 * ||w||^1, excluded bias
            if (l1[r] > 0.0) {
                double l1regularLoss = 0.0;
                for (int i = regularStart[r]; i < regularEnd[r]; i++) {
                    l1regularLoss += Math.abs(w[i]);
                }

                l1regularLoss *= l1[r] * tWeightTrainNum;
                allloss += l1regularLoss;
            }
        }

        double []retloss = new double[20];
        retloss[0] = loss;
        retloss[1] = allloss;

        // loss reduce
        verboseInfo(iter, "retloss array size:" + retloss.length + ", precision is null:" + (precision == null));
        retloss = comm.allreduceArray(retloss, Operands.DOUBLE_OPERAND(), Operators.Double.SUM, 0, retloss.length);

        for (int r = 0; r < l2.length; r++) {
            // l2 gradient
            if (l2[r] > 0.0) {
                for (int i = regularStart[r]; i < regularEnd[r]; i++) {
                    g[i] += tWeightTrainNum * l2[r] * w[i];
                }
            }

            // l1 subgradient
            if (l1[r] > 0.0) {
                for (int i = regularStart[r]; i < regularEnd[r]; i++) {
                    if (w[i] != 0.0) {
                        g[i] += tWeightTrainNum * l1[r] * Math.signum(w[i]);
                    } else {
                        g[i] += tWeightTrainNum * l1[r];
                    }
                }
            }
        }


        // grad reduce
        g = comm.allreduceArray(g, Operands.FLOAT_OPERAND(), Operators.Float.SUM, 0, g.length);

        for (int r = 0; r < l2.length; r++) {
            if (l1[r] > 0.0) {
                double partPos, partNeg;
                for (int i = regularStart[r]; i < regularEnd[r]; i++) {
                    if (w[i] != 0) {
                        partPos = g[i];
                        partNeg = partPos;
                    } else {
                        partPos = g[i];
                        partNeg = partPos - 2 * gWeightTrainNum * l1[r];
                    }

                    if (partNeg > 0.0) {
                        g[i] = (float)partNeg;
                    } else if (partPos < 0.0) {
                        g[i] = (float)partPos;
                    } else {
                        g[i] = 0.0f;
                    }

                }
            }
        }

        return retloss;
    }


    private int lineSearch(int iter, double step, float []w, float []wprev, float []g, float []gprev, float []p) throws Exception {
        int lsIter = 0;
        double prevloss = lossprev;
        double prevpureloss = purelossprev;
        double factor;


        // g(w)^T * p
        double dginit = 0;
        if (owner && rank == 0) {
            dginit = VectorUtils.dot(g, p);
        }
        dginit = comm.allreduce(dginit, Operands.DOUBLE_OPERAND(), Operators.Double.SUM);

        for (;;) {
            verboseInfo(iter, "----line search iter:" + lsIter + ", step:" + step + ", type:" + mode);
            // w = w + step * p
            System.arraycopy(wprev, 0, w, 0, dim);
            VectorUtils.daxpy(step, p, w);

            // orthant
            for (int r = 0; r < l1.length; r++) {
                if (l1[r] > 0.0) {
                    for (int i = regularStart[r]; i < regularEnd[r]; i++) {
                        if (wprev[i] != 0) {
                            if (w[i] * wprev[i] <= 0.0) {
                                w[i] = 0.0f;
                            }
                        } else {
                            if (w[i] * gprev[i] >= 0.0) {
                                w[i] = 0.0f;
                            }
                        }
                    }
                }
            }

            // eval new loss and grad
            double []loss = calcLossAndGrad(iter, w, g);
            lossprev = loss[1];
            purelossprev = loss[0];

            verboseInfo(iter, "----inner line search iter:" + lsIter + ", step:" + step +
                    ", loss:" + loss[1] + ", avg loss:" + loss[1] / gWeightTrainNum +
                    ",  pure loss:" + loss[0] + ", avg pure loss:" + loss[0] / gWeightTrainNum +
                    ", reduce loss:" + (prevloss - loss[1]) +
                    ", avg reduce loss:" + (prevloss - loss[1]) / gWeightTrainNum +
                    ", train data num=" + gWeightTrainNum);

            lsIter ++;

            // a * g(w)^T * p
            double dgtest = 0.;
            if (owner && rank == 0) {
                for (int i = 0; i < dim; i++) {
                    dgtest += (w[i] - wprev[i]) * gprev[i];
                }
            }
            dgtest = comm.allreduce(dgtest, Operands.DOUBLE_OPERAND(), Operators.Double.SUM);

            if (loss[1] > prevloss + backTrackingParams.c1 * dgtest) {
                factor = backTrackingParams.step_decr;
                verboseInfo(iter, "----sufficient decrease condition failed! step will decrease, iter:" + lsIter + ", step:" + step);
            } else {
                // sufficient decrease condition, f(w+a*p) <= f(w) + c1 * a * g(w)^T * p
                if (mode == LineSearchParams.Mode.SUFFICIENT_DECREASE) {
                    verboseInfo(iter, "----sufficient decrease condition meet! iter:" + lsIter +
                            ", step:" + step + ", actred:" + (prevloss - loss[1]) +
                            " >= c1*alpha*g*p:" + (-backTrackingParams.c1 * dgtest));
                    return lsIter;
                }

                // g(w + a*p)^T * p
                double dg = 0.0;
                if (owner && rank == 0) {
                    dg = VectorUtils.dot(p, g);
                }
                dg = comm.allreduce(dg, Operands.DOUBLE_OPERAND(), Operators.Double.SUM);

                if (dg < backTrackingParams.c2 * dginit) {
                    factor = backTrackingParams.step_incr;
                    verboseInfo(iter, "----wolfe condition failed! step will increase, iter:" + lsIter + ", step:" + step);
                } else {
                    // g(w + a*p)^T * p >= c2 * g(w)^T * p
                    if (mode == LineSearchParams.Mode.WOLFE) {
                        verboseInfo(iter, "----wolfe condition meet! iter:" + lsIter +
                                        ", step:" + step + ", actred:" + (prevloss - loss[1]) +
                                        ", g(w + a*p)^T * p:" + dg + " >= c2 * g(w)^T * p:" +
                                        backTrackingParams.c2 * dginit
                        );
                        return lsIter;
                    }

                    if (dg > -backTrackingParams.c2 * dginit) {
                        factor = backTrackingParams.step_decr;
                        verboseInfo(iter, "----strong wolfe condition failed! step will decrease, iter:" + lsIter + ", step:" + step);
                    } else {
                        // |g(w + a*p)^T * p| <= |c2 * g(w)^T * p|
                        verboseInfo(iter, "----strong wolfe condition meet! iter:" + lsIter +
                                        ", step:" + step + ", actred:" + (prevloss - loss[1]) +
                                        ", |g(w + a*p)^T * p|:" + Math.abs(dg) + " <= c2 * |g(w)^T * p|:" +
                                        backTrackingParams.c2 * Math.abs(dginit)
                        );
                        return lsIter;
                    }

                }

            }

            if (step < backTrackingParams.min_step) {
                lossprev = prevloss;
                purelossprev = prevpureloss;
                importantInfo(iter, "----line search step is too small:" + step + ", optimizer will abort!");
                return -1;
            }

            if (step > backTrackingParams.max_step) {
                lossprev = prevloss;
                purelossprev = prevpureloss;
                importantInfo(iter, "----line search step is too large:" + step + ", optimizer will abort!");
                return -2;
            }

            if (backTrackingParams.max_iter <= lsIter) {
                lossprev = prevloss;
                purelossprev = prevpureloss;
                importantInfo(iter, "----line search iter >= max line search iter! step:" + step + ", optimizer will abort!");
                return -3;
            }
            step *= factor;
        }

    }

    public double calTestPureLossAndGrad(float []wtest, float []gtest, int iter, boolean needCalcGrad) throws Mp4jException {
        return -1;
    }

    public void calPrecision() {}

}
