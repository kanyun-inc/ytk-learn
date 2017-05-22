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

package com.fenbi.ytklearn.predictor;

import com.fenbi.ytklearn.dataflow.GBMLRDataFlow;
import com.fenbi.ytklearn.dataflow.PredictCoreData;
import com.fenbi.ytklearn.eval.ConfusionMatrixEvaluator;
import com.fenbi.ytklearn.eval.EvalSet;
import com.fenbi.ytklearn.exception.YtkLearnException;
import com.fenbi.ytklearn.loss.LossFunctions;
import com.fenbi.ytklearn.loss.ILossFunction;
import com.fenbi.ytklearn.dataflow.DataUtils;
import com.fenbi.ytklearn.data.gbdt.GBDTModel;
import com.fenbi.ytklearn.loss.SigmoidFunction;
import com.fenbi.ytklearn.loss.SoftmaxFunction;
import com.fenbi.ytklearn.param.gbdt.GBDTDataParams;
import com.fenbi.ytklearn.data.gbdt.Tree;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.typesafe.config.ConfigException;
import org.python.core.PyByteArray;
import org.python.core.PyFunction;
import org.python.core.PyList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

/**
 * @author wufan
 * @author xialong
 */

public class GBDTOnlinePredictor extends OnlinePredictor implements ITreePredictor {
    public static final Logger LOG = LoggerFactory.getLogger(GBDTOnlinePredictor.class);

    private final ThreadLocal<double[]> predbuffer = new ThreadLocal<>();
    private final ThreadLocal<double[]> scoreBuffer = new ThreadLocal<>();
    private final ThreadLocal<double[]> leafBuffer = new ThreadLocal<>();

    private final GBDTDataParams dataParams;
    private final String modelDataPath;
    private final boolean sampleDepBasePrediction;

    private int useRoundNum;
    protected GBMLRDataFlow.Type learnType;

    // model inner data
    private List<Tree> trees;
    private float baseScore;
    private int numTreeInGroup;
    private ILossFunction lossFunction;

    public GBDTOnlinePredictor(String configPath) throws Exception {
        super(configPath);
        this.dataParams = new GBDTDataParams(config, "");
        this.modelDataPath = config.getString("model.data_path");
        this.sampleDepBasePrediction = config.getBoolean("optimization.sample_dependent_base_prediction");
        try {
            String typeStr = config.getString("type");
            learnType = GBMLRDataFlow.Type.getType(typeStr);
            CheckUtils.check(learnType != null, "[GBDT] learn type(%s) invalid", typeStr);
        } catch (ConfigException.Missing e) {
            learnType = GBMLRDataFlow.Type.GB;
        }

        try {
            this.useRoundNum = config.getInt("optimization.round_num");
        } catch (ConfigException.Missing e) {
            this.useRoundNum = -1;
        }

        loadModel();
    }

    public GBDTOnlinePredictor(Reader configReader) throws Exception {
        super(configReader);
        this.dataParams = new GBDTDataParams(config, "");
        this.modelDataPath = config.getString("model.data_path");
        this.sampleDepBasePrediction = config.getBoolean("optimization.sample_dependent_base_prediction");
        try {
            String typeStr = config.getString("type");
            learnType = GBMLRDataFlow.Type.getType(typeStr);
            CheckUtils.check(learnType != null, "[GBDT] learn type(%s) invalid", typeStr);
        } catch (ConfigException.Missing e) {
            learnType = GBMLRDataFlow.Type.GB;
        }

        try {
            this.useRoundNum = config.getInt("optimization.round_num");
        } catch (ConfigException.Missing e) {
            this.useRoundNum = -1;
        }

        loadModel();
    }

    private Object getEvalObjectInfo() {
        if (lossFunction instanceof SigmoidFunction) {
            return new ConfusionMatrixEvaluator.ConfusionMatrixInfo(2, false);
        } else if (lossFunction instanceof SoftmaxFunction) {
            return new ConfusionMatrixEvaluator.ConfusionMatrixInfo(numTreeInGroup, true);
        } else {
            return null;
        }
    }

    @Override
    protected OnlinePredictor loadModel() throws Exception {
        // check model data path
        if (!fs.exists(modelDataPath)) {
            throw new Exception("gbdt model doesn't exist! path:" + modelDataPath);
        }

        GBDTModel model = new GBDTModel();
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(fs.getReader(modelDataPath));
            model.loadModel(reader);
        } finally {
            if (reader != null) {
                reader.close();
            }
        }

        this.trees = model.trees;
        this.numTreeInGroup = model.numTreeInGroup;
        this.lossFunction = LossFunctions.createLossFunction(model.objName);
        this.baseScore = (float) lossFunction.pred2Score(model.basePrediction);

        LOG.info(String.format("numClass=%d, useRoundNum=%d, totalRoundNum=%d", numTreeInGroup, useRoundNum, trees.size()));

        CheckUtils.check(trees.size() > 0 && trees.size() % numTreeInGroup == 0,
                String.format("[GBDT] model error, treeNum=%d, numClass=%d", trees.size(), numTreeInGroup));

        int roundNum = trees.size() / numTreeInGroup;
        if (useRoundNum <= 0) {
            useRoundNum = roundNum;
        }
        CheckUtils.check(useRoundNum <= roundNum,
                String.format("[GBDT] param error, use round num=%d, but tree only has %d round", useRoundNum, roundNum));

        LOG.info(String.format("GBDT load model finished, " + model.toString()));
        return this;
    }

    // called by predict0, numTreeInGroup = 1, other is margin
    @Override
    public double score(Map<String, Float> features, Object other) {
        double score = 0;
        for (int i = 0; i < useRoundNum; i++) {
            score += trees.get(i).predict(features);
        }

        if (learnType == GBMLRDataFlow.Type.RF) {
            score /= useRoundNum;
        }
        score += baseScore;
        score = (other == null) ? score : score + lossFunction.pred2Score(((Float) other).floatValue());
        return score;
    }

    // called by predict1, multiclass, numTreeInGroup > 1
    @Override
    public double[] scores(Map<String, Float> features, Object other) {
        double[] scores = scoreBuffer.get();
        if (scores == null) {
            scores = new double[numTreeInGroup];
            scoreBuffer.set(scores);
        }

        for (int k = 0; k < numTreeInGroup; k++) {
            scores[k] = 0;
        }
        for (int i = 0; i < useRoundNum; i++) {
            for (int k = 0; k < numTreeInGroup; k++) {
                scores[k] += trees.get(i * numTreeInGroup + k).predict(features);
            }
        }

        for (int k = 0; k < numTreeInGroup; k++) {
            if (learnType == GBMLRDataFlow.Type.RF) {
                scores[k] /= useRoundNum;
            }
            scores[k] += baseScore;
        }

        if (other != null) {
            Float[] initScore = (Float[]) other;
            CheckUtils.check(initScore.length == numTreeInGroup,
                    "[GBDT] sample dependent score num must equal %d", numTreeInGroup);
            for (int k = 0; k < numTreeInGroup; k++) {
                // pred2Score is useless in fact, softmax has no inverse conversion
                scores[k] += lossFunction.pred2Score(initScore[k]);
            }
        }
        return scores;
    }

    @Override
    // numTreeGroup = 1, for regression and binary classification
    public double loss(Map<String, Float> features, double label, Object other) {
        return lossFunction.loss(score(features, other), label);
    }

    @Override
    // numTreeInGroup > 1, for multi-class classification
    public double loss(Map<String, Float> features, double[] labels, Object other) {
        return lossFunction.loss(scores(features, other), labels);
    }

    @Override
    // numTreeInGroup = 1, for regression and binary classification
    public double predict(Map<String, Float> features, Object other) {
        return lossFunction.predict(score(features, other));
    }

    @Override
    // numTreeInGroup >= 1, for regression, binary classification and multi-class classification
    public double[] predicts(Map<String, Float> features, Object other) {
        double[] pred = predbuffer.get();
        if (pred == null) {
            pred = new double[numTreeInGroup];
            predbuffer.set(pred);
        }
        if (numTreeInGroup == 1) {
            pred[0] = lossFunction.predict(score(features, other));
        } else {
            lossFunction.predict(scores(features, other), pred);
        }
        return pred;

    }

    // predict leaf index
    @Override
    public double[] predictLeaf(Map<String, Float> features) {
        double[] leafIndex = leafBuffer.get();
        if (leafIndex == null) {
            leafIndex = new double[numTreeInGroup * useRoundNum];
            leafBuffer.set(leafIndex);
        }

        for (int i = 0; i < leafIndex.length; i++) {
            leafIndex[i] = trees.get(i).getLeafIndex(features);
        }
        return leafIndex;
    }

    private static PyList transform(String line, PyFunction pyTransformFunc) throws UnsupportedEncodingException {
        return (PyList) pyTransformFunc.__call__(new PyByteArray(line.getBytes("utf-8")));
    }

    private static Iterator nextSamples(String line, boolean needPyTransform, PyFunction pyTransformFunc, List<String> lineList) throws UnsupportedEncodingException {
        Iterator iter;
        if (needPyTransform) {
            iter = transform(line, pyTransformFunc).iterator();
        } else {
            lineList.set(0, line);
            iter = lineList.iterator();
        }

        return iter;
    }

    @Override
    public double batchPredictFromFiles(String modelName,
                                        String fileDir,
                                        boolean needPyTransform,
                                        String pyTransformScript,
                                        String resultSaveMode,
                                        String resultFileSuffix,
                                        int maxErrorTol,
                                        String evalMetricStr,
                                        String predictTypeStr) throws Exception {

        PredictType predictType = PredictType.valueOfType(predictTypeStr);
        String predictAsFeatPrefix = "_label_";
        if (predictType == PredictType.PREDICT_LEAFID) {
            predictAsFeatPrefix = "_tree_leaf_";
        }

        // save mode
        ResultSaveMode saveMode = ResultSaveMode.valueOf(resultSaveMode);
        PredictCoreData testData = null;
        boolean needEval = true;
        if (evalMetricStr == null || evalMetricStr.length() == 0) {
            needEval = false;
        }

        PyFunction pyTransformFunc = DataUtils.getTranformFunction(needPyTransform, pyTransformScript);

        List<String> lineList = new ArrayList<>();
        lineList.add("temp");

        List<String> paths = fs.recurGetPaths(Arrays.asList(fileDir));

        String line;
        double[] labels = null;
        Float[] otherinfo = null;
        double loss = 0.0;
        boolean hasLabel = false;

        long realcnt = 0;
        double weightCnt = 0.0;
        int errorNum = 0;
        for (String path : paths) {
            String predictPath = path + resultFileSuffix;
            BufferedReader reader = new BufferedReader(fs.getReader(path));
            PrintWriter writer = new PrintWriter(fs.getWriter(predictPath));
            LOG.info("predict path:" + path);
            LOG.info("predict result path:" + predictPath);
            while ((line = reader.readLine()) != null) {
                Iterator<String> it = nextSamples(line, needPyTransform, pyTransformFunc, lineList);
                while (it.hasNext()) {
                    line = it.next();

                    try {
                        String xsplits[] = line.split(dataParams.delim.x_delim);
                        float weight = Float.parseFloat(xsplits[0]);

                        String[] kvs = xsplits[2].split(dataParams.delim.features_delim);
                        Map<String, Float> fmap = new HashMap<>(kvs.length);
                        for (String kv : kvs) {
                            String[] kvsplit = kv.split(dataParams.delim.feature_name_val_delim);
                            fmap.put(kvsplit[0], Float.parseFloat(kvsplit[1]));
                        }

                        hasLabel |= (xsplits[1].trim().length() > 0);
                        String[] linfo = xsplits[1].split(dataParams.delim.y_delim);

                        if (!hasLabel) {
                            if (saveMode == ResultSaveMode.LABEL_AND_PREDICT ||
                                    saveMode == ResultSaveMode.PREDICT_AS_FEATURE) {
                                throw new YtkLearnException("sample has no label:" + line);

                            }
                        }

                        if (sampleDepBasePrediction) {
                            String[] oinfo = xsplits[3].split(dataParams.delim.y_delim);
                            CheckUtils.check(linfo.length == numTreeInGroup,
                                    "[GBDT] sample dependent score num must equal %d, %s", numTreeInGroup, line);
                            if (otherinfo == null) {
                                otherinfo = new Float[oinfo.length];
                            }
                            for (int i = 0; i < oinfo.length; i++) {
                                otherinfo[i] =  Float.parseFloat(oinfo[i]);
                            }
                        }

                        double[] predicts;
                        if (numTreeInGroup == 1) {
                            predicts = predicts(fmap, sampleDepBasePrediction? otherinfo[0]: null);
                        } else {
                            predicts = predicts(fmap, otherinfo);
                        }

                        if (hasLabel) {
                            if (labels == null) {
                                labels = new double[numTreeInGroup];
                            }

                            if (numTreeInGroup == 1) {
                                labels[0] = Float.parseFloat(linfo[0]);
                                loss += weight * loss(fmap, labels[0], sampleDepBasePrediction? otherinfo[0]: null);

                            } else {
                                if (linfo.length == 1) {
                                    for (int i = 0; i < numTreeInGroup; i++) {
                                        labels[i] = 0;
                                    }
                                    int clazz = Integer.parseInt(linfo[0]);
                                    if (clazz >= numTreeInGroup) {
                                        throw new YtkLearnException("multi classification label must in range [0,K-1]!\n" + line);
                                    }
                                    labels[clazz] = 1.0f;
                                } else if (linfo.length == numTreeInGroup){
                                    for (int i = 0; i < numTreeInGroup; i++) {
                                        labels[i] = Float.parseFloat(linfo[i]);
                                    }
                                } else {
                                    throw new YtkLearnException("label format error:" + line);
                                }
                                loss += weight * loss(fmap, labels, otherinfo);
                            }

                            if (needEval) {
                                if (testData == null) {
                                    testData = new PredictCoreData(null, numTreeInGroup);
                                }
                                testData.addPredict(predicts, labels, weight);
                            }
                        }

                        if (saveMode == ResultSaveMode.PREDICT_RESULT_ONLY ||
                                saveMode == ResultSaveMode.LABEL_AND_PREDICT ) {
                            if (predictType == PredictType.PREDICT_LEAFID) {
                                predicts = predictLeaf(fmap);
                            }
                            StringBuilder sb = new StringBuilder();
                            for (int i = 0; i < predicts.length - 1; i++) {
                                sb.append(predicts[i]).append(dataParams.delim.y_delim);
                            }
                            sb.append(predicts[predicts.length - 1]);
                            if (saveMode == ResultSaveMode.PREDICT_RESULT_ONLY) {
                                writer.println(sb.toString());
                            } else {
                                writer.println(xsplits[1] + dataParams.delim.x_delim + sb.toString());
                            }

                        } else if (saveMode == ResultSaveMode.PREDICT_AS_FEATURE) {
                            if (predictType == PredictType.PREDICT_LEAFID) {
                                predicts = predictLeaf(fmap);
                            }
                            StringBuilder sbx = new StringBuilder();
                            for (int i = 0; i < predicts.length - 1; i++) {
                                sbx.append(modelName).append(predictAsFeatPrefix).append(i).append(dataParams.delim.feature_name_val_delim).
                                        append(predicts[i]).append(dataParams.delim.features_delim);
                            }
                            sbx.append(modelName).append(predictAsFeatPrefix).append(predicts.length - 1).
                                    append(dataParams.delim.feature_name_val_delim).append(predicts[predicts.length - 1]);
                            writer.println(xsplits[0] + dataParams.delim.x_delim + xsplits[1] + dataParams.delim.x_delim +
                                    xsplits[2] + dataParams.delim.features_delim +
                                    sbx.toString());

                         } else {
                            throw new YtkLearnException(modelName + " doesn't support save mode:" + saveMode);
                         }

                        realcnt++;
                        weightCnt += weight;
                    } catch (Exception e) {
                        errorNum++;
                        LOG.error("[ERROR] error format:" + line + "\n" +
                                ", local error total number:" + errorNum +
                                ", max error tol:" + maxErrorTol +
                                ", has read lines:" + realcnt);
                        if (errorNum > maxErrorTol) {
                            LOG.error("[ERROR] error number:" + errorNum +
                                    " > " + "max tol:" + maxErrorTol, e);
                            throw e;
                        }
                    }
                }
            }
            reader.close();
            writer.close();
            }

            LOG.info("error data format line number:" + errorNum);

        if (hasLabel) {
            loss /= weightCnt;
            LOG.info("loss:" + loss + ", sample number:" + realcnt + ", sample weight sum:" + weightCnt);
            if (needEval) {
                testData.endAddData();
                EvalSet evalSet = new EvalSet(null, testData);
                evalSet.addEvals(Arrays.asList(evalMetricStr.split(",")));
                LOG.info("evaluation results:\n" + evalSet.eval(getEvalObjectInfo(), "", testData.needWeightAndReal()));
            }
        }
        LOG.info("predict complete!");
        return hasLabel ? loss : Double.NaN;
    }

    public static void main(String[] args) {
        Float[] otherinfo = new Float[]{1.f, 2.f, 3.f};
        Float[] margin = (Float[]) otherinfo;
    }

}
