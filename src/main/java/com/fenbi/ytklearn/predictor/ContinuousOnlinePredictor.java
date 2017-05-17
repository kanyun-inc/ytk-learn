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

import com.fenbi.ytklearn.dataflow.CoreData;
import com.fenbi.ytklearn.dataflow.DataFlow;
import com.fenbi.ytklearn.dataflow.PredictCoreData;
import com.fenbi.ytklearn.eval.ConfusionMatrixEvaluator;
import com.fenbi.ytklearn.eval.EvalSet;
import com.fenbi.ytklearn.exception.YtkLearnException;
import com.fenbi.ytklearn.loss.LossFunctions;
import com.fenbi.ytklearn.loss.ILossFunction;
import com.fenbi.ytklearn.dataflow.DataUtils;
import com.fenbi.ytklearn.feature.FeatureHash;
import com.fenbi.ytklearn.param.CommonParams;
import com.fenbi.ytklearn.param.DataParams;
import com.fenbi.ytklearn.param.FeatureParams;
import com.fenbi.ytklearn.param.ModelParams;
import org.python.core.PyByteArray;
import org.python.core.PyFunction;
import org.python.core.PyList;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
import java.util.*;

/**
 * @author xialong
 */

public abstract class ContinuousOnlinePredictor<T> extends OnlinePredictor {
    protected final CommonParams commonParams;
    protected final DataParams dataParams;
    protected final ModelParams modelParams;
    protected final FeatureParams featureParams;
    protected final ILossFunction lossFunction;
    protected final Map<String, T> modelMap = new HashMap<>();
    protected FeatureHash featureHash;

    protected final boolean needFeatureTransform;
    protected Map<String, CoreData.TransformNode> transformNodeMap = new HashMap<>();

    public ContinuousOnlinePredictor(String configPath) throws Exception {
        super(configPath);

        this.commonParams = CommonParams.loadParams(config);
        this.dataParams = commonParams.dataParams;
        this.modelParams = commonParams.modelParams;
        featureParams = commonParams.featureParams;
        this.featureHash = FeatureHash.build(commonParams.featureParams.feature_hash.bucket_size,
                commonParams.featureParams.feature_hash.seed, commonParams.featureParams.feature_hash.feature_prefix)
                .withDelim(commonParams.dataParams.delim.features_delim, commonParams.dataParams.delim.feature_name_val_delim);
        this.lossFunction = LossFunctions.createLossFunction(commonParams.lossParams.loss_function);

        this.needFeatureTransform = featureParams != null && featureParams.transform.switch_on;
        if (needFeatureTransform) {
            String statPath = modelParams.data_path + DataFlow.FEATURE_TRANSFORM_STAT_PATH_SUFFIX;
            BufferedReader reader = new BufferedReader(fs.getReader(statPath));
            String line;
            while((line = reader.readLine()) != null) {
                String []info = line.split("###");
                String fn = info[0].trim();
                CoreData.TransformNode node = CoreData.TransformNode.fromString(info[1].trim());
                transformNodeMap.put(fn, node);
            }
            reader.close();
        }

        LOG.info("feature transform stat:" + transformNodeMap);
    }

    public ContinuousOnlinePredictor(Reader configReader) throws Exception {
        super(configReader);

        this.commonParams = CommonParams.loadParams(config);
        this.dataParams = commonParams.dataParams;
        this.modelParams = commonParams.modelParams;
        featureParams = commonParams.featureParams;
        this.featureHash = FeatureHash.build(commonParams.featureParams.feature_hash.bucket_size,
                commonParams.featureParams.feature_hash.seed, commonParams.featureParams.feature_hash.feature_prefix)
                .withDelim(commonParams.dataParams.delim.features_delim, commonParams.dataParams.delim.feature_name_val_delim);
        this.lossFunction = LossFunctions.createLossFunction(commonParams.lossParams.loss_function);

        this.needFeatureTransform = featureParams != null && featureParams.transform.switch_on;
        if (needFeatureTransform) {
            String statPath = modelParams.data_path + DataFlow.FEATURE_TRANSFORM_STAT_PATH_SUFFIX;
            BufferedReader reader = new BufferedReader(fs.getReader(statPath));
            String line;
            while((line = reader.readLine()) != null) {
                String []info = line.split("###");
                String fn = info[0].trim();
                CoreData.TransformNode node = CoreData.TransformNode.fromString(info[1].trim());
                transformNodeMap.put(fn, node);
            }
            reader.close();
        }

        LOG.info("feature transform stat:" + transformNodeMap);
    }

    public Object getEvalObjectInfo() {
        if (LossFunctions.pureClassification(commonParams.lossParams.loss_function)) {
            return new ConfusionMatrixEvaluator.ConfusionMatrixInfo(2, false);
        } else {
            return null;
        }
    }


    public float transform(String fn, float val) {
        if (!needFeatureTransform)
            return val;

        CoreData.TransformNode node = transformNodeMap.get(fn);
        if (node == null) {
            return 0.0f;
        }
        return node.transform(val);
    }

    @Override
    public double predict(Map<String, Float> features, Object other) {
        return lossFunction.predict(score(features, other));
    }

    @Override
    public double loss(Map<String, Float> features, double label, Object other) {
        return lossFunction.loss(score(features, other), label);
    }

    private static PyList transform(String line, PyFunction pyTransformFunc) throws UnsupportedEncodingException {
        return (PyList) pyTransformFunc.__call__(new PyByteArray(line.getBytes("utf-8")));
    }

    private static Iterator nextSamples(String line, boolean needPyTransform, PyFunction pyTransformFunc, List<String>lineList) throws UnsupportedEncodingException {
        Iterator iter;
        if (needPyTransform) {
            iter = transform(line, pyTransformFunc).iterator();
        } else {
            lineList.set(0, line);
            iter = lineList.iterator();
        }

        return iter;
    }

    private boolean isGbst(String modelName) {
        if (modelName.equalsIgnoreCase("gbmlr") || modelName.equalsIgnoreCase("gbsdt") ||
                modelName.equalsIgnoreCase("gbhmlr") || modelName.equalsIgnoreCase("gbhsdt")) {
            return true;
        }

        return false;
    }

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
        if (predictType == PredictType.PREDICT_LEAFID) {
            if (!isGbst(modelName)) {
                throw new YtkLearnException(modelName + " do not support predict type:" + predictTypeStr);
            }
        }

        // save mode
        ResultSaveMode saveMode = ResultSaveMode.valueOf(resultSaveMode);
        PredictCoreData testData = null;
        boolean needEval = true;
        if (evalMetricStr == null || evalMetricStr.length() == 0) {
            needEval = false;
        }
        
        PyFunction pyTransformFunc = DataUtils.getTranformFunction(needPyTransform, pyTransformScript);

        List<String>lineList = new ArrayList<>();
        lineList.add("temp");

        List<String> paths = fs.recurGetPaths(Arrays.asList(fileDir));

        String line;
        double []labels = null;
        float label = -Float.MAX_VALUE;
        double loss = 0.0;
        boolean hasLabel = false;

        int realcnt = 0;
        double weightCnt = 0;
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

                        String []kvs = xsplits[2].split(dataParams.delim.features_delim);
                        Map<String, Float> fmap = new HashMap<>(kvs.length);
                        for (String kv : kvs) {
                            String []kvsplit = kv.split(dataParams.delim.feature_name_val_delim);
                            fmap.put(kvsplit[0], Float.parseFloat(kvsplit[1]));
                        }

                        hasLabel |= (xsplits[1].trim().length() > 0);
                        String []linfo = xsplits[1].split(dataParams.delim.y_delim);

                        if (!hasLabel) {
                            if (saveMode == ResultSaveMode.LABEL_AND_PREDICT ||
                                    saveMode == ResultSaveMode.PREDICT_AS_FEATURE) {
                                throw new YtkLearnException("sample has no label:" + line);
                            }
                        }

                        if (modelName.equalsIgnoreCase("multiclass_linear")) {
                            double []predicts = predicts(fmap, null);  // todo: judge sampleDepdtBaseScore
                            if (hasLabel) {
                                if (labels == null) {
                                    labels = new double[linfo.length];
                                }
                                for (int i = 0; i < linfo.length; i++) {
                                    labels[i] = Float.parseFloat(linfo[i]);
                                }

                                loss += weight * loss(fmap, labels, null);
                                if (needEval) {
                                    if (testData == null) {
                                        testData = new PredictCoreData(null, linfo.length);
                                    }
                                    testData.addPredict(predicts, labels, weight);
                                }
                            }

                            if (saveMode == ResultSaveMode.PREDICT_RESULT_ONLY ||
                                    saveMode == ResultSaveMode.LABEL_AND_PREDICT) {
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
                                StringBuilder sbx = new StringBuilder();
                                for (int i = 0; i < predicts.length - 1; i++) {
                                    sbx.append(modelName).append("_label_").append(i).append(dataParams.delim.feature_name_val_delim).
                                            append(predicts[i]).append(dataParams.delim.features_delim);
                                }
                                sbx.append(modelName).append("_label_").append(predicts.length - 1).
                                        append(dataParams.delim.feature_name_val_delim).append(predicts[predicts.length - 1]);
                                writer.println(xsplits[0] + dataParams.delim.x_delim + xsplits[1] + dataParams.delim.x_delim +
                                        xsplits[2] + dataParams.delim.features_delim +
                                        sbx.toString());
                            } else {
                                throw new YtkLearnException(modelName + " doesn't support save mode:" + saveMode);
                            }

                        } else {
                            Object otherinfo = null;
                            if (isGbst(modelName)) {
                                if (this instanceof GBMLROnlinePredictor || this instanceof GBSDTOnlinePredictor) {
                                    if (config.getBoolean("sample_dependent_base_prediction")) {
                                        otherinfo = new Float(xsplits[3]);
                                    }
                                }
                            }
                            double predict = predict(fmap, otherinfo);

                            if (hasLabel) {
                                label = Float.parseFloat(linfo[0]);
                                loss += weight * loss(fmap, label, otherinfo); // score not predict?
                                if (needEval) {
                                    if (testData == null) {
                                        testData = new PredictCoreData(null, 1);
                                    }
                                    testData.addPredict(predict, label, weight);
                                }
                            }

                            if (saveMode == ResultSaveMode.PREDICT_RESULT_ONLY ||
                                    saveMode == ResultSaveMode.LABEL_AND_PREDICT) {
                                if (predictType == PredictType.PREDICT_VALUE) {
                                    if (saveMode == ResultSaveMode.PREDICT_RESULT_ONLY) {
                                        writer.println(predict);
                                    } else {
                                        writer.println(xsplits[1] + dataParams.delim.x_delim + predict);
                                    }
                                } else {
                                    double[] predicts = ((ITreePredictor)this).predictLeaf(fmap);
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
                                }

                            } else if (saveMode == ResultSaveMode.PREDICT_AS_FEATURE) {
                                if (predictType == PredictType.PREDICT_VALUE) {
                                    writer.println(xsplits[0] + dataParams.delim.x_delim +
                                            xsplits[1] + dataParams.delim.x_delim +
                                            xsplits[2] + dataParams.delim.features_delim +
                                            modelName + "_label_" + dataParams.delim.feature_name_val_delim + predict);
                                } else {
                                    writer.println(xsplits[0] + dataParams.delim.x_delim +
                                            xsplits[1] + dataParams.delim.x_delim +
                                            xsplits[2] + dataParams.delim.features_delim +
                                            ((ITreePredictor)this).leafFeatures(fmap, dataParams.delim.features_delim,
                                                    dataParams.delim.feature_name_val_delim));
                                }

                            } else {
                                throw new Exception(modelName + " doesn't support save mode:" + saveMode);
                            }
                        }

                        realcnt ++;
                        weightCnt += weight;
                    } catch (Exception e) {
                        errorNum++;
                        LOG.error("[ERROR] error format:" + line + "\n" +
                                ", local error total number:" + errorNum +
                                ", max error tol:" + maxErrorTol +
                                ", has read real lines:" + realcnt +
                                ", weight lines:" + weightCnt);
                        if (errorNum > maxErrorTol) {
                            LOG.error("[ERROR] error number:" + errorNum +
                                    " > " + "max tol:" + maxErrorTol);
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
}
