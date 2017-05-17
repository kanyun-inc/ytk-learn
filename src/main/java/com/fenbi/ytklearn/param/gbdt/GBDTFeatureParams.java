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
import com.fenbi.ytklearn.feature.gbdt.approximate.sampler.SampleType;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.fenbi.ytklearn.feature.gbdt.FeatureSplitType;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigException;
import lombok.Data;
import org.apache.commons.lang3.ArrayUtils;

import java.io.Serializable;
import java.util.*;

/**
 * @author wufan
 * @author xialong
 */

@Data
public class GBDTFeatureParams implements Serializable {
    public static final String KEY = "feature.";

    public FeatureSplitType split_Type;

    public boolean enable_missing_value;
    public String featureMissingParams;

    private boolean needFeaAppro;
    private List<Config> feaApproConfList;
    public List<FeatureApproximateParams> featureApproximateParamList;

    public boolean verbose;
    public int filter_threshold;


    @Data
    public static class FeatureApproximateParams implements Serializable {
        private static final String COL_SPLIT = ",";
        private static final String COL_RANGE = "-";

        // cols.length ==1 && cols[0] == -1 mean default cols
        public int[] cols;
        public SampleType type;
        public Map<String, String> params;

        private int[] parseCol(String colConf, Set<Integer> existCols, Map<String, Integer> fName2IndexMap) {
            if (colConf.equalsIgnoreCase("default")) {
                CheckUtils.check(!existCols.contains(-1), "[GBDT] feature approximate config error! default has been set twice");
                existCols.add(-1);
                return new int[]{-1};
            }

            List<Integer> colList = new ArrayList<>(16);
            String[] allCols = colConf.split(COL_SPLIT);

            for (String colField : allCols) {
                colField = colField.trim();
                CheckUtils.check(fName2IndexMap.containsKey(colField), "[GBDT] feature approximate config error! feature(%s) does not exist", colField);
                int col = fName2IndexMap.get(colField);
                CheckUtils.check(!existCols.contains(col), "[GBDT] feature approximate config error!, feature(%s) has been set twice", colField);
                colList.add(col);
                existCols.add(col);
            }
            return ArrayUtils.toPrimitive(colList.toArray(new Integer[colList.size()]));
        }

        public void init(Config config, Set<Integer> configCols, Map<String, Integer> fName2IndexMap, long globalSampleCnt) {

            String typeStr = config.getString("type");
            type = SampleType.valueOfType(typeStr);
            CheckUtils.check(type != null, "[GBDT] feature approximate sample type(%s) invalid", typeStr);

            cols = parseCol(config.getString("cols"), configCols, fName2IndexMap);
            params = new HashMap<>();
            if (type.equals(SampleType.CNT)) {
                int maxCnt = config.getInt("max_cnt");
                CheckUtils.check(maxCnt >= 0, "[GBDT] feature approximate config error! max_cnt(%d) should be >= 0", maxCnt);
                params.put("max_cnt", maxCnt + "");

            } else if (type.equals(SampleType.RATE)) {
                double sampleByRate = config.getDouble("sample_rate");
                CheckUtils.check(sampleByRate >= 0, "[GBDT] feature approximate config error! sample_rate(%f) should be >= 0", sampleByRate);

                int minCnt = config.getInt("min_cnt");
                CheckUtils.check(minCnt >= 0, "[GBDT] feature approximate config error! min_cnt(%d) should be >= 0", minCnt);

                params.put("sample_rate", sampleByRate + "");
                params.put("min_cnt", minCnt + "");

            } else if (type.equals(SampleType.PRECISION)) {
                int dotPrecision = config.getInt("dot_precision");
                CheckUtils.check(dotPrecision >= 0, "[GBDT] feature approximate config error! dot_precision(%d) should be >= 0", dotPrecision);

                boolean useLog = config.getBoolean("use_log");
                boolean useMinMax = config.getBoolean("use_min_max");
                params.put("dot_precision", dotPrecision + "");
                params.put("use_log", useLog + "");
                params.put("use_min_max", useMinMax + "");

            } else if (type.equals(SampleType.QUANTILE)) {
                int maxBinCnt = config.getInt("max_cnt");
                CheckUtils.check(maxBinCnt >= 1, "[GBDT] feature approximate config error! max_bin_cnt(%d) should be >= 1", maxBinCnt);

                int quantile_approximate_bin_factor;
                try {
                    quantile_approximate_bin_factor = config.getInt("quantile_approximate_bin_factor");
                } catch (ConfigException.Missing e) {
                    quantile_approximate_bin_factor = Constants.QUNANTILE_APPROXIMATE_BIN_FACTOR;
                }
                CheckUtils.check(quantile_approximate_bin_factor >= 1,
                        "[GBDT] feature approximate config error! quantile_approximate_bin_factor(%d) should be >= 1, recommend 8", quantile_approximate_bin_factor);

                boolean use_sample_weight;
                try {
                    use_sample_weight = config.getBoolean("use_sample_weight");
                } catch (ConfigException.Missing e) {
                    use_sample_weight = false;
                }

                // down weight sample value cnt
                double alpha;
                try {
                    alpha = config.getDouble("alpha");
                } catch (ConfigException.Missing e) {
                    alpha = 1.0;
                }
                CheckUtils.check(alpha >= 0 && alpha <= 1, "[GBDT] feature approximate config error! alpha(%f) should belong to [0, 1]", alpha);

                params.put("max_cnt", maxBinCnt + "");
                params.put("quantile_approximate_bin_factor", quantile_approximate_bin_factor + "");
                params.put("use_sample_weight", use_sample_weight + "");
                params.put("global_cnt", globalSampleCnt + "");
                params.put("alpha", alpha + "");

            } else if (type.equals(SampleType.NO_SAMPLE)) {

            }
        }
    }

    public GBDTFeatureParams(Config config, String prefix) {
        // default split type: right
        String splitTypeStr;
        try {
            splitTypeStr = config.getString(prefix + KEY + "split_type");
        } catch (ConfigException.Missing e) {
            splitTypeStr = FeatureSplitType.MEAN.getName();
        }
        split_Type = FeatureSplitType.valueOfFeatureSplitType(splitTypeStr);
        CheckUtils.check(split_Type != null, "[GBDT] split type(%s) invalid", splitTypeStr);

//        enable_missing_value = config.getBoolean(prefix + KEY + "enable_missing_value");
        enable_missing_value = true;
        if (enable_missing_value) {
            featureMissingParams = config.getString(prefix + KEY + "missing_value" );
        }

        String treeMakerTypeStr = config.getString(prefix + "optimization.tree_maker");
        TreeMakerType tree_maker_type = TreeMakerType.valueOfType(treeMakerTypeStr);
        CheckUtils.check(tree_maker_type != null, "[GBDT] tree maker type(%s) invalid, data or feature", treeMakerTypeStr);
        needFeaAppro = tree_maker_type == TreeMakerType.DATA_PARALLEL;
        if (needFeaAppro) {
            feaApproConfList = (List<Config>) config.getConfigList(prefix + KEY + "approximate");
        }

        try {
            verbose = config.getBoolean(prefix + KEY + "verbose");
        } catch (ConfigException.Missing e) {
//            verbose = config.getBoolean("verbose");
            verbose = false;
        }

        filter_threshold = config.getInt(prefix + KEY + "filter_threshold");
    }

    // init feature approximate params, called in main thread
    public void init(Map<String, Integer> fName2IndexMap, long globalSampleCnt) {
        if (needFeaAppro) {
            CheckUtils.check(fName2IndexMap != null, "[GBDT] fName2IndexMap shouldn't be null");
            featureApproximateParamList = new ArrayList<>(feaApproConfList.size());
            Set<Integer> existCols = new HashSet<>(16);
            for (Config feaConf : feaApproConfList) {
                FeatureApproximateParams feaAppr = new FeatureApproximateParams();
                feaAppr.init(feaConf, existCols, fName2IndexMap, globalSampleCnt);
                featureApproximateParamList.add(feaAppr);
            }
            feaApproConfList = null;
        }
    }

}
