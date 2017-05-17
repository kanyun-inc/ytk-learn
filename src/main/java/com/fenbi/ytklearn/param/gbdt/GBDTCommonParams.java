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

import com.fenbi.ytklearn.utils.CheckUtils;
import com.typesafe.config.Config;
import lombok.Data;

import java.io.Serializable;

/**
 * @author wufan
 * @author xialong
 */

@Data
public class GBDTCommonParams implements Serializable {

    public boolean verbose;
    public GBDTDataParams dataParams;
    public GBDTModelParams modelParams;
    public GBDTFeatureParams featureParams;
    public GBDTOptimizationParams optimizationParams;

    public static GBDTCommonParams loadParams(Config config) throws Exception {
        GBDTCommonParams params = new GBDTCommonParams();
        params.verbose = config.getBoolean("verbose");

        params.dataParams = new GBDTDataParams(config, "");
        params.modelParams = new GBDTModelParams(config, "");
        params.featureParams = new GBDTFeatureParams(config, "");
        params.optimizationParams = new GBDTOptimizationParams(config, "");

        checkParams(params);
        return params;
    }

    private static void checkParams(GBDTCommonParams params) {

        String trainDataPath = params.dataParams.train.data_path;
        String testDataPath = params.dataParams.test.data_path;
        // train phase
        if (!params.optimizationParams.just_evaluate) {
            CheckUtils.check(trainDataPath != null && trainDataPath.length() > 0,
                    "[GBDT] train data path is empty");

            boolean watchTest = params.optimizationParams.watch_test;
            CheckUtils.check(!(watchTest && (testDataPath == null || testDataPath.trim().length() == 0)),
                    "[GBDT] watch_test=true but test_data_path is empty");
        // just eval
        } else {
            CheckUtils.check((trainDataPath != null && trainDataPath.length() > 0) ||
                            (testDataPath != null) && testDataPath.trim().length() == 0,
                    "[GBDT] train and test data path is both empty");
        }
    }
}
