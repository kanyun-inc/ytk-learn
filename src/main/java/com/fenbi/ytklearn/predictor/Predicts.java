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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author xialong
 */

public final class Predicts {
    public static final Logger LOG = LoggerFactory.getLogger(Predicts.class);

    public static void main(String []args) throws Exception {
        String configPath = args[0];
        String modelName = args[1];
        String fileDir = args[2];
        boolean needPyTransform = Boolean.parseBoolean(args[3]);
        String pyTransformScript = args[4];
        String resultSaveMode = args[5];
        String resultFileSuffix = args[6];
        int maxErrorTol = Integer.parseInt(args[7]);
        String evalMetricStr = args[8];
        String predictTypeStr = "";
        if (args.length >= 10) {
            predictTypeStr = args[9];
        }

        OnlinePredictor predictor = OnlinePredictorFactory.createOnlinePredictor(modelName, configPath);
        predictor.batchPredictFromFiles(modelName, fileDir, needPyTransform,
                pyTransformScript, resultSaveMode, resultFileSuffix, maxErrorTol, evalMetricStr, predictTypeStr);
    }
}
