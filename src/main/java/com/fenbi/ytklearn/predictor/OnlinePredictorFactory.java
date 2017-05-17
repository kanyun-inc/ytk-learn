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

import java.io.Reader;

/**
 * @author xialong
 */

public final class OnlinePredictorFactory {
    public static OnlinePredictor createOnlinePredictor(String modelName, String configPath) throws Exception {
        if (modelName.equalsIgnoreCase("linear")) {
            return new LinearOnlinePredictor(configPath);
        } else if (modelName.equalsIgnoreCase("fm")) {
            return new FMOnlinePredictor(configPath);
        } else if (modelName.equalsIgnoreCase("ffm")) {
            return new FFMOnlinePredictor(configPath);
        } else if (modelName.equalsIgnoreCase("gbdt")) {
            return new GBDTOnlinePredictor(configPath);
        } else if (modelName.equalsIgnoreCase("gbmlr")) {
            return new GBMLROnlinePredictor(configPath);
        } else if (modelName.equalsIgnoreCase("gbsdt")) {
            return new GBSDTOnlinePredictor(configPath);
        } else if (modelName.equalsIgnoreCase("multiclass_linear")) {
            return new MulticlassLinearOnlinePredictor(configPath);
        } else if (modelName.equalsIgnoreCase("gbhmlr")) {
            return new GBHMLROnlinePredictor(configPath);
        } else if (modelName.equalsIgnoreCase("gbhsdt")) {
            return new GBHSDTOnlinePredictor(configPath);
        } else {
            throw new Exception("unkonw model name:" + modelName);
        }
    }

    public static OnlinePredictor createOnlinePredictor(String modelName, Reader configReader) throws Exception {
        if (modelName.equalsIgnoreCase("linear")) {
            return new LinearOnlinePredictor(configReader);
        } else if (modelName.equalsIgnoreCase("fm")) {
            return new FMOnlinePredictor(configReader);
        } else if (modelName.equalsIgnoreCase("ffm")) {
            return new FFMOnlinePredictor(configReader);
        } else if (modelName.equalsIgnoreCase("gbdt")) {
            return new GBDTOnlinePredictor(configReader);
        } else if (modelName.equalsIgnoreCase("gbmlr")) {
            return new GBMLROnlinePredictor(configReader);
        } else if (modelName.equalsIgnoreCase("gbsdt")) {
            return new GBSDTOnlinePredictor(configReader);
        } else if (modelName.equalsIgnoreCase("multiclass_linear")) {
            return new MulticlassLinearOnlinePredictor(configReader);
        } else if (modelName.equalsIgnoreCase("gbhmlr")) {
            return new GBHMLROnlinePredictor(configReader);
        } else if (modelName.equalsIgnoreCase("gbhsdt")) {
            return new GBHSDTOnlinePredictor(configReader);
        } else {
            throw new Exception("unkonw model name:" + modelName);
        }
    }
}
