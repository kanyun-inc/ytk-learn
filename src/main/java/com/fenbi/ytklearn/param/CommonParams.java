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

package com.fenbi.ytklearn.param;

import com.fenbi.ytklearn.utils.CheckUtils;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import lombok.Data;

import java.io.File;
import java.io.Serializable;

/**
 * @author xialong
 */

@Data
public class CommonParams implements Serializable {
    public DataParams dataParams;
    public FeatureParams featureParams;
    public LossParams lossParams;
    public ModelParams modelParams;
    public LineSearchParams lsParams;
    public boolean verbose;

    public static CommonParams loadParams(Config config) {

        CommonParams params = new CommonParams();

        params.dataParams = new DataParams(config, "");
        params.featureParams = new FeatureParams(config, "");
        params.lossParams = new LossParams(config, "");
        params.modelParams = new ModelParams(config, "");
        String optimizer = config.getString("optimization.optimizer");

        params.lsParams = new LineSearchParams(config, "optimization.");
        params.verbose = config.getBoolean("verbose");

        CheckUtils.check(optimizer.equals("line_search"), "optimization.optimizer only support:%s", "line_search");

        return params;
    }

    public static void main(String []args) {
        Config config = ConfigFactory.parseFile(new File("config/model/linear.conf"));
        System.out.println(CommonParams.loadParams(config));
    }
}
