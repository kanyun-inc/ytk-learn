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

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import lombok.Data;

import java.io.File;
import java.io.Serializable;

/**
 * @author xialong
 */

@Data
public class FeatureParams implements Serializable {
    public static final String KEY = "feature.";

    public FeatureHashParams feature_hash;
    public TransformParams transform;
    public int filter_threshold;

    //for gbdt
    public FeatureParams(int filter_threshold) {
        this.feature_hash = null;
        this.transform = null;
        this.filter_threshold = filter_threshold;
    }

    public FeatureParams(Config config, String prefix) {
        feature_hash = new FeatureHashParams(config, prefix + KEY);
        transform = new TransformParams(config, prefix + KEY);
        filter_threshold = config.getInt(prefix + KEY + "filter_threshold");
    }

    public static void main(String []args) {
        Config config = ConfigFactory.parseFile(new File("config/model/linear.conf"));
        FeatureParams params = new FeatureParams(config, "");
        System.out.println(params.toString());
    }

}
