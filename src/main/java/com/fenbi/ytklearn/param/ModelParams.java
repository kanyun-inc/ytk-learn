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
public class ModelParams implements Serializable {
    public static final String KEY = "model.";
    public String data_path;
    public String delim;
    public boolean need_dict;
    public String dict_path;
    public int dump_freq;
    public boolean need_bias;
    public String bias_feature_name;
    public boolean continue_train;

    public ModelParams(Config config, String prefix) {
        data_path = config.getString(prefix + KEY + "data_path");
        delim = config.getString(prefix + KEY + "delim");
        need_dict = config.getBoolean(prefix + KEY + "need_dict");
        dict_path = config.getString(prefix + KEY + "dict_path");
        dump_freq = config.getInt(prefix + KEY + "dump_freq");
        need_bias = config.getBoolean(prefix + KEY + "need_bias");
        bias_feature_name = config.getString(prefix + KEY + "bias_feature_name");
        continue_train = config.getBoolean(prefix + KEY + "continue_train");
    }

    public static void main(String []args) {
        Config config = ConfigFactory.parseFile(new File("config/model/linear.conf"));
        ModelParams params = new ModelParams(config, "");
        System.out.println(params.toString());
    }
}
