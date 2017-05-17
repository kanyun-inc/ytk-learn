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

import com.fenbi.ytklearn.param.DataParams;
import com.fenbi.ytklearn.utils.CheckUtils;
import com.typesafe.config.Config;
import lombok.Data;

import java.io.Serializable;

/**
 * @author wufan
 * @author xialong
 */

@Data
public class GBDTDataParams extends DataParams implements Serializable {

    public int max_feature_dim;

    public GBDTDataParams(Config config, String prefix) {
        super(config, prefix);
        max_feature_dim = config.getInt(prefix + KEY + "max_feature_dim");
        CheckUtils.check(max_feature_dim >= 1, "[GBDT] max_feature_dim(%d) should >=1", max_feature_dim);
    }

    @Override
    public String toString() {
        return super.toString() + ", max_feature_dim=" + max_feature_dim;
    }

}
