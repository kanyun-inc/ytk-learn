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

package com.fenbi.ytklearn.loss;

/**
 * @author xialong
 */
public class LossFunctions {

    public static ILossFunction createLossFunction(String funcName) throws Exception {
        if (funcName.equalsIgnoreCase("sigmoid")) {
            return new SigmoidFunction();
        } else if (funcName.equalsIgnoreCase("sigmoid_cross_entropy")) {
            return new SigmoidFunction();
        } else if (funcName.equalsIgnoreCase("l2")) {
            return new L2Function();
        } else if (funcName.equalsIgnoreCase("hinge")) {
            return new HingeFunction();
        } else if (funcName.equalsIgnoreCase("smooth_hinge")) {
            return new SmoothHingeFunction();
        } else if (funcName.equalsIgnoreCase("l2_hinge")) {
            return new L2HingeFunction();
        } else if (funcName.equalsIgnoreCase("exponential")) {
            return new ExponentialFunction();
        } else if (funcName.equalsIgnoreCase("l1")) {
            return new L1Function();
        } else if (funcName.equalsIgnoreCase("poisson")) {
            return new PoissonFunction();
        } else if (funcName.equalsIgnoreCase("mape")) {
            return new MAPEFunction();
        } else if (funcName.equalsIgnoreCase("inv_mape")) {
            return new InvMAPEFunction();
        } else if (funcName.equalsIgnoreCase("smape")) {
            return new SMAPEFunction();
        } else if (funcName.equalsIgnoreCase("softmax")) {
            return new SoftmaxFunction();
        } else if (funcName.equalsIgnoreCase("softmax_cross_entropy")) {
            return new SoftmaxFunction();
        } else if (funcName.equalsIgnoreCase("multiclass_hinge")) {
            return new MulticlassHingeFunction();
        } else if (funcName.equalsIgnoreCase("multiclass_l2_hinge")) {
            return new MulticlassL2HingeFunction();
        } else if (funcName.equalsIgnoreCase("multiclass_smooth_hinge")) {
            return new MulticlassSmoothHingeFunction();
        } else if (funcName.equalsIgnoreCase("huber")) {
            String []info = funcName.split("@");
            double delta = info.length > 1 ? Double.parseDouble(info[1]) : 0.5;
            return new HuberFunction(delta);
        } else if (funcName.equalsIgnoreCase("hsoftmax")) {
            return new HSoftmaxFunction();
        } else if (funcName.equalsIgnoreCase("hsoftmax_cross_entropy")) {
            return new HSoftmaxFunction();
        } else{
            throw new Exception("Unsupport function name:" + funcName);
        }
    }

    public static boolean pureClassification(String funcName) {
        return funcName.equalsIgnoreCase("sigmoid") || funcName.equalsIgnoreCase("softmax") ||
                funcName.equalsIgnoreCase("hinge") || funcName.equalsIgnoreCase("smooth_hinge") ||
                funcName.equalsIgnoreCase("l2_hinge") || funcName.equalsIgnoreCase("multiclass_l2_hinge") ||
                funcName.equalsIgnoreCase("exponential") || funcName.equalsIgnoreCase("multiclass_hinge") ||
                funcName.equalsIgnoreCase("multiclass_smooth_hinge") || funcName.equalsIgnoreCase("hsoftmax");
    }
}
